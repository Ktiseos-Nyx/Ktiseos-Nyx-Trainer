"""
Unit tests for regex-based network args:
  - exclude_patterns / include_patterns
  - reg_dims / reg_lrs
  - Preset fallback with network arg priority
  - exclude_patterns precedence over exclude_name
"""
import unittest
import logging

import torch
import torch.nn as nn

from lycoris import create_lycoris, LycorisNetwork
from lycoris.logging import logger

logger.setLevel(logging.ERROR)


def reset_globals():
    LycorisNetwork.apply_preset(
        {
            "enable_conv": True,
            "target_module": [
                "Linear",
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "GroupNorm",
                "LayerNorm",
            ],
            "target_name": [],
            "lora_prefix": "lycoris",
            "module_algo_map": {},
            "name_algo_map": {},
            "use_fnmatch": False,
            "exclude_name": [],
            "exclude_patterns": None,
            "include_patterns": None,
            "network_reg_dims": None,
            "network_reg_lrs": None,
        }
    )


class SimpleNet(nn.Module):
    """A simple network with named submodules for testing regex filtering."""

    def __init__(self, dim=16):
        super().__init__()
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, dim * 4)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return x


class LycorisRegexArgsTests(unittest.TestCase):
    """Tests for exclude_patterns, include_patterns, reg_dims, reg_lrs."""

    def _get_lora_names(self, network):
        return sorted([lora.lora_name for lora in network.loras])

    def _get_original_names(self, network):
        return sorted([getattr(lora, 'original_name', '') for lora in network.loras])

    # ── exclude_patterns ──────────────────────────────────────────────

    def test_exclude_patterns_filters_modules(self):
        """Modules matching exclude_patterns should not get LoRA adapters."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*mlp.*"],
            )
            names = self._get_original_names(lycoris_net)
            # mlp_fc1 and mlp_fc2 should be excluded
            for n in names:
                self.assertNotIn("mlp", n, f"Module '{n}' should have been excluded")
            # attn modules should still be present
            attn_names = [n for n in names if "attn" in n]
            self.assertGreater(len(attn_names), 0, "attn modules should be present")
        finally:
            reset_globals()

    def test_exclude_patterns_empty_list_excludes_nothing(self):
        """An empty exclude_patterns list should not exclude anything."""
        try:
            net_all = SimpleNet()
            lycoris_all = create_lycoris(
                net_all, 1, linear_dim=4, linear_alpha=1,
            )
            net_empty = SimpleNet()
            lycoris_empty_exclude = create_lycoris(
                net_empty, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[],
            )
            self.assertEqual(
                len(lycoris_all.loras),
                len(lycoris_empty_exclude.loras),
            )
        finally:
            reset_globals()

    # ── include_patterns overrides exclude ─────────────────────────────

    def test_include_overrides_exclude(self):
        """include_patterns should override exclude_patterns."""
        try:
            net = SimpleNet()
            # Exclude all attn, but include attn_q back
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*attn.*"],
                include_patterns=[r".*attn_q"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            # Only attn_q should survive
            self.assertEqual(len(attn_names), 1, f"Expected 1 attn module, got: {attn_names}")
            self.assertIn("attn_q", attn_names[0])
        finally:
            reset_globals()

    def test_include_overrides_exclude_name(self):
        """include_patterns should also override exclude_name (TARGET_EXCLUDE_NAME)."""
        try:
            # Use exclude_name via preset
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*attn.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                include_patterns=[r".*attn_q"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            # attn_q should be included despite exclude_name
            self.assertEqual(len(attn_names), 1, f"Expected 1 attn module, got: {attn_names}")
            self.assertIn("attn_q", attn_names[0])
        finally:
            reset_globals()

    # ── exclude_patterns precedence over exclude_name ──────────────────

    def test_exclude_patterns_overrides_exclude_name(self):
        """When exclude_patterns is set, exclude_name should be ignored."""
        try:
            # Preset sets exclude_name to exclude mlp
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*mlp.*"],
            })
            net = SimpleNet()
            # exclude_patterns only excludes conv - mlp should now be INCLUDED
            # because exclude_patterns takes precedence over exclude_name
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*conv.*"],
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            conv_names = [n for n in names if "conv" in n]
            # mlp should be present (exclude_name ignored)
            self.assertGreater(len(mlp_names), 0, "mlp modules should be present when exclude_patterns overrides exclude_name")
            # conv should be excluded by exclude_patterns
            self.assertEqual(len(conv_names), 0, "conv modules should be excluded")
        finally:
            reset_globals()

    def test_exclude_name_used_when_no_exclude_patterns(self):
        """When exclude_patterns is not set, exclude_name should still work."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*mlp.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                # No exclude_patterns set
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            self.assertEqual(len(mlp_names), 0, "mlp modules should be excluded by exclude_name")
        finally:
            reset_globals()

    # ── reg_dims ──────────────────────────────────────────────────────

    def test_reg_dims_overrides_module_dim(self):
        """reg_dims should override the default dim for matching modules."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                reg_dims={r".*attn.*": 32},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"Module '{orig}' should have lora_dim=32, got {lora.lora_dim}")
                elif hasattr(lora, 'lora_dim') and "norm" not in lora.lora_name:
                    self.assertEqual(lora.lora_dim, 4, f"Module '{orig}' should have default lora_dim=4, got {lora.lora_dim}")
        finally:
            reset_globals()

    def test_reg_dims_multiple_patterns(self):
        """Multiple reg_dims patterns should each apply to their matching modules."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                reg_dims={r".*attn.*": 16, r".*mlp.*": 64},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 16, f"'{orig}' should have lora_dim=16")
                elif "mlp" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 64, f"'{orig}' should have lora_dim=64")
        finally:
            reset_globals()

    # ── reg_lrs ──────────────────────────────────────────────────────

    def test_reg_lrs_stored_on_network(self):
        """reg_lrs should be stored on the network object for optimizer use."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                reg_lrs={r".*attn.*": 1e-4, r".*mlp.*": 5e-5},
            )
            self.assertIsNotNone(lycoris_net.reg_lrs)
            self.assertEqual(len(lycoris_net.reg_lrs), 2)
        finally:
            reset_globals()

    # ── original_name attribute ───────────────────────────────────────

    def test_original_name_set_on_loras(self):
        """All created lora modules should have original_name attribute."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            for lora in lycoris_net.loras:
                self.assertTrue(
                    hasattr(lora, 'original_name'),
                    f"Module '{lora.lora_name}' missing original_name attribute",
                )
                self.assertIsNotNone(lora.original_name)
                self.assertNotEqual(lora.original_name, "")
        finally:
            reset_globals()

    # ── Preset fallback ───────────────────────────────────────────────

    def test_preset_exclude_patterns_fallback(self):
        """exclude_patterns from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*mlp.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                # No exclude_patterns in network args
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            self.assertEqual(len(mlp_names), 0, "mlp modules should be excluded by preset exclude_patterns")
        finally:
            reset_globals()

    def test_preset_include_patterns_fallback(self):
        """include_patterns from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*attn.*"],
                "include_patterns": [r".*attn_q"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            self.assertEqual(len(attn_names), 1, f"Only attn_q should survive, got: {attn_names}")
        finally:
            reset_globals()

    def test_preset_reg_dims_fallback(self):
        """reg_dims from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_dims": {r".*attn.*": 32},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"'{orig}' should have lora_dim=32 from preset")
        finally:
            reset_globals()

    def test_preset_reg_lrs_fallback(self):
        """reg_lrs from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_lrs": {r".*attn.*": 1e-4},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            self.assertIsNotNone(lycoris_net.reg_lrs)
            self.assertIn(r".*attn.*", lycoris_net.reg_lrs)
        finally:
            reset_globals()

    # ── Network args override preset ──────────────────────────────────

    def test_network_args_override_preset_exclude_patterns(self):
        """Network arg exclude_patterns should override preset."""
        try:
            # Preset excludes attn
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*attn.*"],
            })
            net = SimpleNet()
            # Network arg excludes mlp instead
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*mlp.*"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            mlp_names = [n for n in names if "mlp" in n]
            # attn should be present (preset overridden)
            self.assertGreater(len(attn_names), 0, "attn should be present (preset overridden)")
            # mlp should be excluded (network arg applied)
            self.assertEqual(len(mlp_names), 0, "mlp should be excluded (network arg)")
        finally:
            reset_globals()

    def test_network_args_override_preset_reg_dims(self):
        """Network arg reg_dims should override preset."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_dims": {r".*attn.*": 8},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                reg_dims={r".*attn.*": 64},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 64, f"'{orig}' should have lora_dim=64 from network arg, not 8 from preset")
        finally:
            reset_globals()

    # ── Combined scenarios ────────────────────────────────────────────

    def test_exclude_and_reg_dims_together(self):
        """exclude_patterns and reg_dims should work together."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*conv.*"],
                reg_dims={r".*attn.*": 32},
            )
            names = self._get_original_names(lycoris_net)
            conv_names = [n for n in names if "conv" in n]
            self.assertEqual(len(conv_names), 0, "conv should be excluded")
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"'{orig}' should have lora_dim=32")
        finally:
            reset_globals()

    def test_all_features_from_preset(self):
        """All new features should work when set entirely from preset."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*conv.*", r".*norm.*"],
                "include_patterns": [r".*norm.*"],  # re-include norm
                "network_reg_dims": {r".*attn.*": 16},
                "network_reg_lrs": {r".*mlp.*": 2e-4},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                train_norm=True,
            )
            names = self._get_original_names(lycoris_net)
            # conv should be excluded
            conv_names = [n for n in names if "conv" in n]
            self.assertEqual(len(conv_names), 0, "conv should be excluded")
            # norm should be included (include overrides exclude)
            norm_names = [n for n in names if "norm" in n]
            self.assertGreater(len(norm_names), 0, "norm should be included via include_patterns")
            # attn should have lora_dim=16
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 16)
            # reg_lrs should be stored
            self.assertIsNotNone(lycoris_net.reg_lrs)
        finally:
            reset_globals()


if __name__ == "__main__":
    unittest.main()
