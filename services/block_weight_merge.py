"""
Block-weighted (MBW) checkpoint merge engine.

Clean-room reimplementation of SuperMerger's block-weighted merge.
Loads/saves via the vendored Kohya sd-scripts. No SuperMerger source
is copied — only the block-mapping algorithm and preset weight arrays
(which are facts/data, not code).

Modes:
  - Weight:    A·(1−α) + B·α
  - Add:       A + (B − C)·α
  - Triple:    A·(1−α−β) + B·α + C·β
  - Twice:     merge(A, B, α) → merge(AB, C, β)
"""

import json
import logging
import os
import re
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Block definitions — 26-block (SD1.5) and 20-block (SDXL)
#
# These are *facts* (layout definitions), not copyrighted source.
# Both layouts are well-known SuperMerger conventions.
# ═══════════════════════════════════════════════════════════════

BLOCK26_NAMES = [
    "BASE",   # 0
    "IN00", "IN01", "IN02", "IN03", "IN04", "IN05",  # 1-6
    "IN06", "IN07", "IN08", "IN09", "IN10", "IN11",  # 7-12
    "MID",    # 13
    "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05",  # 14-19
    "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11",  # 20-25
]

BLOCK20_NAMES = [
    "BASE",   # 0
    "IN00", "IN01", "IN02", "IN03", "IN04", "IN05",  # 1-6
    "IN06", "IN07", "IN08",           # 7-9
    "MID",    # 10
    "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05",  # 11-16
    "OUT06", "OUT07", "OUT08",        # 17-19
]

NON_UNET_PREFIXES = [
    "first_stage_model",
    "cond_stage_model",
    "conditioner.",
]

# ═══════════════════════════════════════════════════════════════
# Block-from-key mapping
# ═══════════════════════════════════════════════════════════════

RE_SD15_INPUT = re.compile(r"model\.diffusion_model\.input_blocks\.(\d+)\.")
RE_SD15_OUTPUT = re.compile(r"model\.diffusion_model\.output_blocks\.(\d+)\.")
RE_SDXL_INPUT = re.compile(r"(?:model\.diffusion_model\.)?input_blocks\.(\d+)\.")
RE_SDXL_OUTPUT = re.compile(r"(?:model\.diffusion_model\.)?output_blocks\.(\d+)\.")


def block_from_key(key: str, is_sdxl: bool = False) -> int:
    """Map a state-dict key to its 0-based block index.

    Returns -1 for non-UNet keys (TE / VAE) — these use base_alpha.
    """
    # Non-UNet → base_alpha territory
    for prefix in NON_UNET_PREFIXES:
        if prefix in key:
            return -1

    if is_sdxl:
        return _block_from_key_sdxl(key)
    return _block_from_key_sd15(key)


def _block_from_key_sd15(key: str) -> int:
    # time_embed, conv_in, out → BASE (0)
    if "time_embed" in key or key.endswith("input_blocks.0.0.weight") or "out." in key:
        return 0

    m = RE_SD15_INPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= 12:
            return idx  # IN00=1, IN01=2, …, IN11=12

    if "middle_block" in key:
        return 13  # MID

    m = RE_SD15_OUTPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 11:
            return 14 + idx  # OUT00=14, …, OUT11=25

    return 0  # fallback → BASE


def _block_from_key_sdxl(key: str) -> int:
    # time_embed, label_emb, conv_in, out → BASE (0)
    if any(x in key for x in ("time_embed", "label_emb")) or "out." in key:
        return 0

    m = RE_SDXL_INPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 0 == idx:
            return 0  # conv_in → BASE
        if 1 <= idx <= 9:
            return idx  # IN00=1, …, IN08=9

    if "middle_block" in key:
        return 10  # MID

    m = RE_SDXL_OUTPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 8:
            return 11 + idx  # OUT00=11, …, OUT08=19

    return 0  # fallback → BASE


def detect_architecture(state_dict_keys: list[str]) -> bool:
    """Detect whether a state dict is SDXL or SD1.5 based on key patterns.

    Returns True if SDXL.
    """
    sdxl_hints = ("conditioner.", "label_emb", "input_blocks.8")
    for key in state_dict_keys:
        for hint in sdxl_hints:
            if hint in key:
                return True
    return False


# ═══════════════════════════════════════════════════════════════
# Merge modes
# ═══════════════════════════════════════════════════════════════

def _get_block_weight(key: str, block_weights: list[float], is_sdxl: bool, base_alpha: float) -> float:
    """Get the per-key blend weight from block weights + base_alpha."""
    block_idx = block_from_key(key, is_sdxl)
    if block_idx < 0:
        return base_alpha
    if block_idx < len(block_weights):
        return block_weights[block_idx]
    return base_alpha


def _load_safetensors(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a safetensors file and return its state dict."""
    return load_file(path, device=device)


def merge_weight(
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    block_weights: list[float],
    base_alpha: float = 0.5,
    is_sdxl: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Weight mode: A·(1−α) + B·α where α is per-block.

    Args:
        model_a: State dict for model A
        model_b: State dict for model B
        block_weights: Per-block alpha values (26 for SD1.5, 20 for SDXL)
        base_alpha: Fallback alpha for non-UNet keys (TE, VAE)
        is_sdxl: Whether model is SDXL (auto-detect if None)

    Returns:
        Merged state dict
    """
    if is_sdxl is None:
        is_sdxl = detect_architecture(list(model_a.keys()))

    merged = {}
    all_keys = set(model_a.keys()) | set(model_b.keys())

    for key in all_keys:
        a = model_a.get(key)
        b = model_b.get(key)

        if a is None:
            merged[key] = b.clone()
            continue
        if b is None:
            merged[key] = a.clone()
            continue

        alpha = _get_block_weight(key, block_weights, is_sdxl, base_alpha)
        a = a.float()
        b = b.float()
        merged[key] = (a * (1 - alpha) + b * alpha).to(a.dtype)

    return merged


def merge_add(
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    model_c: dict[str, torch.Tensor],
    block_weights: list[float],
    base_alpha: float = 1.0,
    is_sdxl: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Add mode: A + (B − C)·α"""
    if is_sdxl is None:
        is_sdxl = detect_architecture(list(model_a.keys()))

    merged = {}
    all_keys = set(model_a.keys()) | set(model_b.keys()) | set(model_c.keys())

    for key in all_keys:
        a = model_a.get(key)
        b = model_b.get(key)
        c = model_c.get(key)

        if a is None:
            merged[key] = (b or c).clone()
            continue

        if b is None or c is None:
            merged[key] = a.clone()
            continue

        alpha = _get_block_weight(key, block_weights, is_sdxl, base_alpha)
        a_f, b_f, c_f = a.float(), b.float(), c.float()
        merged[key] = (a_f + (b_f - c_f) * alpha).to(a.dtype)

    return merged


def merge_triple(
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    model_c: dict[str, torch.Tensor],
    block_weights_b: list[float],
    block_weights_c: list[float],
    base_alpha: float = 0.333,
    is_sdxl: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Triple mode: A·(1−α−β) + B·α + C·β"""
    if is_sdxl is None:
        is_sdxl = detect_architecture(list(model_a.keys()))

    merged = {}
    all_keys = set(model_a.keys()) | set(model_b.keys()) | set(model_c.keys())

    for key in all_keys:
        a = model_a.get(key)
        b = model_b.get(key)
        c = model_c.get(key)

        if a is None and b is None:
            merged[key] = c.clone()
            continue
        if a is None:
            merged[key] = b.clone()
            continue
        if b is None and c is None:
            merged[key] = a.clone()
            continue

        alpha = _get_block_weight(key, block_weights_b, is_sdxl, base_alpha)
        beta = _get_block_weight(key, block_weights_c, is_sdxl, base_alpha)

        a_f = a.float()
        b_f = b.float() if b is not None else torch.zeros_like(a_f)
        c_f = c.float() if c is not None else torch.zeros_like(a_f)

        merged[key] = (a_f * (1 - alpha - beta) + b_f * alpha + c_f * beta).to(a.dtype)

    return merged


def merge_twice(
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    model_c: dict[str, torch.Tensor],
    block_weights_ab: list[float],
    block_weights_bc: list[float],
    base_alpha_ab: float = 0.5,
    base_alpha_bc: float = 0.5,
    is_sdxl: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Twice mode: merge(A, B, α) → merge(AB, C, β)"""
    ab = merge_weight(model_a, model_b, block_weights_ab, base_alpha_ab, is_sdxl)
    return merge_weight(ab, model_c, block_weights_bc, base_alpha_bc, is_sdxl)


# ═══════════════════════════════════════════════════════════════
# Save helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(state_dict: dict[str, torch.Tensor], output_path: str, metadata: Optional[dict] = None) -> None:
    """Save a merged checkpoint as safetensors."""
    if metadata is None:
        metadata = {"format": "pt"}
    metadata["block_weight_merge"] = "true"

    save_file(state_dict, output_path, metadata=metadata)
    logger.info("Saved merged checkpoint to %s", output_path)


def load_checkpoint(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a checkpoint safetensors file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info("Loading checkpoint from %s", path)
    return _load_safetensors(path, device)
