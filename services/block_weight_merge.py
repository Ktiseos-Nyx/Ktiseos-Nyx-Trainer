"""
Block-weighted (MBW) checkpoint merge engine.

Clean-room reimplementation of SuperMerger's block-weighted merge.
Supports SD1.5 (26-block), SDXL (20-block), and Anima DiT (35-block).

Loads/saves via safetensors directly. No SuperMerger source is copied —
only the block-mapping algorithm and preset weight arrays (facts/data).

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
# Block definitions
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

ANIMA_BLOCK_NAMES = [
    *[f"B{i:02d}" for i in range(28)],         # 0-27  net.blocks
    *[f"LLM{i}" for i in range(6)],             # 28-33 llm_adapter.blocks
    "OTHER",                                     # 34     unlayered
]

NON_UNET_PREFIXES = [
    "first_stage_model",
    "cond_stage_model",
    "conditioner.",
]

ANIMA_NON_DIT_PREFIXES = [
    "vae.",
    "text_encoder_model.",
]

# ═══════════════════════════════════════════════════════════════
# Key regex patterns
# ═══════════════════════════════════════════════════════════════

RE_SD15_INPUT = re.compile(r"model\.diffusion_model\.input_blocks\.(\d+)\.")
RE_SD15_OUTPUT = re.compile(r"model\.diffusion_model\.output_blocks\.(\d+)\.")
RE_SDXL_INPUT = re.compile(r"(?:model\.diffusion_model\.)?input_blocks\.(\d+)\.")
RE_SDXL_OUTPUT = re.compile(r"(?:model\.diffusion_model\.)?output_blocks\.(\d+)\.")
RE_ANIMA_BLOCK      = re.compile(r"(?:net|model\.diffusion_model)\.blocks\.(\d+)\.")
RE_ANIMA_LLM_BLOCK  = re.compile(r"(?:net|model\.diffusion_model)\.llm_adapter\.blocks\.(\d+)\.")


def block_from_key(key: str, arch: str = "sd15") -> int:
    """Map a state-dict key to its 0-based block index.

    Returns -1 for non-model keys (TE / VAE) — these use base_alpha.
    """
    prefixes = ANIMA_NON_DIT_PREFIXES if arch == "anima" else NON_UNET_PREFIXES
    for prefix in prefixes:
        if prefix in key:
            return -1

    if arch == "sdxl":
        return _block_from_key_sdxl(key)
    if arch == "anima":
        return _block_from_key_anima(key)
    return _block_from_key_sd15(key)


def _block_from_key_sd15(key: str) -> int:
    if "time_embed" in key or key.endswith("input_blocks.0.0.weight") or "out." in key:
        return 0
    m = RE_SD15_INPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= 12:
            return idx
    if "middle_block" in key:
        return 13
    m = RE_SD15_OUTPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 11:
            return 14 + idx
    return 0


def _block_from_key_sdxl(key: str) -> int:
    if any(x in key for x in ("time_embed", "label_emb")) or "out." in key:
        return 0
    m = RE_SDXL_INPUT.search(key)
    if m:
        idx = int(m.group(1))
        if idx == 0:
            return 0
        if 1 <= idx <= 9:
            return idx
    if "middle_block" in key:
        return 10
    m = RE_SDXL_OUTPUT.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 8:
            return 11 + idx
    return 0


def _block_from_key_anima(key: str) -> int:
    """Map Anima DiT keys to 0-34 block index."""
    m = RE_ANIMA_BLOCK.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 27:
            return idx
    m = RE_ANIMA_LLM_BLOCK.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 5:
            return 28 + idx
    return 34  # unlayered


def detect_architecture(state_dict_keys: list[str]) -> str:
    """Detect architecture from state dict keys.

    Returns ``"sd15"``, ``"sdxl"``, or ``"anima"``.
    """
    for k in state_dict_keys:
        if "model.diffusion_model.blocks." in k or "model.diffusion_model.llm_adapter." in k:
            return "anima"
    sdxl_hints = ("conditioner.", "label_emb", "input_blocks.8")
    for key in state_dict_keys:
        for hint in sdxl_hints:
            if hint in key:
                return "sdxl"
    return "sd15"


def arch_block_count(arch: str) -> int:
    if arch == "anima":
        return 35
    if arch == "sdxl":
        return 20
    return 26


def arch_block_names(arch: str) -> list[str]:
    if arch == "anima":
        return ANIMA_BLOCK_NAMES
    if arch == "sdxl":
        return BLOCK20_NAMES
    return BLOCK26_NAMES


# ═══════════════════════════════════════════════════════════════
# Merge modes
# ═══════════════════════════════════════════════════════════════

def _get_block_weight(key: str, block_weights: list[float], arch: str, base_alpha: float) -> float:
    block_idx = block_from_key(key, arch)
    if block_idx < 0:
        return base_alpha
    if block_idx < len(block_weights):
        return block_weights[block_idx]
    return base_alpha


def _load_safetensors(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    return load_file(path, device=device)


def merge_weight(
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    block_weights: list[float],
    base_alpha: float = 0.5,
    arch: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Weight mode: A·(1−α) + B·α where α is per-block."""
    if arch is None:
        arch = detect_architecture(list(model_a.keys()))

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

        alpha = _get_block_weight(key, block_weights, arch, base_alpha)
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
    arch: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Add mode: A + (B − C)·α"""
    if arch is None:
        arch = detect_architecture(list(model_a.keys()))

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

        alpha = _get_block_weight(key, block_weights, arch, base_alpha)
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
    arch: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Triple mode: A·(1−α−β) + B·α + C·β"""
    if arch is None:
        arch = detect_architecture(list(model_a.keys()))

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

        alpha = _get_block_weight(key, block_weights_b, arch, base_alpha)
        beta = _get_block_weight(key, block_weights_c, arch, base_alpha)

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
    arch: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Twice mode: merge(A, B, α) → merge(AB, C, β)"""
    ab = merge_weight(model_a, model_b, block_weights_ab, base_alpha_ab, arch)
    return merge_weight(ab, model_c, block_weights_bc, base_alpha_bc, arch)


# ═══════════════════════════════════════════════════════════════
# Save helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(state_dict: dict[str, torch.Tensor], output_path: str, metadata: Optional[dict] = None) -> None:
    if metadata is None:
        metadata = {"format": "pt"}
    metadata["block_weight_merge"] = "true"
    save_file(state_dict, output_path, metadata=metadata)
    logger.info("Saved merged checkpoint to %s", output_path)


def load_checkpoint(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info("Loading checkpoint from %s", path)
    return _load_safetensors(path, device)
