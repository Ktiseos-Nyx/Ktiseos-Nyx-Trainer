# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

# Modified from kohya's resize script to allow removing of conv or linear dims
import os
from pathlib import Path

os.chdir("sd_scripts")

import argparse
import json
from math import log10

import torch
from safetensors.torch import load_file, save_file, safe_open
from safetensors import safe_open as safe_open_lazy
from tqdm import tqdm
import numpy as np

from library import train_util, model_util
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

MIN_SV = 1e-6

# Key pattern pairs: (down_suffix, up_suffix)
DOWN_UP_PAIRS = [
    ("lora_down", "lora_up"),
    ("lora_A", "lora_B"),
    ("down", "up")
]


def _parse_down_key(key):
    """If key is a recognized 'down' weight key, return
    (block_name, weight_suffix, down_part, up_part).  Otherwise return None."""
    for down_part, up_part in DOWN_UP_PAIRS:
        marker = f".{down_part}."
        if marker in key:
            block_name = key.rsplit(marker, 1)[0]
            weight_suffix = key.rsplit(".", 1)[-1]
            return block_name, weight_suffix, down_part, up_part
    return None


# Model save and load functions
def load_state_dict(file_name, dtype):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor and sd[key].is_floating_point():
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].is_floating_point():
                state_dict[key] = state_dict[key].to(dtype)

    if model_util.is_safetensors(file_name):
        save_file(state_dict, file_name, metadata)
    else:
        torch.save(state_dict, file_name)


# Indexing functions
def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target))
    index = max(0, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2))
    index = max(0, min(index, len(S) - 1))

    return index


def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item()) - 1
    index = max(0, min(index, len(S) - 1))
    return index


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(
        lora_rank, in_size, kernel_size, kernel_size
    ).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()

    U, S, Vh = torch.linalg.svd(weight.to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert (
        in_rank == out_rank and kernel_size == k_
    ), f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


# Calculate new rank
def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param)
    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param)
    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param)
    else:
        new_rank = rank
    new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > rank:  # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    return {
        "new_rank": new_rank,
        "new_alpha": new_alpha,
        "sum_retained": s_rank / s_sum,
        "fro_retained": fro_percent,
        "max_ratio": S[0] / S[new_rank - 1],
    }


BYTES_IN_MEGABYTE = 1 << 20

# ─── Architecture detection ───────────────────────────────────────────────────

def detect_model_type_from_lora(lora_sd):
    """Detect whether a LoRA is for SDXL or Anima based on key patterns."""
    for key in lora_sd:
        if any(key.startswith(p) for p in (
            "lora_unet_input_blocks_", "lora_unet_output_blocks_",
            "lora_unet_middle_block_", "lora_te1_", "lora_te2_",
        )):
            return "sdxl"
        if any(key.startswith(p) for p in (
            "lora_unet_blocks_", "lora_unet_t_embedder_",
            "lora_unet_final_layer_", "lora_unet_x_embedder_",
            "lora_te_",
        )):
            return "anima"
    return "unknown"


ANIMA_MODULE_ROOTS = ("blocks.", "t_embedder.", "x_embedder.", "final_layer.")


def _find_anima_prefix(checkpoint_keys):
    """Scan checkpoint keys to find the prefix that precedes Anima module paths.

    Supports no prefix (``blocks.0...``) or a single path segment prefix
    (``model.blocks.0...``, ``network.blocks.0...``, etc.).
    Returns ``(found: bool, prefix: str)`` where *prefix* is the string to strip.
    """
    for key in checkpoint_keys:
        if not key.endswith(".weight"):
            continue
        for root in ANIMA_MODULE_ROOTS:
            if key.startswith(root):
                return True, ""
        # Try stripping one leading path segment, e.g. "model." or "network."
        dot = key.find(".")
        if dot != -1:
            rest = key[dot + 1 :]
            for root in ANIMA_MODULE_ROOTS:
                if rest.startswith(root):
                    return True, key[: dot + 1]  # include the dot
    return False, ""


def detect_model_type_from_checkpoint(checkpoint_keys):
    """Detect whether a checkpoint is SDXL or Anima based on key patterns."""
    keys_list = list(checkpoint_keys)
    for key in keys_list:
        if key.startswith("model.diffusion_model."):
            return "sdxl"
    found, _ = _find_anima_prefix(keys_list)
    if found:
        return "anima"
    return "unknown"


# ─── Key mappers (base checkpoint key → LoRA key) ────────────────────────────

SDXL_UNET_PREFIX = "model.diffusion_model."
SDXL_TE_PREFIXES = [
    "conditioner.embedders.0.transformer.text_model.encoder.layers.",
    "conditioner.embedders.1.model.transformer.resblocks.",
]
SDXL_LORA_UNET_PREFIX = "lora_unet_"
SDXL_LORA_TE_PREFIXES = [
    "lora_te1_text_model_encoder_layers_",
    "lora_te2_text_model_encoder_layers_",
]


def get_sdxl_lora_key(base_key):
    """Map an SDXL base-model weight key to LoRA key name(s).

    Returns None if not mappable, or (is_split, name_or_names).
    """
    layer_name = base_key.removesuffix(".weight")

    if layer_name.startswith(SDXL_UNET_PREFIX):
        return False, SDXL_LORA_UNET_PREFIX + layer_name.removeprefix(SDXL_UNET_PREFIX).replace(".", "_")

    for te_prefix, lora_te_prefix in zip(SDXL_TE_PREFIXES, SDXL_LORA_TE_PREFIXES):
        if layer_name.startswith(te_prefix):
            lora_key = lora_te_prefix + layer_name.removeprefix(te_prefix).replace(".", "_")

            if te_prefix is SDXL_TE_PREFIXES[1]:  # CLIP G
                if "attn_in_proj_weight" in lora_key:
                    base = lora_key.replace("_attn_in_proj_weight", "_self_attn")
                    return True, [f"{base}_{c}_proj" for c in "kqv"]
                elif lora_key.endswith("_attn_out_proj"):
                    lora_key = lora_key.replace("_attn_out_proj", "_self_attn_out_proj")
                elif "_ln_" in lora_key:
                    lora_key = lora_key.replace("_ln_", "_layer_norm")
                elif "_mlp_" in lora_key:
                    lora_key = lora_key.replace("_c_fc", "_fc1").replace("_c_proj", "_fc2")
            return False, lora_key

    return None


def get_anima_lora_key(base_key, prefix=""):
    """Map an Anima base-model weight key to LoRA key name.

    Strips *prefix* from *base_key* before converting dots to underscores.
    Anima LoRA naming: ``lora_unet_{module_path_dots_to_underscores}``
    Returns None if not mappable, or (is_split, name).
    """
    if not base_key.endswith(".weight"):
        return None
    layer_name = base_key.removesuffix(".weight")
    if prefix:
        if not layer_name.startswith(prefix):
            return None
        layer_name = layer_name[len(prefix) :]
    return False, "lora_unet_" + layer_name.replace(".", "_")


# ─── Base model checkpoint wrapper ────────────────────────────────────────────

class BaseModelCheckpoint:
    """Indexes base model weights by LoRA layer names for scoring during resize.

    Uses safetensors lazy loading to avoid loading the full checkpoint into memory.
    """

    def __init__(self, path, model_type="auto"):
        self.path = path
        self.fd = safe_open_lazy(path, framework="pt")
        base_keys = list(self.fd.keys())

        if model_type == "auto":
            model_type = detect_model_type_from_checkpoint(base_keys)
        self.model_type = model_type

        if model_type == "sdxl":
            key_mapper = get_sdxl_lora_key
        elif model_type == "anima":
            found, anima_prefix = _find_anima_prefix(base_keys)
            if not found:
                logger.warning(
                    "Could not detect Anima module prefix in checkpoint '%s'; "
                    "key mapping may fail. First 5 keys: %s",
                    path, base_keys[:5],
                )
                anima_prefix = ""
            else:
                logger.info(
                    "Anima checkpoint prefix detected: %r (%d total keys)",
                    anima_prefix, len(base_keys),
                )
            key_mapper = lambda bk, p=anima_prefix: get_anima_lora_key(bk, p)
        else:
            raise ValueError(f"Unknown or undetectable model type: {model_type}")

        self._lora2base = {}   # lora_key -> base_key  or  (base_key, chunk_idx, n_chunks)
        self._shapes = {}
        self._norm_cache = {}

        for base_key in base_keys:
            shape = self.fd.get_slice(base_key).get_shape()
            self._shapes[base_key] = shape
            if not base_key.endswith(".weight") or len(shape) < 2:
                continue
            mapped = key_mapper(base_key)
            if mapped is None:
                continue
            is_split, names = mapped
            if is_split:
                n = len(names)
                for i, part_name in enumerate(names):
                    self._lora2base[part_name] = (base_key, i, n)
            elif isinstance(names, list):
                for alias in names:
                    self._lora2base[alias] = base_key
            else:
                self._lora2base[names] = base_key

    def has_key(self, lora_key):
        return lora_key in self._lora2base

    def get_weights(self, lora_key):
        """Return the base-model weight tensor corresponding to *lora_key*."""
        base_info = self._lora2base[lora_key]
        if isinstance(base_info, tuple):
            base_key, chunk_idx, n_chunks = base_info
            W = self.fd.get_tensor(base_key)
            chunk_len = W.shape[0] // n_chunks
            W = W[chunk_idx * chunk_len : (chunk_idx + 1) * chunk_len]
        else:
            W = self.fd.get_tensor(base_info)
        return W

    def frobenius_norm(self, lora_key, dtype=torch.float32):
        cache_key = ("fro", lora_key)
        if cache_key in self._norm_cache:
            return self._norm_cache[cache_key]
        W = self.get_weights(lora_key).to(dtype=dtype)
        val = torch.linalg.matrix_norm(W.flatten(start_dim=1), ord="fro").cpu().item()
        self._norm_cache[cache_key] = val
        return val

    def spectral_norm(self, lora_key, dtype=torch.float32, **_kw):
        cache_key = ("spn", lora_key)
        if cache_key in self._norm_cache:
            return self._norm_cache[cache_key]
        W = self.get_weights(lora_key).to(dtype=dtype)
        val = torch.svd_lowrank(W.flatten(start_dim=1), q=1, niter=64)[1][0].cpu().item()
        self._norm_cache[cache_key] = val
        return val


# ─── Fast SVD via QR ──────────────────────────────────────────────────────────

def fast_decompose(up, down):
    """Efficient SVD decomposition of up @ down via QR factorisations."""
    Qu, Ru = torch.linalg.qr(up.flatten(start_dim=1))
    Qd, Rd = torch.linalg.qr(down.flatten(start_dim=1).mT)
    Uc, Sc, Vhc = torch.linalg.svd(Ru @ Rd.mT, full_matrices=False)
    return Qu @ Uc, Sc, Vhc @ Qd.mT


# ─── Decomposed LoRA layer ───────────────────────────────────────────────────

class DecomposedLoRALayer:
    """A single LoRA layer decomposed via SVD for scoring-based resizing."""

    def __init__(self, name, alpha, up, down, dora_scale=None):
        self.name = name
        self.input_shape = down.shape[1:]
        self.U, S, self.Vh = fast_decompose(up, down)
        self.alpha_factor = alpha / down.shape[0]
        self.S = S * self.alpha_factor
        self.dora_scale = dora_scale

    @property
    def dim(self):
        return self.S.shape[0]

    def dim_size(self, element_size=2):
        return element_size * (self.U.shape[0] + self.Vh.shape[1])

    def statedict(self, mask=None, rescale=1.0, dtype=None):
        S, Vh, U = self.S, self.Vh, self.U
        if mask is not None:
            S, Vh, U = S[mask], Vh[mask], U[:, mask]
        dim = S.shape[0]
        if dim == 0:
            return {}

        alpha_factor = self.alpha_factor
        S_sqrt = torch.sqrt(S * (rescale / alpha_factor))
        down = (Vh * S_sqrt.unsqueeze(1)).view(dim, *self.input_shape)
        up = (U * S_sqrt).view(*U.shape, *([1] * (len(self.input_shape) - 1)))
        alpha = torch.scalar_tensor(alpha_factor * dim, dtype=down.dtype)

        d = {
            f"{self.name}.alpha": alpha,
            f"{self.name}.lora_down.weight": down,
            f"{self.name}.lora_up.weight": up,
        }
        if self.dora_scale is not None:
            d[f"{self.name}.dora_scale"] = self.dora_scale
        if dtype is not None:
            d = {k: v.to(dtype) if v.is_floating_point() else v for k, v in d.items()}
        return d

    def compute_subspace_scales(self, W_base):
        W_base = W_base.flatten(start_dim=1).to(device=self.S.device, dtype=self.S.dtype)
        return torch.linalg.vecdot(self.Vh @ W_base.T, self.U.T)

    def to(self, **kwargs):
        self.Vh = self.Vh.to(**kwargs)
        self.S = self.S.to(**kwargs)
        self.U = self.U.to(**kwargs)
        return self


# ─── Scoring recipe ──────────────────────────────────────────────────────────

class ResizeRecipe:
    """Scoring recipe for base-model-aware LoRA resizing.

    Recipe string format: ``key1=val1,key2=val2,...``
    Score keys: spn_lora, spn_ckpt, subspace, fro_lora, fro_ckpt, params
    Control keys: size=<MB>, thr=<threshold>, rescale=<factor>

    At least one score key and either ``size`` or ``thr`` must be specified.
    """

    def __init__(self, recipe_str):
        self.recipe_str = recipe_str
        parsed = {
            "spn_lora": 0.0, "spn_ckpt": 0.0, "subspace": 0.0,
            "fro_lora": 0.0, "fro_ckpt": 0.0, "params": 0.0,
        }
        self.target_size = None
        self.threshold = None
        self.rescale = 1.0

        for part in recipe_str.split(","):
            key, _, value = part.partition("=")
            if value:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Could not parse {key}={value} in recipe {recipe_str}")
            if key in parsed:
                parsed[key] = 1.0 if value == "" else value
                continue
            if value == "":
                raise ValueError(f"Empty value not accepted for key {key} in recipe {recipe_str}")
            if key == "size":
                self.target_size = value
            elif key == "thr":
                self.threshold = value
            elif key == "rescale":
                self.rescale = value
            else:
                raise ValueError(f"Unknown key {key} in recipe {recipe_str}")

        wsum = sum(parsed.values())
        if wsum == 0.0:
            raise ValueError("At least one score type must be specified in the recipe")
        self.weights = {k: v / wsum for k, v in parsed.items()}

        if self.target_size is None and self.threshold is None:
            raise ValueError("Either 'size' or 'thr' must be specified in the recipe")

    def __str__(self):
        return self.recipe_str

    def score_dims(self, layer, checkpoint, **compute_kwargs):
        w = self.weights
        S = layer.S
        scores = torch.log10(S)
        if self.rescale is not None:
            scores = scores + log10(self.rescale)
        if abs(w["subspace"]) > 1e-6:
            W_base = checkpoint.get_weights(layer.name).to(**compute_kwargs)
            scores = scores - w["subspace"] * torch.log10(
                layer.compute_subspace_scales(W_base).abs().cpu()
            )
        if abs(w["spn_ckpt"]) > 1e-6:
            scores = scores - w["spn_ckpt"] * log10(
                checkpoint.spectral_norm(layer.name, **compute_kwargs)
            )
        if abs(w["spn_lora"]) > 1e-6:
            scores = scores - w["spn_lora"] * torch.log10(S[0])
        if abs(w["fro_ckpt"]) > 1e-6:
            scores = scores - w["fro_ckpt"] * log10(
                checkpoint.frobenius_norm(layer.name, dtype=torch.float32)
            )
        if abs(w["fro_lora"]) > 1e-6:
            scores = scores - w["fro_lora"] * torch.log10(torch.linalg.vector_norm(S))
        if abs(w["params"]) > 1e-6:
            scores = scores - w["params"] * log10(layer.dim_size(1))
        return scores

    def resize_lora(self, lora_layers, checkpoint, compute_kwargs, output_dtype):
        output_elem_size = 2 if output_dtype == torch.float16 else 4

        scores = [self.score_dims(layer, checkpoint, **compute_kwargs) for layer in lora_layers]

        if self.target_size is not None:
            flat_scores = torch.cat(scores)
            flat_scores, order = flat_scores.sort(descending=True)
            cum_sizes = torch.repeat_interleave(
                *torch.tensor(
                    [(layer.dim_size(output_elem_size), layer.dim) for layer in lora_layers],
                    dtype=torch.int32,
                ).T
            )[order].cumsum(0)
            target_bytes = self.target_size * BYTES_IN_MEGABYTE
            if target_bytes < cum_sizes[-1]:
                idx = torch.searchsorted(cum_sizes, target_bytes).item()
                threshold = flat_scores[idx].item()
                logger.info(
                    "Total rank: %d. threshold %.2f: target:%.3fM <= real:%.3fM",
                    idx + 1, threshold, self.target_size, cum_sizes[idx] / BYTES_IN_MEGABYTE,
                )
            else:
                threshold = -torch.inf
        else:
            threshold = self.threshold

        sd = {}
        for layer, layer_scores in zip(lora_layers, scores):
            mask = layer_scores >= threshold
            sd.update(layer.statedict(mask=mask, dtype=output_dtype, rescale=self.rescale))

        return sd, threshold


# ─── Base-model-aware resize entry point ──────────────────────────────────────

def resize_lora_model_with_base(
    lora_sd,
    base_model_path,
    base_model_type,
    score_recipe_str,
    save_dtype,
    device,
    verbose,
):
    """Resize a LoRA using a base model checkpoint for scoring-based rank selection.

    This approach decomposes every LoRA layer via SVD, scores each singular
    value against base-model statistics, and keeps only those that exceed a
    threshold (or that fit a target file size).
    """
    checkpoint = BaseModelCheckpoint(base_model_path, model_type=base_model_type)
    recipe = ResizeRecipe(score_recipe_str)

    compute_kwargs = dict(dtype=torch.float32)

    # Discover all LoRA layer names
    lora_layer_names = set()
    for key in lora_sd:
        if key.endswith(".alpha"):
            lora_layer_names.add(key.removesuffix(".alpha"))

    lora_layers = []
    passthrough_keys = set()
    n_total = len(lora_layer_names)

    for name in tqdm(sorted(lora_layer_names), desc="Decomposing"):
        alpha_val = lora_sd.get(f"{name}.alpha")
        if alpha_val is None:
            continue
        alpha = float(alpha_val)

        # Find down and up weights
        down = up = None
        for down_part, up_part in DOWN_UP_PAIRS:
            d_key = f"{name}.{down_part}.weight"
            u_key = f"{name}.{up_part}.weight"
            if d_key in lora_sd and u_key in lora_sd:
                down = lora_sd[d_key].to(dtype=torch.float32)
                up = lora_sd[u_key].to(dtype=torch.float32)
                break

        if down is None or up is None:
            continue

        # Handle Tucker decomposition (lora_mid) — reconstruct full weight then re-factor
        mid_key = f"{name}.lora_mid.weight"
        if mid_key in lora_sd:
            mid = lora_sd[mid_key].to(dtype=torch.float32, device=device)
            full = torch.einsum(
                "or,rihw,ij->ojhw",
                up.flatten(start_dim=1).to(device),
                mid,
                down.flatten(start_dim=1).to(device),
            )
            out_size = full.shape[0]
            in_size = full.shape[1]
            kernel_size = full.shape[2]
            rank = min(down.shape[0], up.shape[1])
            U_f, S_f, Vh_f = torch.linalg.svd(full.reshape(out_size, -1), full_matrices=False)
            rank = min(rank, S_f.shape[0])
            S_sqrt = torch.sqrt(S_f[:rank])
            down = (Vh_f[:rank] * S_sqrt.unsqueeze(1)).reshape(rank, in_size, kernel_size, kernel_size)
            up = (U_f[:, :rank] * S_sqrt).reshape(out_size, rank, 1, 1)
            alpha = float(rank)

        dora_scale = lora_sd.get(f"{name}.dora_scale")

        if not checkpoint.has_key(name):
            logger.warning(
                "LoRA layer %s not found in base model, passing through unchanged", name
            )
            passthrough_keys.add(name)
            continue

        layer = DecomposedLoRALayer(name, alpha, up.to(device), down.to(device), dora_scale)
        layer.to(device="cpu")

        if layer.S[0].abs().item() < MIN_SV:
            if verbose:
                logger.info("Skipping zero layer: %s", name)
            continue

        lora_layers.append(layer)

    n_matched = len(lora_layers)
    n_passthrough = len(passthrough_keys)
    logger.info(
        "Layer matching: %d/%d scored, %d passed through (no base match)",
        n_matched, n_total, n_passthrough,
    )
    if n_passthrough == n_total:
        logger.error(
            "ALL layers passed through — base model key mapping produced zero matches. "
            "Check that the correct checkpoint is provided and model type is correct."
        )

    if not lora_layers:
        raise ValueError("No valid LoRA layers found for scoring-based resize")

    logger.info("Scoring %d layers with recipe: %s", n_matched, score_recipe_str)
    new_sd, threshold = recipe.resize_lora(
        lora_layers,
        checkpoint,
        compute_kwargs=compute_kwargs,
        output_dtype=save_dtype,
    )

    orig_dims = sum(layer.dim for layer in lora_layers)
    kept_dims = sum(
        t.shape[0] for k, t in new_sd.items() if k.endswith(".lora_down.weight")
    )
    logger.info(
        "Dims: %d → %d (%.1f%% kept, threshold=%.4f)",
        orig_dims, kept_dims,
        100.0 * kept_dims / orig_dims if orig_dims else 0.0,
        threshold if threshold is not None else float("nan"),
    )

    # Pass through unmatched layers as-is
    for name in passthrough_keys:
        for key in lora_sd:
            if key.startswith(name + "."):
                new_sd[key] = lora_sd[key].to(save_dtype) if lora_sd[key].is_floating_point() else lora_sd[key]

    # Determine output network dim
    dims = set()
    for key in new_sd:
        if ".lora_down.weight" in key:
            dims.add(new_sd[key].shape[0])
    network_dim = max(dims) if dims else 0

    return new_sd, network_dim, threshold


def resize_lora_model(
    lora_sd,
    new_rank,
    new_conv_rank,
    save_dtype,
    device,
    dynamic_method,
    dynamic_param,
    verbose,
    del_linear,
    del_conv,
):  # sourcery skip: use-fstring-for-concatenation
    network_alpha = None
    network_dim = None
    verbose_str = "\n"
    fro_list = []

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if network_alpha is None and "alpha" in key:
            network_alpha = value
        if network_dim is None:
            if _parse_down_key(key) is not None and len(value.size()) >= 2:
                network_dim = value.size()[0]
        if network_alpha is not None and network_dim is not None:
            break

    if network_alpha is None:
        network_alpha = network_dim

    if isinstance(network_alpha, torch.Tensor):
        network_alpha = float(network_alpha)

    if network_dim is None:
        raise ValueError(
            "Could not determine network dim from the model. "
            "No recognized down weight keys found (expected lora_down, lora_A, or down)."
        )

    scale = network_alpha / network_dim

    if dynamic_method:
        logger.info(
            f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}"
        )

    o_lora_sd = lora_sd.copy()
    new_alpha = 0.0

    with torch.no_grad():
        for key, value in tqdm(lora_sd.items()):
            parsed = _parse_down_key(key)
            if parsed is None:
                continue

            block_name, weight_suffix, down_part, up_part = parsed
            lora_down_weight = value
            # find corresponding lora_up, optional lora_mid (Tucker), and alpha
            lora_up_weight = lora_sd.get(
                f"{block_name}.{up_part}.{weight_suffix}", None
            )
            lora_mid_weight = lora_sd.get(
                f"{block_name}.lora_mid.{weight_suffix}", None
            )
            lora_alpha = lora_sd.get(f"{block_name}.alpha", None)

            if lora_up_weight is None:
                continue

            # Original key names for cleanup
            orig_down_key = f"{block_name}.{down_part}.{weight_suffix}"
            orig_up_key = f"{block_name}.{up_part}.{weight_suffix}"
            orig_mid_key = f"{block_name}.lora_mid.{weight_suffix}"
            alpha_key = f"{block_name}.alpha"
            dora_key = f"{block_name}.dora_scale"

            has_mid = lora_mid_weight is not None
            conv2d = has_mid or len(lora_down_weight.size()) == 4
            scale = (
                1.0
                if lora_alpha is None
                else float(lora_alpha) / lora_down_weight.size()[0]
            )

            if conv2d:
                if del_conv:
                    for k in [orig_down_key, orig_up_key, alpha_key, dora_key]:
                        o_lora_sd.pop(k, None)
                    if has_mid:
                        o_lora_sd.pop(orig_mid_key, None)
                    continue

                if has_mid:
                    # Tucker decomposition: reconstruct full conv weight
                    full_weight_matrix = torch.einsum(
                        "or,rihw,ij->ojhw",
                        lora_up_weight.flatten(start_dim=1).to(device),
                        lora_mid_weight.to(device),
                        lora_down_weight.flatten(start_dim=1).to(device),
                    )
                else:
                    full_weight_matrix = merge_conv(
                        lora_down_weight, lora_up_weight, device
                    )
                param_dict = extract_conv(
                    full_weight_matrix,
                    new_conv_rank,
                    dynamic_method,
                    dynamic_param,
                    device,
                    scale,
                )
            else:
                if del_linear:
                    for k in [orig_down_key, orig_up_key, alpha_key, dora_key]:
                        o_lora_sd.pop(k, None)
                    continue

                full_weight_matrix = merge_linear(
                    lora_down_weight, lora_up_weight, device
                )
                param_dict = extract_linear(
                    full_weight_matrix,
                    new_rank,
                    dynamic_method,
                    dynamic_param,
                    device,
                    scale,
                )

            if verbose:
                max_ratio = param_dict["max_ratio"]
                sum_retained = param_dict["sum_retained"]
                fro_retained = param_dict["fro_retained"]
                if not np.isnan(fro_retained):
                    fro_list.append(float(fro_retained))

                verbose_str += f"{block_name:75} | "
                verbose_str += f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"

            if verbose and dynamic_method:
                verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
            else:
                verbose_str += "\n"

            new_alpha = param_dict["new_alpha"]
            # Remove original keys (handles renamed patterns and Tucker mid)
            for k in [orig_down_key, orig_up_key, alpha_key]:
                o_lora_sd.pop(k, None)
            if has_mid:
                o_lora_sd.pop(orig_mid_key, None)

            # Write standardized lora_down/lora_up output keys
            o_lora_sd[f"{block_name}.lora_down.weight"] = (
                param_dict["lora_down"].to(save_dtype).contiguous()
            )
            o_lora_sd[f"{block_name}.lora_up.weight"] = (
                param_dict["lora_up"].to(save_dtype).contiguous()
            )
            o_lora_sd[f"{block_name}.alpha"] = torch.tensor(
                param_dict["new_alpha"]
            ).to(save_dtype)

            del param_dict

    if verbose:
        print(verbose_str)
        print(
            f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}"
        )
    logger.info("resizing complete")
    return o_lora_sd, network_dim, new_alpha


def resize(args):
    if args.save_to is None or not (
        args.save_to.endswith(".ckpt")
        or args.save_to.endswith(".pt")
        or args.save_to.endswith(".pth")
        or args.save_to.endswith(".safetensors")
    ):
        raise Exception(
            "The --save_to argument must be specified and must be a .ckpt , .pt, .pth or .safetensors file."
        )

    args.new_conv_rank = (
        args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    )

    def str_to_dtype(p):
        if p in ["float", "fp32"]:
            return torch.float
        if p in ["fp16", "half"]:
            return torch.float16
        return torch.bfloat16 if p in ["bf16", "bfloat16"] else None

    merge_dtype = str_to_dtype(
        "float"
    )  # matmul method above only seems to work in float32
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    logger.info("loading Model...")
    lora_sd, metadata = load_state_dict(args.model, merge_dtype)

    # ── Base-model-aware scoring path ──
    base_model = getattr(args, "base_model", None)
    if base_model:
        base_model_type = getattr(args, "base_model_type", "auto")
        score_recipe = getattr(args, "score_recipe", "fro_ckpt=1,thr=-3.0")

        logger.info("Resizing LoRA with base model reference...")
        state_dict, old_dim, threshold = resize_lora_model_with_base(
            lora_sd,
            base_model_path=base_model,
            base_model_type=base_model_type,
            score_recipe_str=score_recipe,
            save_dtype=save_dtype,
            device=args.device,
            verbose=args.verbose,
        )

        if metadata is None:
            metadata = {}
        comment = metadata.get("ss_training_comment", "")
        metadata["ss_training_comment"] = (
            f"Resized with base model scoring (recipe: {score_recipe}, threshold: {threshold}); {comment}"
        )
        metadata["ss_network_dim"] = "Dynamic"
        metadata["ss_network_alpha"] = "Dynamic"
        metadata["resize_recipe"] = score_recipe
        metadata["resize_threshold"] = str(threshold)

    # ── Standard resize path (no base model) ──
    else:
        if args.dynamic_method and not args.dynamic_param:
            raise Exception("If using dynamic_method, then dynamic_param is required")

        logger.info("Resizing Lora...")
        state_dict, old_dim, new_alpha = resize_lora_model(
            lora_sd,
            args.new_rank,
            args.new_conv_rank,
            save_dtype,
            args.device,
            args.dynamic_method,
            args.dynamic_param,
            args.verbose,
            args.del_linear,
            args.del_conv,
        )

        if metadata is None:
            metadata = {}

        comment = metadata.get("ss_training_comment", "")

        if not args.dynamic_method:
            if args.del_conv:
                conv_desc = "(conv: Deleted Conv Dims)"
            else:
                conv_desc = (
                    ""
                    if args.new_rank == args.new_conv_rank
                    else f" (conv: {args.new_conv_rank})"
                )
            if args.del_linear:
                metadata["ss_training_comment"] = (
                    f"Deleted Linear Dims{conv_desc}; {comment}"
                )
            else:
                metadata["ss_training_comment"] = (
                    f"dimension is resized from {old_dim} to {args.new_rank}{conv_desc}; {comment}"
                )
            metadata["ss_network_dim"] = str(0 if args.del_linear else args.new_rank)
            metadata["ss_network_alpha"] = str(new_alpha)
        else:
            if args.del_linear:
                linear_message = "Deleted Linear Dims"
            else:
                linear_message = f"Dynamically resized linear dims with {args.dynamic_method}: {args.dynamic_param} from {old_dim}"
            if args.del_conv:
                conv_message = "Deleted Conv Dims"
            else:
                conv_message = f"Dynamically resized conv dims with {args.dynamic_method}: {args.dynamic_param}"

            metadata["ss_training_comment"] = f"{linear_message}({conv_message}); {comment}"
            metadata["ss_network_dim"] = "Dynamic"
            metadata["ss_network_alpha"] = "Dynamic"

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(
        state_dict, metadata
    )
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, float if omitted / 保存時の精度、未指定時はfloat",
    )
    parser.add_argument(
        "--new_rank",
        type=int,
        default=4,
        help="Specify rank of output LoRA / 出力するLoRAのrank (dim)",
    )
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LoRA model to resize at to new rank: ckpt or safetensors file / 読み込むLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose resizing information / rank変更時の詳細情報を出力する",
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        default=None,
        choices=[None, "sv_ratio", "sv_fro", "sv_cumulative"],
        help="Specify dynamic resizing method, --new_rank is used as a hard limit for max rank",
    )
    parser.add_argument(
        "--dynamic_param",
        type=float,
        default=None,
        help="Specify target for dynamic reduction",
    )
    parser.add_argument(
        "--del_conv",
        action="store_true",
        help="Removes the Conv Dims of the model while resizing",
    )
    parser.add_argument(
        "--del_linear",
        action="store_true",
        help="Removes the Linear Dims of the model while resizing",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Path to base model checkpoint (.safetensors) for scoring-based resize. "
             "When provided, singular values are scored against base-model statistics "
             "instead of using the standard rank-truncation approach.",
    )
    parser.add_argument(
        "--base_model_type",
        type=str,
        default="auto",
        choices=["auto", "sdxl", "anima"],
        help="Type of base model (auto-detected from checkpoint keys if 'auto')",
    )
    parser.add_argument(
        "--score_recipe",
        type=str,
        default="fro_ckpt=1,thr=-3.0",
        help="Scoring recipe for base-model-aware resize. Format: key=val,key=val,... "
             "Score keys: spn_lora, spn_ckpt, subspace, fro_lora, fro_ckpt, params. "
             "Control keys: size=<MB>, thr=<threshold>, rescale=<factor>. "
             "At least one score key and either 'size' or 'thr' must be specified.",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    output_folder = Path(args.save_to).parent
    if output_folder.name == "default_output":
        output_folder = Path("../default_output")
        if not output_folder.is_dir():
            output_folder.mkdir()
        args.save_to = (
            output_folder.joinpath(Path(args.save_to).name).resolve().as_posix()
        )
    resize(args)
