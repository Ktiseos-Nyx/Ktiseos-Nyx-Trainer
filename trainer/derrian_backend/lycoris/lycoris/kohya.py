import os
import ast
import fnmatch
import re
import logging

from typing import Any, List, Optional, Dict, Tuple
import numbers

import numpy as np

import torch

import math

from .utils import precalculate_safetensors_hashes
from .wrapper import LycorisNetwork, network_module_dict, deprecated_arg_dict
from .modules.abba import AbbaModule
from .modules.locon import LoConModule
from .modules.loha import LohaModule
from .modules.ia3 import IA3Module
from .modules.lokr import LokrModule
from .modules.dylora import DyLoraModule
from .modules.glora import GLoRAModule
from .modules.norms import NormModule
from .modules.full import FullModule
from .modules.diag_oft import DiagOFTModule
from .modules.boft import ButterflyOFTModule
from .modules.tlora import TLoraModule
from .modules import make_module, get_module

from .config import PRESET
from .utils.preset import read_preset
from .utils import str_bool
from .logging import logger
import warnings

from collections import defaultdict

try:
    from ramtorch.modules.linear import CPUBouncingLinear
except ImportError:
    CPUBouncingLinear = type(None)

LORA_PLUS_TARGETS = ["lora_up","a1","b1"]


def _parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, Any]:
    """Parse a string of key-value pairs separated by commas (e.g. 'regex1=8,regex2=16')."""
    pairs = {}
    for pair in kv_pair_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning(f"Invalid format: {pair}, expected 'key=value'")
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            pairs[key] = int(value) if is_int else float(value)
        except ValueError:
            logger.warning(f"Invalid value for {key}: {value}")
    return pairs


def create_network(
    multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs
):
    for key, value in list(kwargs.items()):
        if key in deprecated_arg_dict:
            logger.warning(
                f"{key} is deprecated. Please use {deprecated_arg_dict[key]} instead.",
                stacklevel=2,
            )
            kwargs[deprecated_arg_dict[key]] = value
    if network_dim is None:
        network_dim = 4  # default
    conv_dim = int(kwargs.get("conv_dim", network_dim) or network_dim)
    conv_alpha = float(kwargs.get("conv_alpha", network_alpha) or network_alpha)
    dropout = float(kwargs.get("dropout", 0.0) or 0.0)
    rank_dropout = float(kwargs.get("rank_dropout", 0.0) or 0.0)
    module_dropout = float(kwargs.get("module_dropout", 0.0) or 0.0)
    algo = (kwargs.get("algo", "lora") or "lora").lower()
    use_tucker = str_bool(
        not kwargs.get("disable_conv_cp", True)
        or kwargs.get("use_conv_cp", False)
        or kwargs.get("use_cp", False)
        or kwargs.get("use_tucker", False)
    )
    use_scalar = str_bool(kwargs.get("use_scalar", False))
    block_size = int(kwargs.get("block_size", 4) or 4)
    train_norm = str_bool(kwargs.get("train_norm", False))
    constraint = float(kwargs.get("constraint", 0.0) or 0.0)
    rescaled = str_bool(kwargs.get("rescaled", False))
    weight_decompose = str_bool(kwargs.get("dora_wd", False))
    wd_on_output = str_bool(kwargs.get("wd_on_output", True))
    full_matrix = str_bool(kwargs.get("full_matrix", False))
    bypass_mode = str_bool(kwargs.get("bypass_mode", False))
    rs_lora = str_bool(kwargs.get("rs_lora", False))
    unbalanced_factorization = str_bool(kwargs.get("unbalanced_factorization", False))
    orthogonalize = str_bool(kwargs.get("orthogonalize", False))
    if orthogonalize:
        logger.info("Orthogonalization of weights for Lycoris is enabled")
        if use_scalar == False:
            logger.info("Forcing usage of use_scalar as orthogonalization is enabled")
            use_scalar = True

    train_t5xxl = str_bool(kwargs.get("train_t5xxl", False))
    torch_compile = str_bool(kwargs.get("torch_compile", False))
    torch_compile_mode = kwargs.get("torch_compile_mode", "max-autotune")
    torch_compile_dynamic = str_bool(kwargs.get("torch_compile_dynamic", False))
    torch_compile_fullgraph = str_bool(kwargs.get("torch_compile_fullgraph", True))
    train_llm_adapter = str_bool(kwargs.get("train_llm_adapter", False))

    # exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    # include patterns (overrides exclude)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    # regex-specific learning rates / dimensions
    network_reg_lrs_str = kwargs.get("network_reg_lrs", None)
    reg_lrs = _parse_kv_pairs(network_reg_lrs_str, is_int=False) if network_reg_lrs_str is not None else None

    network_reg_dims_str = kwargs.get("network_reg_dims", None)
    reg_dims = _parse_kv_pairs(network_reg_dims_str, is_int=True) if network_reg_dims_str is not None else None

    ggpo_beta = kwargs.get("ggpo_beta", None)
    ggpo_sigma = kwargs.get("ggpo_sigma", None)
    ggpo_conv = kwargs.get("ggpo_conv", False)
    ggpo_conv_weight_sample_size = kwargs.get("ggpo_conv_weight_sample_size", 100)

    if ggpo_beta is not None:
        ggpo_beta = float(ggpo_beta)

    if ggpo_sigma is not None:
        ggpo_sigma = float(ggpo_sigma)

    if ggpo_conv is not None:
        ggpo_conv = bool(ggpo_conv)

    if ggpo_conv_weight_sample_size is not None:
        ggpo_conv_weight_sample_size = int(ggpo_conv_weight_sample_size)

    if ggpo_beta is not None and ggpo_sigma is not None:
        logger.info(f"LoRA-GGPO training sigma: {ggpo_sigma} beta: {ggpo_beta}")
    # lora_plus
    loraplus_lr_ratio = (
        float(kwargs.get("loraplus_lr_ratio", None))
        if kwargs.get("loraplus_lr_ratio", None) is not None
        else None
    )
    loraplus_unet_lr_ratio = (
        float(kwargs.get("loraplus_unet_lr_ratio", None))
        if kwargs.get("loraplus_unet_lr_ratio", None) is not None
        else None
    )
    loraplus_text_encoder_lr_ratio = (
        float(kwargs.get("loraplus_text_encoder_lr_ratio", None))
        if kwargs.get("loraplus_text_encoder_lr_ratio", None) is not None
        else None
    )

    if unbalanced_factorization:
        logger.info("Unbalanced factorization for LoKr is enabled")

    if weight_decompose:
        logger.info("Weight decomposition (DoRA) is enabled")

    if bypass_mode and weight_decompose:
        bypass_mode = False
        logger.info("Because weight decomposition (DoRA) is enabled, bypass mode has been disabled")
    elif bypass_mode:
        logger.info("Bypass mode is enabled")

    if full_matrix:
        logger.info("Full matrix mode for LoKr is enabled")

    preset_str = kwargs.get("preset", "full")
    if preset_str not in PRESET:
        preset = read_preset(preset_str)
    else:
        preset = PRESET[preset_str]
    assert preset is not None
    LycorisNetworkKohya.apply_preset(preset)

    # Auto-add Anima embedding modules when "Block" is targeted
    _is_anima = False
    if isinstance(unet, torch.nn.Module):
        _is_anima = unet.__class__.__name__.lower() == "anima"
    if _is_anima:
        if "exclude_name" not in preset:
            anima_default_excludes = [r".*(_modulation|_embedder|final_layer).*"]
            for p in anima_default_excludes:
                if p not in LycorisNetworkKohya.TARGET_EXCLUDE_NAME:
                    LycorisNetworkKohya.TARGET_EXCLUDE_NAME.append(p)
            logger.info(f"Anima model detected: added {anima_default_excludes} to target exclude names")

    logger.info(f"Using rank adaptation algo: {algo}")

    if algo == "ia3" and preset_str != "ia3":
        logger.warning("It is recommended to use preset ia3 for IA^3 algorithm")

    if torch_compile:
        logger.info(f"Torch compile enabled for network.\n \
                    dynamic={torch_compile_dynamic}\n \
                    mode={torch_compile_mode}\n \
                    fullgraph={torch_compile_fullgraph}")

    network = LycorisNetworkKohya(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        conv_lora_dim=conv_dim,
        alpha=network_alpha,
        conv_alpha=conv_alpha,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        use_tucker=use_tucker,
        use_scalar=use_scalar,
        network_module=algo,
        train_norm=train_norm,
        decompose_both=kwargs.get("decompose_both", False),
        factor=kwargs.get("factor", -1),
        block_size=block_size,
        constraint=constraint,
        rescaled=rescaled,
        weight_decompose=weight_decompose,
        wd_on_output=wd_on_output,
        full_matrix=full_matrix,
        bypass_mode=bypass_mode,
        rs_lora=rs_lora,
        unbalanced_factorization=unbalanced_factorization,
        train_t5xxl=train_t5xxl,
        ggpo_beta=ggpo_beta,
        ggpo_sigma=ggpo_sigma,
        ggpo_conv=ggpo_conv,
        ggpo_conv_weight_sample_size=ggpo_conv_weight_sample_size,
        orthogonalize=orthogonalize,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
    )
    if (
        loraplus_lr_ratio is not None
        or loraplus_unet_lr_ratio is not None
        or loraplus_text_encoder_lr_ratio is not None
    ):
        network.set_loraplus_lr_ratio(
            loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
        )

    if torch_compile:
        with torch._dynamo.utils.disable_cache_limit():
            return torch.compile(network, dynamic=torch_compile_dynamic, mode=torch_compile_mode, fullgraph=torch_compile_fullgraph)
    else:
        return network


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    unet_loras = {}
    te_loras = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
            unet_loras[lora_name] = None
        elif lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
            te_loras[lora_name] = None

    for name, modules in unet.named_modules():
        lora_name = f"{LycorisNetworkKohya.LORA_PREFIX_UNET}_{name}".replace(".", "_")
        if lora_name in unet_loras:
            unet_loras[lora_name] = modules

    if text_encoder:
        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        for idx, te in enumerate(text_encoders):
            if use_index:
                prefix = f"{LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER}{idx+1}"
            else:
                prefix = LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
            for name, modules in te.named_modules():
                lora_name = f"{prefix}_{name}".replace(".", "_")
                if lora_name in te_loras:
                    te_loras[lora_name] = modules

    original_level = logger.level
    logger.setLevel(logging.ERROR)
    network = LycorisNetworkKohya(text_encoder, unet)
    network.unet_loras = []
    network.text_encoder_loras = []
    logger.setLevel(original_level)

    logger.info("Loading UNet Modules from state dict...")
    for lora_name, orig_modules in unet_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.unet_loras.append(module)
    logger.info(f"{len(network.unet_loras)} Modules Loaded")

    logger.info("Loading TE Modules from state dict...")

    if text_encoder:
        for lora_name, orig_modules in te_loras.items():
            if orig_modules is None:
                continue
            lyco_type, params = get_module(weights_sd, lora_name)
            module = make_module(lyco_type, params, lora_name, orig_modules)
            if module is not None:
                network.text_encoder_loras.append(module)
        logger.info(f"{len(network.text_encoder_loras)} Modules Loaded")

    for lora in network.unet_loras + network.text_encoder_loras:
        lora.multiplier = multiplier

    return network, weights_sd


class LycorisNetworkKohya(LycorisNetwork):
    """
    LoRA + LoCon
    """

    # Ignore proj_in or proj_out, their channels is only a few.
    ENABLE_CONV = True
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        "HunYuanDiTBlock",
        "DoubleStreamBlock",
        "SingleStreamBlock",
        "SingleDiTBlock",
        "MMDoubleStreamBlock",  # HunYuanVideo
        "MMSingleStreamBlock",  # HunYuanVideo
        "WanAttentionBlock", # Wan
        "HunyuanVideoTransformerBlock", # FramePack
        "HunyuanVideoSingleTransformerBlock", # FramePack
        "JointTransformerBlock", # lumina-image-2
        "FinalLayer", # lumina-image-2
        "QwenImageTransformerBlock", # Qwen
        "Block", # Anima?
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "CLIPAttention",
        "CLIPSdpaAttention",
        "CLIPMLP",
        "MT5Block",
        "BertLayer",
        "Gemma2Attention",
        "Gemma2FlashAttention2",
        "Gemma2SdpaAttention",
        "Gemma2MLP",
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}
    USE_FNMATCH = False
    TARGET_EXCLUDE_NAME = []
    EXCLUDE_PATTERNS = None
    INCLUDE_PATTERNS = None
    REG_DIMS = None
    REG_LRS = None

    @classmethod
    def apply_preset(cls, preset):
        if "enable_conv" in preset:
            cls.ENABLE_CONV = preset["enable_conv"]
        if "unet_target_module" in preset:
            cls.UNET_TARGET_REPLACE_MODULE = preset["unet_target_module"]
        if "unet_target_name" in preset:
            cls.UNET_TARGET_REPLACE_NAME = preset["unet_target_name"]
        if "text_encoder_target_module" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_MODULE = preset[
                "text_encoder_target_module"
            ]
        if "text_encoder_target_name" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_NAME = preset["text_encoder_target_name"]
        if "module_algo_map" in preset:
            cls.MODULE_ALGO_MAP = preset["module_algo_map"]
        if "name_algo_map" in preset:
            cls.NAME_ALGO_MAP = preset["name_algo_map"]
        if "use_fnmatch" in preset:
            cls.USE_FNMATCH = preset["use_fnmatch"]
        if "exclude_name" in preset:
            cls.TARGET_EXCLUDE_NAME = preset["exclude_name"]
        if "exclude_patterns" in preset:
            cls.EXCLUDE_PATTERNS = preset["exclude_patterns"]
        if "include_patterns" in preset:
            cls.INCLUDE_PATTERNS = preset["include_patterns"]
        if "reg_dims" in preset:
            cls.REG_DIMS = preset["reg_dims"]
        if "reg_lrs" in preset:
            cls.REG_LRS = preset["reg_lrs"]
        return cls

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=0,
        alpha=1,
        conv_alpha=1,
        use_tucker=False,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        network_module: str = "locon",
        norm_modules=NormModule,
        train_norm=False,
        train_t5xxl=False,
        train_llm_adapter=False,
        exclude_patterns=None,
        include_patterns=None,
        reg_dims=None,
        reg_lrs=None,
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        root_kwargs = kwargs
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.train_t5xxl = train_t5xxl
        self.train_llm_adapter = train_llm_adapter
        self._current_step = 0

        self.ggpo_beta = kwargs.get("ggpo_beta", None)
        self.ggpo_sigma = kwargs.get("ggpo_sigma", None)
        self.ggpo_conv = kwargs.get("ggpo_conv", False)
        self.ggpo_conv_weight_sample_size = kwargs.get("ggpo_conv_weight_sample_size", 100)

        self.wd_on_output = kwargs.get("wd_on_output", True)

        if self.ggpo_beta is not None:
            self.ggpo_beta = float(self.ggpo_beta)

        if self.ggpo_sigma is not None:
            self.ggpo_sigma = float(self.ggpo_sigma)

        if self.ggpo_conv is not None:
            self.ggpo_conv = bool(self.ggpo_conv)

        if self.ggpo_conv_weight_sample_size is not None:
            self.ggpo_conv_weight_sample_size = int(self.ggpo_conv_weight_sample_size)

        # 初始化LoRA+相关属性
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if not self.ENABLE_CONV:
            conv_lora_dim = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            logger.info("Apply different lora dim for conv layer")
            logger.info(f"Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}")
        elif self.conv_lora_dim == 0:
            logger.info("Disable conv layer")

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            logger.info("Apply different alpha value for conv layer")
            logger.info(f"Conv alpha: {conv_alpha}, Linear alpha: {alpha}")

        if 1 >= dropout >= 0:
            logger.info(f"Use Dropout value: {dropout}")

        if self.wd_on_output is not None:
            logger.info(f"wd_on_output={self.wd_on_output}")

        if self.loraplus_lr_ratio is not None:
            logger.info(f"loraplus_lr_ratio={self.loraplus_lr_ratio}")

        if self.loraplus_unet_lr_ratio is not None:
            logger.info(f"loraplus_unet_lr_ratio={self.loraplus_unet_lr_ratio}")

        if self.loraplus_text_encoder_lr_ratio is not None:
            logger.info(f"loraplus_text_encoder_lr_ratio={self.loraplus_text_encoder_lr_ratio}")


        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.use_tucker = use_tucker

        # Merge preset values with network arg values (network args take priority)
        effective_exclude_patterns = exclude_patterns if exclude_patterns is not None else self.EXCLUDE_PATTERNS
        effective_include_patterns = include_patterns if include_patterns is not None else self.INCLUDE_PATTERNS
        effective_reg_dims = reg_dims if reg_dims is not None else self.REG_DIMS
        effective_reg_lrs = reg_lrs if reg_lrs is not None else self.REG_LRS

        # Store reg_dims and reg_lrs
        self.reg_dims = effective_reg_dims
        self.reg_lrs = effective_reg_lrs

        # Compile include/exclude regex patterns
        def _compile_patterns(patterns):
            compiled = []
            if patterns:
                for p in patterns:
                    try:
                        compiled.append(re.compile(p))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern '{p}': {e}")
            return compiled

        exclude_re_patterns = _compile_patterns(effective_exclude_patterns)
        include_re_patterns = _compile_patterns(effective_include_patterns)

        if exclude_re_patterns:
            logger.info(f"Exclude patterns: {[p.pattern for p in exclude_re_patterns]}")
        if include_re_patterns:
            logger.info(f"Include patterns (override exclude): {[p.pattern for p in include_re_patterns]}")
        if self.reg_dims:
            logger.info(f"Regex-specific dimensions: {self.reg_dims}")
        if self.reg_lrs:
            logger.info(f"Regex-specific learning rates: {self.reg_lrs}")

        def _is_excluded(full_name):
            """Check if a module name should be excluded, respecting include overrides.
            exclude_patterns takes precedence over exclude_name/TARGET_EXCLUDE_NAME."""
            excluded = False
            if exclude_re_patterns:
                excluded = any(p.fullmatch(full_name) for p in exclude_re_patterns)
            elif self.TARGET_EXCLUDE_NAME and (
                full_name in self.TARGET_EXCLUDE_NAME
                or any(self.match_fn(t, full_name) for t in self.TARGET_EXCLUDE_NAME)
            ):
                excluded = True
            if excluded and include_re_patterns:
                if any(p.fullmatch(full_name) for p in include_re_patterns):
                    excluded = False
            return excluded

        def _get_reg_dim(full_name):
            """Check if a module name matches any reg_dims regex and return the override dim."""
            if self.reg_dims:
                for reg_pattern, d in self.reg_dims.items():
                    if re.fullmatch(reg_pattern, full_name):
                        logger.info(f"Module {full_name} matched regex '{reg_pattern}' -> dim: {d}")
                        return d
            return None

        def create_single_module(
            lora_name: str,
            module: torch.nn.Module,
            algo_name,
            dim=None,
            alpha=None,
            use_tucker=self.use_tucker,
            **kwargs,
        ):
            for k, v in root_kwargs.items():
                if k in kwargs:
                    continue
                kwargs[k] = v

            if train_norm and "Norm" in module.__class__.__name__:
                return norm_modules(
                    lora_name=lora_name,
                    org_module=module,
                    multiplier=self.multiplier,
                    rank_dropout=self.rank_dropout,
                    module_dropout=self.module_dropout,
                    **kwargs,
                )
            lora = None
            if isinstance(module, (torch.nn.Linear, CPUBouncingLinear)) and lora_dim > 0:
                dim = dim or lora_dim
                alpha = alpha or self.alpha
            elif isinstance(
                module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                k_size, *_ = module.kernel_size
                if k_size == 1 and lora_dim > 0:
                    dim = dim or lora_dim
                    alpha = alpha or self.alpha
                elif conv_lora_dim > 0 or dim:
                    dim = dim or conv_lora_dim
                    alpha = alpha or self.conv_alpha
                else:
                    return None
            else:
                return None
            lora = network_module_dict[algo_name](
                lora_name,
                module,
                self.multiplier,
                dim,
                alpha,
                self.dropout,
                self.rank_dropout,
                self.module_dropout,
                use_tucker,
                **kwargs,
            )
            return lora

        def create_modules_(
            prefix: str,
            root_module: torch.nn.Module,
            algo,
            configs={},
            full_prefix: str = "",
        ):
            loras = {}
            lora_names = []
            for name, module in root_module.named_modules():
                full_name = (
                    f"{full_prefix}.{name}" if full_prefix and name else (full_prefix or name)
                )
                if _is_excluded(full_name):
                    continue

                module_name = module.__class__.__name__
                if module_name in self.MODULE_ALGO_MAP and module is not root_module:
                    next_config = self.MODULE_ALGO_MAP[module_name]
                    next_algo = next_config.get("algo", algo)
                    new_loras, new_lora_names = create_modules_(
                        f"{prefix}_{name}",
                        module,
                        next_algo,
                        next_config,
                        full_prefix=full_name,
                    )
                    for lora_name, lora in zip(new_lora_names, new_loras):
                        if lora_name not in loras:
                            loras[lora_name] = lora
                            lora_names.append(lora_name)
                    continue
                if name:
                    lora_name = prefix + "." + name
                else:
                    lora_name = prefix
                lora_name = lora_name.replace(".", "_")
                if lora_name in loras:
                    continue

                module_configs = dict(configs)
                reg_dim = _get_reg_dim(full_name)
                if reg_dim is not None:
                    module_configs['dim'] = reg_dim

                lora = create_single_module(lora_name, module, algo, **module_configs)
                if lora is not None:
                    lora.original_name = full_name
                    loras[lora_name] = lora
                    lora_names.append(lora_name)
            return [loras[lora_name] for lora_name in lora_names], lora_names

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
        ) -> List:
            logger.info("Create LyCORIS Module")
            loras = []
            next_config = {}
            for name, module in root_module.named_modules():
                if _is_excluded(name):
                    continue

                module_name = module.__class__.__name__
                if module_name in target_replace_modules and not any(
                    self.match_fn(t, name) for t in target_replace_names
                ):
                    if module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    loras.extend(
                        create_modules_(
                            f"{prefix}_{name}",
                            module,
                            algo,
                            next_config,
                            full_prefix=name,
                        )[0]
                    )
                    next_config = {}
                elif name in target_replace_names or any(
                    self.match_fn(t, name) for t in target_replace_names
                ):
                    conf_from_name = self.find_conf_for_name(name)
                    if conf_from_name is not None:
                        next_config = conf_from_name
                        algo = next_config.get("algo", network_module)
                    elif module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    module_configs = dict(next_config)
                    reg_dim = _get_reg_dim(name)
                    if reg_dim is not None:
                        module_configs['dim'] = reg_dim
                    lora = create_single_module(lora_name, module, algo, **module_configs)
                    next_config = {}
                    if lora is not None:
                        lora.original_name = name
                        loras.append(lora)
            return loras

        if network_module == GLoRAModule:
            logger.info("GLoRA enabled, only train transformer")
            # only train transformer (for GLoRA)
            LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = []

        self.text_encoder_loras = []
        if text_encoder:
            if isinstance(text_encoder, list):
                text_encoders = text_encoder
                use_index = True
            else:
                text_encoders = [text_encoder]
                use_index = False

            for i, te in enumerate(text_encoders):
                self.text_encoder_loras.extend(
                    create_modules(
                        LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
                        + (f"{i+1}" if use_index else ""),
                        te,
                        LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                        LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME,
                    )
                )
            logger.info(
                f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules."
            )

        unet_target_modules = list(LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE)
        if self.train_llm_adapter:
            unet_target_modules.append("LLMAdapterTransformerBlock")

        self.unet_loras = create_modules(
            LycorisNetworkKohya.LORA_PREFIX_UNET,
            unet,
            unet_target_modules,
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME,
        )
        logger.info(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        algo_table = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        logger.info(f"module type table: {algo_table}")

        self.weights_sd = None

        self.loras = self.text_encoder_loras + self.unet_loras
        # assertion
        names = set()
        for lora in self.loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        self._setup_ramtorch_device_handling()

    def _setup_ramtorch_device_handling(self):
        """Ensure RamTorch modules stay on CPU while LoRA adapters are on GPU"""
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        for lora in self.loras:
            if lora.is_ramtorch_org:
                # Move parameters to GPU

                # Lora/locon
                if hasattr(lora, 'lora_up'):
                    lora.lora_up.to(device)
                if hasattr(lora, 'lora_down'):
                    lora.lora_down.to(device)
                if hasattr(lora, 'lora_mid'):
                    lora.lora_mid.to(device)

                # Glora
                if hasattr(lora, 'a1'):
                    lora.a1.to(device)
                if hasattr(lora, 'a2'):
                    lora.a2.to(device)
                if hasattr(lora, 'b1'):
                    lora.b1.to(device)
                if hasattr(lora, 'b2'):
                    lora.b2.to(device)
                if hasattr(lora, 'bm'):
                    lora.bm.to(device)

                # oft/boft
                if hasattr(lora, 'oft_blocks'):
                    lora.oft_blocks.to(device)
                if hasattr(lora, 'rescale'):
                    lora.rescale.to(device)

                # lokr
                if hasattr(lora, 'lokr_w1'):
                    lora.lokr_w1.to(device)
                if hasattr(lora, 'lokr_w1_a'):
                    lora.lokr_w1_a.to(device)
                if hasattr(lora, 'lokr_w1_b'):
                    lora.lokr_w1_b.to(device)
                if hasattr(lora, 'lokr_w2'):
                    lora.lokr_w2.to(device)
                if hasattr(lora, 'lokr_w2_a'):
                    lora.lokr_w2_a.to(device)
                if hasattr(lora, 'lokr_w2_b'):
                    lora.lokr_w2_b.to(device)
                if hasattr(lora, 'lokr_t1'):
                    lora.lokr_t1.to(device)
                if hasattr(lora, 'lokr_t2'):
                    lora.lokr_t2.to(device)

                # loha
                if hasattr(lora, 'hada_w1_a'):
                    lora.hada_w1_a.to(device)
                if hasattr(lora, 'hada_w1_b'):
                    lora.hada_w1_b.to(device)
                if hasattr(lora, 'hada_w2_a'):
                    lora.hada_w2_a.to(device)
                if hasattr(lora, 'hada_w2_b'):
                    lora.hada_w2_b.to(device)
                if hasattr(lora, 'hada_t1'):
                    lora.hada_t1.to(device)
                if hasattr(lora, 'hada_t2'):
                    lora.hada_t2.to(device)

                #abba
                if hasattr(lora, 'lora_up1'):
                    lora.lora_up1.to(device)
                if hasattr(lora, 'lora_down1'):
                    lora.lora_down1.to(device)
                if hasattr(lora, 'lora_up2'):
                    lora.lora_up2.to(device)
                if hasattr(lora, 'lora_down2'):
                    lora.lora_down2.to(device)

                #tlora
                if hasattr(lora, 'q_layer'):
                    lora.q_layer.to(device)
                if hasattr(lora, 'p_layer'):
                    lora.p_layer.to(device)
                if hasattr(lora, 'lambda_layer'):
                    lora.lambda_layer.data = lora.lambda_layer.data.to(device)
                if hasattr(lora, 'base_q'):
                    lora.base_q = lora.base_q.to(device)
                if hasattr(lora, 'base_p'):
                    lora.base_p = lora.base_p.to(device)
                if hasattr(lora, 'base_lambda'):
                    lora.base_lambda = lora.base_lambda.to(device)

                #ia3
                if hasattr(lora, 'weight'):
                    lora.weight.to(device)

                #full
                if hasattr(lora, 'diff'):
                    lora.diff.to(device)
                if hasattr(lora, 'diff_b'):
                    lora.diff_b.to(device)

                #dylora
                if hasattr(lora, 'up_list'):
                    lora.up_list.to(device)
                if hasattr(lora, 'down_list'):
                    lora.down_list.to(device)
                
                #dora
                if hasattr(lora, 'dora_scale'):
                    lora.dora_scale.to(device)
            

                #scalar
                if hasattr(lora, 'scalar'):
                    lora.scalar.to(device)

                # Keep original module on CPU
                lora.org_module[0].cpu()
                
                # Ensure dtype consistency
                lora.dtype_tensor = lora.dtype_tensor.to(device)
            else:
                # Standard device placement
                lora.to(device)

    def match_fn(self, pattern: str, name: str) -> bool:
        if self.USE_FNMATCH:
            return fnmatch.fnmatch(name, pattern)
        return re.match(pattern, name)

    def find_conf_for_name(
        self,
        name: str,
    ) -> dict[str, Any]:
        if name in self.NAME_ALGO_MAP.keys():
            return self.NAME_ALGO_MAP[name]

        for key, value in self.NAME_ALGO_MAP.items():
            if self.match_fn(key, name):
                return value

        return None

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(self.weights_sd, strict=False)
        state = {}
        if missing:
            state["missing keys"] = missing
        if unexpected:
            state["unexpected keys"] = unexpected
        return state

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        assert (
            apply_text_encoder is not None and apply_unet is not None
        ), f"internal error: flag not set"

        if apply_text_encoder:
            logger.info("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras

        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            logger.info(f"weights are loaded: {info}")

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras
        super().merge_to(1)

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for module in self.unet_loras + self.text_encoder_loras:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            key_scaled += scaled

        return key_scaled, sum(norms) / len(norms), max(norms)
    
    @torch.no_grad()
    def get_norms(self, device):
        unscaled_norms = []
        for module in self.unet_loras + self.text_encoder_loras:
            unscaled_norm = module.get_norm(device)
            if isinstance(unscaled_norm, torch.Tensor):
                unscaled_norms.append(unscaled_norm)

        return torch.stack(unscaled_norms)

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(
            f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}"
        )
        logger.info(
            f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}"
        )



    def prepare_optimizer_params(self,
                                     text_encoder_lr: Optional[float|int|List[float]|Tuple[float]] = None,
                                     unet_lr: Optional[float] = None,
                                     learning_rate: Optional[float] = None,
                                     apply_orthograd: bool = False,
                                     orthograd_targets: List[str] = [],
                                     ) -> List[Dict[str, Any]]:
        """
        Prepares parameter groups for the optimizer, grouping by major component
        (UNet, TE1, TE2, ...) parsed from parameter names and optionally
        splitting based on OrthoGrad targets.

        Args:
            text_encoder_lr: Learning rate for ALL text encoder LoRA parameters.
                             If different LRs per TE are needed, this logic
                             would need adjustment (e.g., pass a dict/list).
            unet_lr: Learning rate for UNet LoRA parameters.
            learning_rate: Fallback LR if none present.
            apply_orthograd: If True, split parameters within each major component
                             into two groups based on orthograd_targets. One group
                             will have {'orthograd': True}, the other {'orthograd': False}.
                             If False, each component gets one group {'orthograd': False}.
            orthograd_targets: A list of strings. Parameter names containing any
                               of these strings will be assigned to the
                               'orthograd': True group if apply_orthograd is True.
                               Usually targets weights like '.lora_down.weight', '.lora_up.weight'.

        Returns:
            A list of parameter group dictionaries suitable for a PyTorch optimizer.
        """
        found_te_ids = set()

        self.requires_grad_(True) # Ensure grads are enabled

        # Temporary storage: key=(component_type, component_index, is_ortho_target)
        # component_type = 'unet' or 'te'
        # component_index = 0 for unet, 1, 2, ... for te
        # is_ortho_target = True or False
        grouped_params = defaultdict(list)

        # Regex to find 'te' followed by digits
        te_regex = re.compile(r'lora_te(\d+)_')

        # Build reg_lr lookup: lora_name -> reg_lr_idx
        reg_lr_lookup = {}  # lora_name -> reg_lr_idx
        reg_lr_values = {}  # reg_lr_idx -> lr_value
        if self.reg_lrs:
            reg_lrs_list = list(self.reg_lrs.items())
            for i, (_, lr_val) in enumerate(reg_lrs_list):
                reg_lr_values[i] = lr_val
            for lora in self.loras:
                if hasattr(lora, 'original_name'):
                    for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                        if re.fullmatch(regex_str, lora.original_name):
                            reg_lr_lookup[lora.lora_name] = i
                            logger.info(f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                            break

        # Iterate through all named parameters of the model
        for name, param in self.named_parameters():
            comp_type = 'unet' # Default to unet
            comp_idx = 0       # Default index for unet

            # Check if the name matches the text encoder pattern
            match = te_regex.search(name)
            if match:
                comp_type = 'textencoder'
                comp_idx = int(match.group(1)) # Extract the number (e.g., 1 from 'te1')
                found_te_ids.add(comp_idx)
            # else: Parameter remains classified as 'unet', comp_idx 0

            # Determine if this parameter name contains any of the target strings
            is_target = any(target in name for target in orthograd_targets)

            # Determine if this parameter should go into the OrthoGrad=True group
            is_ortho_group = apply_orthograd and is_target

            is_lora_plus = (name is not None and any(target in name for target in LORA_PLUS_TARGETS) and
                            ((comp_type == 'textencoder' and (self.loraplus_text_encoder_lr_ratio is not None or self.loraplus_lr_ratio is not None)) or 
                             (comp_type == 'unet' and (self.loraplus_unet_lr_ratio is not None or self.loraplus_lr_ratio is not None))))

            # Check if parameter belongs to a module with regex-specific LR
            lora_module_name = name.split('.')[0]
            reg_lr_idx = reg_lr_lookup.get(lora_module_name)

            # Assign the parameter to the correct temporary list
            group_key = (comp_type, comp_idx, is_ortho_group, is_lora_plus, reg_lr_idx)
            grouped_params[group_key].append(param)

        num_of_te = len(found_te_ids)

        # make sure text_encoder_lr as list of two elements
        # if float, use the same value for both text encoders
        # Condition 1: None or empty list/tuple
        if text_encoder_lr is None or (isinstance(text_encoder_lr, (list, tuple)) and len(text_encoder_lr) == 0):
            text_encoder_lr = [learning_rate] * num_of_te

        # Condition 2: Single number (int or float)
        elif isinstance(text_encoder_lr, numbers.Number): # Check if it's a number (int, float, etc.)
            text_encoder_lr = [float(text_encoder_lr)] * num_of_te # Ensure float values

        # Condition 3: List or tuple, and its length is less than num_of_te
        elif isinstance(text_encoder_lr, (list, tuple)) and len(text_encoder_lr) < num_of_te:
            # Convert to list (if it was a tuple) and pad
            padding_needed = num_of_te - len(text_encoder_lr)
            text_encoder_lr = list(text_encoder_lr) + [learning_rate] * padding_needed

        # --- Construct Final Parameter Groups ---
        all_param_groups = []
        all_lr_descriptions = []
        for (comp_type, comp_idx, is_ortho_group, is_lora_plus, reg_lr_idx), params in grouped_params.items():

            # Determine Learning Rate for this group
            current_lr = None
            if reg_lr_idx is not None:
                # Use regex-specific learning rate
                base_lr = reg_lr_values[reg_lr_idx]
                if is_lora_plus:
                    ratio = None
                    if comp_type == 'unet':
                        ratio = self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio
                    elif comp_type == 'textencoder':
                        ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
                    current_lr = base_lr * ratio if ratio else base_lr
                else:
                    current_lr = base_lr
            elif comp_type == 'unet':
                if is_lora_plus:
                    current_lr = ((unet_lr if unet_lr is not None else learning_rate) * 
                                  (self.loraplus_unet_lr_ratio if self.loraplus_unet_lr_ratio is not None else self.loraplus_lr_ratio))
                else:
                    current_lr = unet_lr if unet_lr is not None else learning_rate
            elif comp_type == 'textencoder':
                if is_lora_plus:
                    current_lr = ((text_encoder_lr[comp_idx - 1] * 
                                   (self.loraplus_text_encoder_lr_ratio if self.loraplus_text_encoder_lr_ratio is not None else self.loraplus_lr_ratio)))
                else:
                    current_lr = text_encoder_lr[comp_idx - 1]
                
            group_name_prefix = f"{comp_type}{comp_idx if comp_type == 'textencoder' else ''}"
            reg_lr_suffix = f"_reg{reg_lr_idx}" if reg_lr_idx is not None else ""
            if current_lr is None or current_lr <= 0.0:
                # We won't train groups that lack a LR or have a lr <= 0.0
                logger.warning(f"Not training {group_name_prefix}{reg_lr_suffix} as LR is {str(current_lr)}.")
                continue

            lr_description = f"{comp_type}{comp_idx if comp_type == 'textencoder' else ''}{' Ortho' if is_ortho_group else ''}{' plus' if is_lora_plus else ''}{' reg_lr_' + str(reg_lr_idx) if reg_lr_idx is not None else ''}"

            group_dict = {
                'params': params,
                'lr_description': lr_description,
                'is_ortho_group': is_ortho_group,
                'is_lora_plus_group': is_lora_plus,
                'lr': torch.tensor(current_lr),
            }

            group_name = f"{group_name_prefix}{reg_lr_suffix}{'_Ortho' if is_ortho_group else ''}{'_Plus' if is_lora_plus else ''}"
            group_dict['name'] = group_name
            all_param_groups.append(group_dict)
            all_lr_descriptions.append(lr_description)

        logger.info(f"Training the following {len(all_param_groups)} parameter groups:")
        # Sort groups for consistent print order (optional)
        all_param_groups.sort(key=lambda g: g.get('name', ''))
        for i, group in enumerate(all_param_groups):
             logger.info(f"  Group {i} ('{group.get('name', 'Unnamed')}'): "
                   f"is_ortho_group={group.get('is_ortho_group', 'N/A')}, "
                   f"is_lora_plus_group={group.get('is_lora_plus_group', 'N/A')}, "
                   f"lr={group.get('lr', 'Default')}, "
                   f"NumParams={len(group['params'])}")

        if not all_param_groups:
             raise Exception("No parameter groups were created. Check model parameters and targets.")

        return all_param_groups, all_lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, *args):
        self.requires_grad_(True)

    def on_epoch_start(self, *args):
        self.train()

    def on_step_start(self, *args):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash = precalculate_safetensors_hashes(state_dict)
            metadata["sshs_model_hash"] = model_hash
            metadata["wd_on_output"] = str(self.wd_on_output)

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
