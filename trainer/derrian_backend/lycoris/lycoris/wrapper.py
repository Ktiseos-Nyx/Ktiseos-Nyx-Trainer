# General LyCORIS wrapper based on kohya-ss/sd-scripts' style
import os
import fnmatch
import re
import logging

from typing import Any, List

import numpy as np

import torch
import torch.nn as nn

import math

from .utils import precalculate_safetensors_hashes
from .modules.locon import LoConModule
from .modules.loha import LohaModule
from .modules.lokr import LokrModule
from .modules.dylora import DyLoraModule
from .modules.glora import GLoRAModule
from .modules.norms import NormModule
from .modules.full import FullModule
from .modules.diag_oft import DiagOFTModule
from .modules.boft import ButterflyOFTModule
from .modules import get_module, make_module

from .config import PRESET
from .utils.preset import read_preset
from .utils import str_bool
from .logging import logger

from typing import Optional


VALID_PRESET_KEYS = [
    "enable_conv",
    "target_module",
    "target_name",
    "module_algo_map",
    "name_algo_map",
    "lora_prefix",
    "use_fnmatch",
    "unet_target_module",
    "unet_target_name",
    "text_encoder_target_module",
    "text_encoder_target_name",
    "exclude_name",
]


network_module_dict = {
    "lora": LoConModule,
    "locon": LoConModule,
    "loha": LohaModule,
    "lokr": LokrModule,
    "dylora": DyLoraModule,
    "glora": GLoRAModule,
    "full": FullModule,
    "diag-oft": DiagOFTModule,
    "boft": ButterflyOFTModule,
}
deprecated_arg_dict = {
    "disable_conv_cp": "use_tucker",
    "use_cp": "use_tucker",
    "use_conv_cp": "use_tucker",
    "constrain": "constraint",
}


def create_lycoris(module, multiplier=1.0, linear_dim=4, linear_alpha=1, **kwargs):
    for key, value in list(kwargs.items()):
        if key in deprecated_arg_dict:
            logger.warning(
                f"{key} is deprecated. Please use {deprecated_arg_dict[key]} instead.",
                stacklevel=2,
            )
            kwargs[deprecated_arg_dict[key]] = value
    if linear_dim is None:
        linear_dim = 4  # default
    conv_dim = int(kwargs.get("conv_dim", linear_dim) or linear_dim)
    conv_alpha = float(kwargs.get("conv_alpha", linear_alpha) or linear_alpha)
    dropout = float(kwargs.get("dropout", 0.0) or 0.0)
    rank_dropout = float(kwargs.get("rank_dropout", 0.0) or 0.0)
    module_dropout = float(kwargs.get("module_dropout", 0.0) or 0.0)
    lora_dropout = float(kwargs.get("lora_dropout", 0.0) or 0.0)
    aid_dropout = float(kwargs.get("aid_dropout", 0.0) or 0.0)
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
    constraint = float(kwargs.get("constraint", 0) or 0)
    rescaled = str_bool(kwargs.get("rescaled", False))
    weight_decompose = str_bool(kwargs.get("dora_wd", False))
    wd_on_output = str_bool(kwargs.get("wd_on_output", False))
    full_matrix = str_bool(kwargs.get("full_matrix", False))
    bypass_mode = str_bool(kwargs.get("bypass_mode", False))
    rs_lora = str_bool(kwargs.get("rs_lora", False))
    unbalanced_factorization = str_bool(kwargs.get("unbalanced_factorization", False))

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

    if unbalanced_factorization:
        logger.info("Unbalanced factorization for LoKr is enabled")

    if bypass_mode:
        logger.info("Bypass mode is enabled")

    if weight_decompose:
        logger.info("Weight decomposition is enabled")

    if bypass_mode and weight_decompose:
        bypass_mode = False
        logger.info("Because weight decomposition (DoRA) is enabled, bypass mode has been disabled")
    elif bypass_mode:
        logger.info("Bypass mode is enabled")

    if full_matrix:
        logger.info("Full matrix mode for LoKr is enabled")

    if lora_dropout is not None:
        lora_dropout = float(lora_dropout)

    if aid_dropout is not None:
        aid_dropout = float(aid_dropout)

    preset = kwargs.get("preset", "full")
    if preset not in PRESET:
        preset = read_preset(preset)
    else:
        preset = PRESET[preset]
    assert preset is not None
    LycorisNetwork.apply_preset(preset)

    logger.info(f"Using rank adaptation algo: {algo}")

    network = LycorisNetwork(
        module,
        multiplier=multiplier,
        lora_dim=linear_dim,
        conv_lora_dim=conv_dim,
        alpha=linear_alpha,
        conv_alpha=conv_alpha,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        lora_dropout=lora_dropout,
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
        wd_on_out=wd_on_output,
        full_matrix=full_matrix,
        bypass_mode=bypass_mode,
        rs_lora=rs_lora,
        unbalanced_factorization=unbalanced_factorization,
        ggpo_beta=ggpo_beta,
        ggpo_sigma=ggpo_sigma,
        ggpo_conv_weight_sample_size=ggpo_conv_weight_sample_size,
    )

    return network


def create_lycoris_from_weights(multiplier, file, module, weights_sd=None, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    loras = {}
    for key in weights_sd:
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        loras[lora_name] = None

    for name, modules in module.named_modules():
        lora_name = f"{LycorisNetwork.LORA_PREFIX}_{name}".replace(".", "_")
        if lora_name in loras:
            loras[lora_name] = modules

    original_level = logger.level
    logger.setLevel(logging.ERROR)
    network = LycorisNetwork(module, init_only=True)
    network.multiplier = multiplier
    network.loras = []
    logger.setLevel(original_level)

    logger.info("Loading Modules from state dict...")
    for lora_name, orig_modules in loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.loras.append(module)
            network.algo_table[module.__class__.__name__] = (
                network.algo_table.get(module.__class__.__name__, 0) + 1
            )
    logger.info(f"{len(network.loras)} Modules Loaded")

    for lora in network.loras:
        lora.multiplier = multiplier

    return network, weights_sd


class LycorisNetwork(torch.nn.Module):
    ENABLE_CONV = True
    TARGET_REPLACE_MODULE = [
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "GroupNorm",
        "LayerNorm",
    ]
    TARGET_REPLACE_NAME = []
    LORA_PREFIX = "lycoris"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}
    USE_FNMATCH = False
    TARGET_EXCLUDE_NAME = []

    @classmethod
    def apply_preset(cls, preset):
        for preset_key in preset.keys():
            if preset_key not in VALID_PRESET_KEYS:
                raise KeyError(
                    f'Unknown preset key "{preset_key}". Valid keys: {VALID_PRESET_KEYS}'
                )

        if "enable_conv" in preset:
            cls.ENABLE_CONV = preset["enable_conv"]
        if "target_module" in preset:
            cls.TARGET_REPLACE_MODULE = preset["target_module"]
        if "target_name" in preset:
            cls.TARGET_REPLACE_NAME = preset["target_name"]
        if "module_algo_map" in preset:
            cls.MODULE_ALGO_MAP = preset["module_algo_map"]
        if "name_algo_map" in preset:
            cls.NAME_ALGO_MAP = preset["name_algo_map"]
        if "lora_prefix" in preset:
            cls.LORA_PREFIX = preset["lora_prefix"]
        if "use_fnmatch" in preset:
            cls.USE_FNMATCH = preset["use_fnmatch"]
        if "exclude_name" in preset:
            cls.TARGET_EXCLUDE_NAME = preset["exclude_name"]
        return cls

    def __init__(
        self,
        module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=4,
        alpha=1,
        conv_alpha=1,
        use_tucker=False,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        lora_dropout=0.0,
        aid_dropout=0.0,
        network_module: str = "locon",
        norm_modules=NormModule,
        train_norm=False,
        init_only=False,
        **kwargs,
    ) -> None:
        super().__init__()
        root_kwargs = kwargs
        self.weights_sd = None
        self._current_step = 0

        self.ggpo_beta = kwargs.get("ggpo_beta", None)
        self.ggpo_sigma = kwargs.get("ggpo_sigma", None)
        self.ggpo_conv = kwargs.get("ggpo_conv", False)
        self.ggpo_conv_weight_sample_size = kwargs.get("ggpo_conv_weight_sample_size", 100)
        self.lora_dropout = kwargs.get("lora_dropout", 0.0)
        self.aid_dropout = kwargs.get("aid_dropout", 0.0)

        self.wd_on_output = kwargs.get("wd_on_output", False)

        if self.ggpo_beta is not None:
            self.ggpo_beta = float(self.ggpo_beta)

        if self.ggpo_sigma is not None:
            self.ggpo_sigma = float(self.ggpo_sigma)

        if self.ggpo_conv is not None:
            self.ggpo_conv = bool(self.ggpo_conv)

        if self.ggpo_conv_weight_sample_size is not None:
            self.ggpo_conv_weight_sample_size = int(self.ggpo_conv_weight_sample_size)

        if self.lora_dropout is not None:
            self.lora_dropout  = float(self.lora_dropout)

        if self.aid_dropout is not None:
            self.aid_dropout  = float(self.aid_dropout)

        if init_only:
            self.multiplier = 1
            self.lora_dim = 0
            self.alpha = 1
            self.conv_lora_dim = 0
            self.conv_alpha = 1
            self.dropout = 0
            self.rank_dropout = 0
            self.module_dropout = 0
            self.lora_dropout = 0,
            self.use_tucker = False
            self.loras = []
            self.algo_table = {}
            return
        self.multiplier = multiplier
        self.lora_dim = lora_dim

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

        if 1 >= lora_dropout >= 0:
            logger.info(f"Use LORA Dropout value: {lora_dropout}")

        if 1 >= aid_dropout >= 0:
            logger.info(f"Use AID Dropout value: {aid_dropout}")

        if self.wd_on_output is not None:
            logger.info(f"wd_on_output={self.wd_on_output}")

        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.lora_dropout = lora_dropout
        self.aid_dropout = aid_dropout

        self.use_tucker = use_tucker

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
                    lora_name,
                    module,
                    self.multiplier,
                    self.rank_dropout,
                    self.module_dropout,
                    self.lora_dropout,
                    self.aid_dropout,
                    **kwargs,
                )
            lora = None
            if isinstance(module, torch.nn.Linear) and lora_dim > 0:
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
                self.lora_dropout,
                self.aid_dropout,
                use_tucker,
                self.ggpo_beta,
                self.ggpo_sigma,
                self.ggpo_conv,
                self.ggpo_conv_weight_sample_size,
                **kwargs,
            )
            return lora

        def create_modules_(
            prefix: str,
            root_module: torch.nn.Module,
            algo,
            current_lora_map: dict[str, Any],
            configs={},
        ):
            assert current_lora_map is not None, "No mapping supplied"
            loras = current_lora_map
            lora_names = []
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in self.MODULE_ALGO_MAP and module is not root_module:
                    next_config = self.MODULE_ALGO_MAP[module_name]
                    next_algo = next_config.get("algo", algo)
                    new_loras, new_lora_names, new_lora_map = create_modules_(
                        f"{prefix}_{name}" if name else prefix,
                        module,
                        next_algo,
                        loras,
                        configs=next_config,
                    )
                    loras = {**loras, **new_lora_map}
                    for lora_name, lora in zip(new_lora_names, new_loras):
                        if lora_name not in loras and lora_name not in current_lora_map:
                            loras[lora_name] = lora
                        if lora_name not in lora_names:
                            lora_names.append(lora_name)
                    continue

                if name:
                    lora_name = prefix + "." + name
                else:
                    lora_name = prefix

                if f"{self.LORA_PREFIX}_." in lora_name:
                    lora_name = lora_name.replace(
                        f"{self.LORA_PREFIX}_.",
                        f"{self.LORA_PREFIX}.",
                    )

                lora_name = lora_name.replace(".", "_")
                if lora_name in loras:
                    continue

                lora = create_single_module(lora_name, module, algo, **configs)
                if lora is not None:
                    loras[lora_name] = lora
                    lora_names.append(lora_name)
            return [loras[lora_name] for lora_name in lora_names], lora_names, loras

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
            target_exclude_names=[],
        ) -> List:
            logger.info("Create LyCORIS Module")
            loras = []
            lora_map = {}
            next_config = {}
            for name, module in root_module.named_modules():
                if name in target_exclude_names or any(
                    self.match_fn(t, name) for t in target_exclude_names
                ):
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

                    lora_lst, _, _lora_map = create_modules_(
                        f"{prefix}_{name}",
                        module,
                        algo,
                        lora_map,
                        configs=next_config,
                    )
                    lora_map = {**lora_map, **_lora_map}
                    loras.extend(lora_lst)
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

                    if lora_name in lora_map:
                        continue

                    lora = create_single_module(lora_name, module, algo, **next_config)
                    next_config = {}
                    if lora is not None:
                        lora_map[lora.lora_name] = lora
                        loras.append(lora)
            return loras

        self.loras = create_modules(
            LycorisNetwork.LORA_PREFIX,
            module,
            list(
                set(
                    [
                        *LycorisNetwork.TARGET_REPLACE_MODULE,
                        *LycorisNetwork.MODULE_ALGO_MAP.keys(),
                    ]
                )
            ),
            list(
                set(
                    [
                        *LycorisNetwork.TARGET_REPLACE_NAME,
                        *LycorisNetwork.NAME_ALGO_MAP.keys(),
                    ]
                )
            ),
            target_exclude_names=LycorisNetwork.TARGET_EXCLUDE_NAME,
        )
        logger.info(f"create LyCORIS: {len(self.loras)} modules.")

        algo_table = {}
        for lora in self.loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        logger.info(f"module type table: {algo_table}")

        # Assertion to ensure we have not accidentally wrapped some layers
        # multiple times.
        names = set()
        for lora in self.loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def match_fn(self, pattern: str, name: str) -> bool:
        if self.USE_FNMATCH:
            return fnmatch.fnmatch(name, pattern)
        return bool(re.match(pattern, name))

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

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.loras:
            lora.multiplier = self.multiplier

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

    def apply_to(self):
        """
        Register to modules to the subclass so that torch sees them.
        """
        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            logger.info(f"weights are loaded: {info}")

    def is_mergeable(self):
        return True

    def restore(self):
        for lora in self.loras:
            lora.restore()

    def merge_to(self, weight=1.0):
        for lora in self.loras:
            lora.merge_to(weight)

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for module in self.loras:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            key_scaled += scaled

        return key_scaled, sum(norms) / len(norms), max(norms)
    
    def get_norms(self, device):
        scaled_norms = []
        unscaled_norms = []
        for module in self.loras:
            unscaled_norm, scaled_norm = module.get_norm(device)
            if not (unscaled_norm is None or np.isnan(unscaled_norm) or np.isinf(unscaled_norm)):
                unscaled_norms.append(unscaled_norm)
            if not (scaled_norm is None or np.isnan(scaled_norm) or np.isinf(scaled_norm)):
                scaled_norms.append(scaled_norm)

        return unscaled_norms, scaled_norms

    def enable_gradient_checkpointing(self):
        # not supported
        def make_ckpt(module):
            if isinstance(module, torch.nn.Module):
                module.grad_ckpt = True

        self.apply(make_ckpt)
        pass

    def prepare_optimizer_params(self, lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        param_data = {"params": enumerate_params(self.loras)}
        if lr is not None:
            param_data["lr"] = torch.tensor(lr)
        all_params.append(param_data)
        return all_params

    def prepare_grad_etc(self, *args):
        self.requires_grad_(True)

    def on_epoch_start(self, *args):
        self.train()

    def get_trainable_params(self, *args):
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

            if "sshs_model_hash" not in metadata:
                model_hash = precalculate_safetensors_hashes(state_dict)
                metadata["sshs_model_hash"] = model_hash
            metadata["wd_on_output"] = str(self.wd_on_output)
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    @torch.no_grad()
    def update_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_norms()

    @torch.no_grad()
    def update_grad_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_grad_norms()

    @torch.no_grad()
    def grad_norms(self) -> torch.Tensor:
        """Efficiently collect gradient norms from all modules."""
        # Use cached values when possible
        if hasattr(self, '_cached_grad_norms') and self._grad_norm_cache_step == self._current_step:
            return self._cached_grad_norms
            
        # Collect norms 
        all_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "grad_norms") and lora.grad_norms is not None:
                # Take mean of each module's gradient norms to get a scalar
                try:
                    module_norm = lora.grad_norms.mean().item()
                    if not (math.isnan(module_norm) or math.isinf(module_norm)):
                        all_norms.append(module_norm)
                except:
                    # Skip problematic modules
                    continue
        
        # Create tensor from scalars (very efficient)
        result = torch.tensor(all_norms) if all_norms else torch.tensor([])
        
        # Cache the result
        self._cached_grad_norms = result
        self._grad_norm_cache_step = getattr(self, '_current_step', 0)
        
        return result

    @torch.no_grad()
    def weight_norms(self) -> torch.Tensor:
        """Efficiently collect weight norms from all modules."""
        # Use cached values when possible
        if hasattr(self, '_cached_weight_norms') and self._weight_norm_cache_step == self._current_step:
            return self._cached_weight_norms
            
        # Collect norms efficiently
        all_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "weight_norms") and lora.weight_norms is not None:
                try:
                    module_norm = lora.weight_norms.mean().item()
                    if not (math.isnan(module_norm) or math.isinf(module_norm)):
                        all_norms.append(module_norm)
                except:
                    # Skip problematic modules
                    continue
        
        # Create tensor from scalars
        result = torch.tensor(all_norms) if all_norms else torch.tensor([])
        
        # Cache the result
        self._cached_weight_norms = result
        self._weight_norm_cache_step = getattr(self, '_current_step', 0)
        
        return result

    @torch.no_grad()
    def combined_weight_norms(self) -> torch.Tensor:
        """Efficiently collect combined weight norms from all modules."""
        # Use cached values when possible
        if hasattr(self, '_cached_combined_norms') and self._combined_weight_norm_cache_step == self._current_step:
            return self._cached_combined_norms
            
        # Collect norms efficiently
        all_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "combined_weight_norms") and lora.combined_weight_norms is not None:
                try:
                    module_norm = lora.combined_weight_norms.mean().item()
                    if not (math.isnan(module_norm) or math.isinf(module_norm)):
                        all_norms.append(module_norm)
                except:
                    # Skip problematic modules
                    continue
        
        # Create tensor from scalars
        result = torch.tensor(all_norms) if all_norms else torch.tensor([])
        
        # Cache the result
        self._cached_combined_norms = result
        self._combined_weight_norm_cache_step = getattr(self, '_current_step', 0)
        
        return result
    
    @torch.no_grad()
    def accumulate_grad(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.accumulate_grad()

    @torch.no_grad()
    def sum_grads(self):
        sum_grads = []
        sum_squared_grads = []
        count = 0
        for lora in self.text_encoder_loras + self.unet_loras:
            if lora.sum_grads is not None:
                sum_grads.append(lora.sum_grads)
            if lora.sum_grads is not None:
                sum_squared_grads.append(lora.sum_squared_grads)
            count += lora.grad_count

        return (
            torch.stack(sum_grads) if len(sum_grads) > 0 else torch.tensor([]),
            torch.stack(sum_squared_grads) if len(sum_squared_grads) > 0 else torch.tensor([]),
            count
        )

    @torch.no_grad()
    def gradient_noise_scale(self):
        sum_grads, sum_squared_grads, count = self.sum_grads()

        if count == 0:
            return None, None

        # Calculate mean gradient and mean squared gradient
        mean_grad = torch.mean(sum_grads / count, dim=0)
        mean_squared_grad = torch.mean(sum_squared_grads / count, dim=0)

        # Variance = E[X²] - E[X]²
        variance = mean_squared_grad - mean_grad**2

        # GNS = trace(Σ) / ||μ||²
        # trace(Σ) = sum of variances = count * variance (for uniform variance assumption)
        trace_cov = count * variance
        grad_norm_squared = count * mean_grad**2

        gradient_noise_scale = trace_cov / grad_norm_squared
        # mean_grad = torch.mean(all_grads, dim=0)
        #
        # # Calculate trace of covariance matrix
        # centered_grads = all_grads - mean_grad
        # trace_cov = torch.mean(torch.sum(centered_grads**2, dim=0))
        #
        # # Calculate norm of mean gradient squared
        # grad_norm_squared = torch.sum(mean_grad**2)
        #
        # # Calculate GNS using provided gradient norm squared
        # gradient_noise_scale = trace_cov / grad_norm_squared

        return gradient_noise_scale.item(), variance.item()