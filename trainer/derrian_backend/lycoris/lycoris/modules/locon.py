import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional.general import rebuild_tucker
from ..logging import logger

from typing import Optional

from ..utils.general import lora_dropout_down, lora_dropout_up


@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA) will cause network dropout to be applied to the forward input, "
        "instead of to the layers, as per the DoRA paper."
    )


class LoConModule(LycorisBaseModule):
    name = "locon"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
    ]
    weight_list_det = ["lora_up.weight"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        lora_dropout=0.0,
        aid_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_out=False,
        bypass_mode=None,
        rs_lora=False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            lora_dropout,
            aid_dropout,
            rank_dropout_scale,
            bypass_mode,
            ggpo_beta,
            ggpo_sigma,
            ggpo_conv,
            ggpo_conv_weight_sample_size
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in LoRA/LoCon algo.")
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        if self.module_type.startswith("conv"):
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            use_tucker = use_tucker and any(i != 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op
            if use_tucker and any(i != 1 for i in k_size):
                self.lora_down = self.module(in_dim, lora_dim, 1, bias=False)
                self.lora_mid = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = self.module(lora_dim, out_dim, 1, bias=False)
        elif isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError

        self.wd = weight_decompose
        self.wd_on_out = wd_on_out
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_out:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                        dim=1,
                        keepdim=True,
                    )
                    .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                    .transpose(1, 0)
                ).float()

        if dropout:
            self.dropout = nn.Dropout(dropout)
            if self.wd:
                log_wd()
        else:
            self.dropout = nn.Identity()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.init_ggpo()

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, mid, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            down.size(0),
            float(alpha),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        if self.tucker:
            t = self.lora_mid.weight
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            weight = rebuild_tucker(t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        weight = weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]
        weight = self.org_weight
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    def apply_weight_decompose(self, weight, multiplier=1):
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_out:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1

        return weight * scale

    def custom_state_dict(self):
        destination = {}
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar
        destination["lora_down.weight"] = self.lora_down.weight
        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio
            return scaled, orig_norm * ratio
        else:
            return 0, orig_norm
        
    @torch.no_grad()
    def get_norm(self, device=None):
        # Norm before scale determined by alpha / r_factor
        unscaled_norm = self.make_weight(device).norm()
        # Norm after scale determined by alpha / r_factor
        scaled_norm = unscaled_norm * self.scale
        return unscaled_norm.item(), scaled_norm.item()

    def bypass_forward_diff(self, x, scale=1):
        if self.lora_dropout is not None and self.training and self.lora_dropout > 0:
            mid = lora_dropout_down(self.lora_down.weight, x, dropout_prob=self.lora_dropout)
        else:
            mid = self.lora_down(x)

        if self.tucker:
            mid = self.lora_mid(mid)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        if self.lora_dropout is not None and self.training and self.lora_dropout > 0:
            up = lora_dropout_up(self.lora_up.weight, x, dropout_prob=self.lora_dropout)
        else:
            up = self.lora_up(mid)

        if self.training and self.aid_dropout is not None and self.aid_dropout > 0:
            up = self.aid_drop(up)

        return self.dropout(up * self.scalar * self.scale * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        
        # Check if perturbation is needed - early return if not in training
        apply_ggpo = (self.training and 
                    self.ggpo_sigma is not None and 
                    self.ggpo_beta is not None and 
                    self.combined_weight_norms is not None and 
                    self.grad_norms is not None and
                    (self.module_type == "linear" or self.ggpo_conv))
        
        # Handle bypass mode first - simpler path
        if self.bypass_mode:
            result = self.bypass_forward(x, scale=self.multiplier)
            
            if apply_ggpo:
                with torch.no_grad():
                    perturbation_output = self.ggpo_pertubation(x)
                    
                # Add perturbation to result and return
                result = result + perturbation_output
                    
            return result
        
        # Non-bypass mode with perturbation
        dtype = self.dtype
        
        # Apply lora dropout during weight computation if enabled
        if self.training and ((self.lora_dropout is not None and self.lora_dropout > 0 and not self.wd) or self.tucker or self.rank_dropout):
            # Get the lora weights
            wa = self.lora_up.weight.to(x.device).to(dtype)
            wb = self.lora_down.weight.to(x.device).to(dtype)
            
            if self.training and self.lora_dropout is not None and self.lora_dropout > 0 and not self.wd:
                # Generate dropout masks
                up_mask = torch.bernoulli(
                    torch.ones(wa.shape[0], device=x.device) * (1 - self.lora_dropout)
                )
                down_mask = torch.bernoulli(
                    torch.ones(wb.shape[1], device=x.device) * (1 - self.lora_dropout)
                )
                
                # Apply dropout masks (matching the pattern in lora_dropout_up/down)
                wa_dropped = wa * up_mask.view(-1, 1)
                wb_dropped = wb * down_mask.view(1, -1)
            else:
                wa_dropped = wa
                wb_dropped = wb
            
            # Compute the combined weight
            if self.tucker:
                t = self.lora_mid.weight.to(x.device).to(dtype)
                wa_dropped = wa_dropped.view(wa_dropped.size(0), -1).transpose(0, 1)
                wb_dropped = wb_dropped.view(wb_dropped.size(0), -1)
                diff_weight = rebuild_tucker(t, wa_dropped, wb_dropped)
            else:
                diff_weight = wa_dropped.view(wa_dropped.size(0), -1) @ wb_dropped.view(wb_dropped.size(0), -1)
            
            # Apply additional processing
            diff_weight = diff_weight.view(self.shape)
            if self.training and self.rank_dropout:
                drop = (torch.rand(diff_weight.size(0), device=x.device) > self.rank_dropout).to(
                    diff_weight.dtype
                )
                drop = drop.view(-1, *[1] * len(diff_weight.shape[1:]))
                if self.rank_dropout_scale:
                    drop /= drop.mean()
                diff_weight *= drop
            
            diff_weight = (diff_weight * self.scalar.to(device=x.device)).to(dtype=dtype) * self.scale
        else:
            diff_weight = self.make_weight(x.device).to(dtype) * self.scale
        
        # Apply the weight to the input
        weight = self.org_module[0].weight.data.to(dtype)
        
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, self.multiplier)

            # Additionally apply lora dropout to input if enabled
            if self.training and self.lora_dropout is not None and self.lora_dropout > 0:
                # For weight decomposition, apply the lora dropout directly to input dimensions
                # This creates a mask for input features that simulates the effect of lora_dropout_down
                input_mask = torch.bernoulli(
                    torch.ones(x.shape[-1], device=x.device) * (1 - self.lora_dropout)
                ).to(x.dtype)
                
                # Apply mask to the input - this affects which input features contribute to the output
                x = x * input_mask
                
                # Note: We don't need to simulate lora_dropout_up here because the weight_decompose 
                # approach already handles the output feature scaling differently

            x = self.dropout(x)
        else:
            weight = weight + diff_weight * self.multiplier
        
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        
        # Apply operation with weights
        result = self.op(x, weight, bias, **self.kw_dict)

        if self.training and self.aid_dropout is not None and self.aid_dropout > 0:
            result = self.aid_drop(result)
        
        # Apply GGPO perturbation if needed
        if apply_ggpo:
            with torch.no_grad():
                perturbation_output = self.ggpo_pertubation(x)
                
            # Add perturbation to result
            result = result + perturbation_output
        
        return result

    def ggpo_pertubation(self, x):
        # More efficient scale calculation
        perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (self.ggpo_beta * (self.grad_norms**2))
        perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)
        
        # Optimized perturbation generation based on module type
        if self.module_type == "linear":
            # For linear layers, use efficient matrix multiplication
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            perturbation = perturbation * perturbation_scale_factor.view(-1, 1)
            return x @ perturbation.T
        else:
            # For convolution layers, generate efficient perturbation
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            
            # Apply scaling with efficient broadcasting
            view_shape = [perturbation.shape[0]] + [1] * (len(perturbation.shape) - 1)
            perturbation = perturbation * perturbation_scale_factor.view(*view_shape)
            
            # Use the appropriate convolution operation
            return self.op(x, perturbation, None, **self.kw_dict)