from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import factorization
from ..logging import logger

from typing import Optional

@cache
def log_oft_factorize(dim, factor, num, bdim):
    logger.info(
        f"Use OFT(block num: {num}, block dim: {bdim})"
        f" (equivalent to lora_dim={num}) "
        f"for {dim=} and lora_dim={factor=}"
    )


class DiagOFTModule(LycorisBaseModule):
    name = "diag-oft"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "oft_blocks",
        "rescale",
        "alpha",
    ]
    weight_list_det = ["oft_blocks"]

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
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        constraint=0,
        rescaled=False,
        bypass_mode=None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
            ggpo_beta,
            ggpo_sigma
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Diag-OFT algo.")

        out_dim = self.dim
        self.block_size, self.block_num = factorization(out_dim, lora_dim)
        # block_num > block_size
        self.rescaled = rescaled
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.block_num, self.block_size, self.block_size)
        )

        self.register_buffer("I", torch.eye(self.block_size, device=self.oft_blocks.device, dtype=self.oft_blocks.dtype))
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        log_oft_factorize(
            dim=out_dim,
            factor=lora_dim,
            num=self.block_num,
            bdim=self.block_size,
        )

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        if f"{lora_name}.oft_blocks" in state_dict:
            oft_blocks = state_dict[f"{lora_name}.oft_blocks"]
            if oft_blocks.ndim == 3:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, oft_blocks, rescale, alpha
    ):
        n, s, _ = oft_blocks.shape
        module = cls(
            lora_name,
            orig_module,
            1,
            lora_dim=s,
            constraint=float(alpha),
            rescaled=rescale is not None,
        )
        module.oft_blocks.copy_(oft_blocks)
        if rescale is not None:
            module.rescale.copy_(rescale)
        return module

    def get_r(self):
        I = self.I.to(device=self.oft_blocks.device, dtype=self.oft_blocks.dtype)
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        normed_q = q
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        # use float() to prevent unsupported type

        q_float = normed_q.float()
        I_float = I.float()
        r = (I_float + q_float) @ torch.inverse(I_float - q_float)
        return r
    
    def _apply_multiplicative_dropout(self, r: torch.Tensor) -> torch.Tensor:
            """
            Applies multiplicative dropout to the orthogonal block matrices r for DiagOFT.
            Selected blocks (dim 0) are replaced by identity matrices during training.

            Args:
                r (torch.Tensor): The orthogonal matrices computed by get_r(),
                                shape (block_num, block_size, block_size).

            Returns:
                torch.Tensor: The dropout-applied orthogonal matrices.
            """
            # self.dropout is the dropout rate for the blocks.
            # It's initialized from the 'dropout' parameter in the constructor.
            if not self.training or self.dropout == 0:
                return r

            # r has shape (block_num, block_size, block_size)
            num_blocks, block_size, _ = r.shape
            device = r.device
            dtype = r.dtype
            # self.I is already (block_size, block_size)
            identity_matrix = self.I.to(device=device, dtype=dtype)

            # Mask for blocks (True = keep, False = replace with I)
            # We operate on the first dimension (num_blocks)
            # self.dropout is the probability of *dropping* a block (setting to I)
            block_keep_mask = torch.rand(num_blocks, device=device) >= self.dropout
            # Reshape for broadcasting: (num_blocks, 1, 1)
            block_keep_mask = block_keep_mask.view(num_blocks, 1, 1)

            # Use torch.where to select between original r and identity_matrix
            # r: (num_blocks, block_size, block_size)
            # block_keep_mask: (num_blocks, 1, 1) -> broadcasts to (num_blocks, block_size, block_size)
            # identity_matrix: (block_size, block_size) -> broadcasts to (num_blocks, block_size, block_size)
            r_dropped = torch.where(block_keep_mask, r, identity_matrix)

            return r_dropped

    def make_weight(self, scale=1, device=None, diff=False):
        r = self.get_r()
        r = self._apply_multiplicative_dropout(r)

        # Ensure org_weight is on the correct device and dtype early
        org_weight = self.get_org_weight_for_compute(device)

        _, *shape = org_weight.shape

        org_weight_dtype = org_weight.dtype
        r_dtype = r.dtype # Usually float32 due to inverse, ensure consistency
        target_dtype = torch.promote_types(org_weight_dtype, r_dtype)

        if device is None:
            device = self.oft_blocks.device

        org_weight = org_weight.to(target_dtype, non_blocking=True)
        org_weight = org_weight.view(self.block_num, self.block_size, *shape)
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = torch.einsum(
            "k n m, k n ... -> k m ...",
            self.rank_drop(r * scale) - scale * self.I + (0 if diff else self.I),
            org_weight,
        ).view(-1, *shape)
        if self.rescaled:
            weight = self.rescale * weight
            if diff:
                weight = weight + (self.rescale - 1) * org_weight
        return weight.to(org_weight_dtype)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = norm != desired
        if scaled:
            self.oft_blocks *= ratio
            return scaled, orig_norm * ratio
        else:
            return 0, orig_norm
        
    @torch.no_grad()
    def get_norm(self, device=None):
        # Norm before scale determined by alpha / r_factor
        unscaled_norm = self.oft_blocks.norm()
        return unscaled_norm

    def _bypass_forward(self, x, scale=1, diff=False):
        r = self.get_r()
        r = self._apply_multiplicative_dropout(r)

        org_out = self.org_forward(x)
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            org_out = org_out.transpose(1, -1)
        *shape, _ = org_out.shape
        org_out = org_out.view(*shape, self.block_num, self.block_size)
        mask = neg_mask = 1
        if self.dropout != 0 and self.training:
            mask = torch.ones_like(org_out)
            mask = self.drop(mask)
            neg_mask = torch.max(mask) - mask
        oft_out = torch.einsum(
            "k n m, ... k n -> ... k m",
            r * scale * mask + (1 - scale) * self.I * neg_mask,
            org_out,
        )
        if diff:
            out = out - org_out
        out = oft_out.view(*shape, -1)
        if self.rescaled:
            out = self.rescale.transpose(-1, 0) * out
            out = out + (self.rescale.transpose(-1, 0) - 1) * org_out
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            out = out.transpose(1, -1)
        return out

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        if self.bypass_mode:
            return self.bypass_forward(x, scale)
        else:
            w = self.make_weight(scale, x.device)
            
            current_bias = self.get_org_bias_for_compute(x.device)
            if current_bias is not None:
                current_bias = current_bias.to(x.dtype, non_blocking=True)
            
            kw_dict = {**self.kw_dict, "weight": w, "bias": current_bias}

            return self.op(x, **kw_dict)
