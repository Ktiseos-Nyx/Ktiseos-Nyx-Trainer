from functools import cache
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisBaseModule
from ..functional import power2factorization
from ..logging import logger

from typing import Optional

@cache
def log_butterfly_factorize(dim, factor, result):
    logger.info(
        f"Use BOFT(num_stages={int(log2(result[1]))}, elementary_block_size={result[0]})"
        f" (equivalent to factor={result[0]}) "
        f"for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    m, n = power2factorization(dimension, factor)

    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constraints."
        )

    log_butterfly_factorize(dimension, factor, (m, n))
    return m, n


class ButterflyOFTModule(LycorisBaseModule):
    name = "boft"
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
            raise ValueError(f"{self.module_type} is not supported in BOFT algo.")

        out_dim = self.dim
        b, m_exp = butterfly_factor(out_dim, lora_dim)
        self.block_size = b
        self.block_num = m_exp
        # BOFT(m, b)
        self.boft_b = b
        self.boft_m = int(log2(m_exp))
        # block_num > block_size
        self.rescaled = rescaled
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )

        self.register_buffer("I", torch.eye(self.block_size, device=self.oft_blocks.device, dtype=self.oft_blocks.dtype))
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        if f"{lora_name}.oft_blocks" in state_dict:
            oft_blocks = state_dict[f"{lora_name}.oft_blocks"]
            if oft_blocks.ndim == 4:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, oft_blocks, rescale, alpha
    ):
        m, n, s, _ = oft_blocks.shape
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
        q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
        normed_q = q
        # Diag OFT style constrain
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
        Applies multiplicative dropout to the orthogonal matrices r.
        Selected components (dim 0) or blocks within components (dim 1)
        are replaced by identity matrices during training.

        Args:
            r (torch.Tensor): The orthogonal matrices computed by get_r(),
                              shape (boft_m, block_num, block_size, block_size).

        Returns:
            torch.Tensor: The dropout-applied orthogonal matrices.
        """
        if not self.training or (self.dropout == 0):
            return r

        m, n, b, _ = r.shape
        device = r.device
        dtype = r.dtype
        identity_matrix = self.I.to(device=device, dtype=dtype) # Shape (b, b)

        # Create masks
        # Mask for components (True = keep, False = replace with I)
        comp_mask = torch.rand(m, device=device) >= self.dropout
        # Mask for blocks within components (True = keep, False = replace with I)
        block_mask = torch.rand(m, n, device=device) >= self.dropout

        # Combine masks: Keep if component is kept AND block is kept
        # Shape: (m, n) -> (m, n, 1, 1) for broadcasting with r
        keep_mask = (comp_mask.unsqueeze(1) & block_mask).view(m, n, 1, 1)

        # Use torch.where to select between original r and identity
        # Expand identity to match r's shape for torch.where
        # identity_expanded = identity_matrix.unsqueeze(0).unsqueeze(0).expand_as(r) # More explicit expand
        # r_dropped = torch.where(keep_mask, r, identity_expanded)

        # Alternative using broadcasting (more efficient):
        # torch.where expects condition and tensors to be broadcastable.
        # keep_mask (m, n, 1, 1) is broadcastable with r (m, n, b, b)
        # identity_matrix (b, b) is broadcastable with r (m, n, b, b)
        r_dropped = torch.where(keep_mask, r, identity_matrix)

        return r_dropped

    def make_weight(self, scale=1, device=None, diff=False):
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        
        r = self._apply_multiplicative_dropout(r)
        
        # Ensure org_weight is on the correct device and dtype early
        org_weight = self.get_org_weight_for_compute(device)
        org_weight_dtype = org_weight.dtype
        r_dtype = r.dtype # Usually float32 due to inverse, ensure consistency
        target_dtype = torch.promote_types(org_weight_dtype, r_dtype)

        if device is None:
            device = self.oft_blocks.device
            org_weight
        inp = org = org_weight.to(target_dtype, non_blocking=True)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I.to(device=bi.device, dtype=bi.dtype)
            inp = (
                inp.unflatten(0, (-1, g, k))
                .transpose(1, 2)
                .flatten(0, 2)
                .unflatten(0, (-1, b))
            )
            inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
            inp = (
                inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)
            )

        if self.rescaled:
            inp = inp * self.rescale.to(device=inp.device, dtype=inp.dtype)

        if diff:
            inp = inp - org

        return inp.to(org_weight_dtype)

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
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        r = self._apply_multiplicative_dropout(r)
        inp = org = self.org_forward(x)
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I.to(device=bi.device, dtype=bi.dtype)
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, b))
            )
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = (
                inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)
            )

        if self.rescaled:
            inp = inp * self.rescale.to(device=inp.device, dtype=inp.dtype).transpose(0, -1)

        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        if diff:
            inp = inp - org
        return inp

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x, *args, **kwargs):
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

