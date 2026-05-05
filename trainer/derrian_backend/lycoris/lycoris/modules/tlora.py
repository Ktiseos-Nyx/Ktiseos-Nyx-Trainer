"""
T-LoRA: Timestep-dependent LoRA with SVD-based orthogonal initialization.

Based on the paper "T-LoRA: Timestep-aware LoRA for Diffusion Models"
Key features:
1. SVD-based orthogonal initialization from original weights or random matrix
2. Learnable singular values (lambda_layer)
3. Timestep-dependent rank masking (set externally via set_timestep_mask)
4. Residual subtraction from base state for zero-shot preservation

The mask is applied as: output = P(Q(x) * λ * mask) - P_base(Q_base(x) * λ_base * mask)
This ensures that when weights haven't changed from init, the output contribution is zero.
"""

import gc
import math
from functools import cache
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger


SigType = Literal["principal", "last", "middle"]

# Thread-local storage for timestep mask (set by training loop)
_timestep_mask_storage: dict[int, torch.Tensor] = {}


def set_timestep_mask(mask: torch.Tensor, group_id: int = 0) -> None:
    """
    Set the timestep mask for T-LoRA modules.

    Call this before the forward pass with the appropriate mask for the current timestep.
    The mask should be shape (1, rank) with 1s for active ranks and 0s for masked ranks.

    Args:
        mask: Binary mask tensor of shape (1, max_rank) or (batch, max_rank)
        group_id: Optional group ID for multi-network scenarios
    """
    _timestep_mask_storage[group_id] = mask


def get_timestep_mask(group_id: int = 0) -> Optional[torch.Tensor]:
    """Get the current timestep mask, or None if not set."""
    return _timestep_mask_storage.get(group_id, None)


def clear_timestep_mask(group_id: int = 0) -> None:
    """Clear the timestep mask after forward pass."""
    _timestep_mask_storage.pop(group_id, None)


def compute_timestep_mask(
    timestep: int,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute a binary rank mask based on timestep.

    At high noise (high timestep): fewer ranks active (structure-level adaptation)
    At low noise (low timestep): more ranks active (detail-level adaptation)

    Args:
        timestep: Current denoising timestep
        max_timestep: Maximum timestep (e.g., 1000)
        max_rank: Maximum rank (all ranks active at t=0)
        min_rank: Minimum rank (active even at t=max_timestep)
        alpha: Scaling exponent (1.0 = linear, >1 = more aggressive at high noise)

    Returns:
        Binary mask of shape (1, max_rank)
    """
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    r = min(r, max_rank)  # Clamp to max_rank
    mask = torch.zeros((1, max_rank))
    mask[:, :r] = 1.0
    return mask


def compute_timestep_mask_batch(
    timesteps: torch.Tensor,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-sample binary rank masks for a batch of timesteps.

    This avoids the bias of using a single mask (e.g. max timestep) for
    the entire batch: each sample gets a mask matching its own noise level.

    Args:
        timesteps: Tensor of shape (B,) with timestep values
        max_timestep: Maximum timestep (e.g., 1000)
        max_rank: Maximum rank (all ranks active at t=0)
        min_rank: Minimum rank (active even at t=max_timestep)
        alpha: Scaling exponent (1.0 = linear, >1 = more aggressive at high noise)

    Returns:
        Binary mask of shape (B, max_rank)
    """
    t = timesteps.float()
    # Active ranks per sample: higher timestep -> fewer ranks
    r_float = ((max_timestep - t) / max_timestep) ** alpha * (max_rank - min_rank) + min_rank
    r_int = r_float.int().clamp(min=min_rank, max=max_rank)  # (B,)

    # Build (B, max_rank) mask: position j is active if j < r_int[i]
    rank_indices = torch.arange(max_rank, device=timesteps.device).unsqueeze(0)  # (1, max_rank)
    mask = (rank_indices < r_int.unsqueeze(1)).float()  # (B, max_rank)
    return mask


@cache
def log_tlora_init():
    return logger.info(
        "T-LoRA: Using SVD-based orthogonal initialization with timestep-dependent rank masking"
    )


class TLoraModule(LycorisBaseModule):
    """
    T-LoRA module with SVD-orthogonal initialization and timestep-dependent rank masking.

    The weight delta is computed as:
        ΔW = P @ diag(λ * mask) @ Q - P_base @ diag(λ_base * mask) @ Q_base

    Where P, Q are orthogonal matrices from SVD, λ are learnable singular values,
    and mask is the timestep-dependent binary mask.
    """

    name = "tlora"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "q_layer.weight",
        "p_layer.weight",
        "lambda_layer",
        "alpha",
        "base_q",
        "base_p",
        "base_lambda",
    ]
    weight_list_det = ["lambda_layer"]  # Unique identifier for T-LoRA

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_tucker: bool = False,
        use_scalar: bool = False,
        rank_dropout_scale: bool = False,
        bypass_mode: bool = None,
        sig_type: SigType = "principal",
        use_data_init: bool = True,
        mask_group_id: int = 0,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        """
        Initialize T-LoRA module.

        Note on alpha scaling: The original T-LoRA paper applies NO alpha/dim
        scaling. To match the original behavior, set alpha=lora_dim (i.e.
        --network_alpha equal to --network_dim). The LyCORIS default alpha=1
        applies a 1/dim scaling factor for consistency with other LyCORIS algos.

        Note on sig_type: The original T-LoRA uses "last" for random init
        (OrthogonalLoRALinearLayer) and "principal" for data-dependent init
        (LOrthogonalLoRALinearLayer). The default here is "principal" to match
        the data-dependent variant (use_data_init=True).

        Args:
            lora_name: Unique name for this LoRA module
            org_module: Original module to wrap
            multiplier: Output multiplier
            lora_dim: Rank of the LoRA decomposition
            alpha: Alpha scaling factor (set equal to lora_dim to match original T-LoRA)
            dropout: Dropout probability
            rank_dropout: Rank-wise dropout probability
            module_dropout: Module-level dropout probability
            rank_dropout_scale: Whether to scale by dropout rate
            bypass_mode: Use bypass forward mode
            sig_type: Which singular vectors to use ("principal", "last", "middle")
            use_data_init: If True, use SVD of original weights; if False, use random matrix
            mask_group_id: Group ID for timestep mask lookup
        """
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
            ggpo_sigma,
            ggpo_conv,
            ggpo_conv_weight_sample_size,
        )

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in T-LoRA algo.")

        self.use_orthogonal_weights = False
        self.lora_dim = lora_dim
        self.sig_type = sig_type
        self.use_data_init = use_data_init
        self.mask_group_id = mask_group_id

        log_tlora_init()

        if rank_dropout:
            logger.warning(
                "T-LoRA: rank_dropout is ignored. The timestep mask provides "
                "structured rank selection; random rank dropout would undermine it."
            )

        # Determine dimensions based on module type
        if self.module_type.startswith("conv"):
            self.isconv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding

            # For convolutions, we work with reshaped weights
            # Original weight shape: (out_channels, in_channels, *kernel_size)
            # We treat it as (out_dim, in_dim * prod(kernel_size)) for SVD
            self.conv_shape = (out_dim, in_dim, *k_size)
            flat_in_dim = in_dim * math.prod(k_size)

            # Q projects from input space to rank space
            # For conv, we use 1x1 convs to avoid kernel complexity in LoRA
            self.q_layer = self.module(in_dim, lora_dim, 1, bias=False)
            self.p_layer = self.module(lora_dim, out_dim, 1, bias=False)

            # Store conv params for the actual forward
            # Q is a 1x1 conv: use original stride for spatial downsampling,
            # but padding=0 and dilation=1 since kernel is 1x1
            self.down_op = self.op
            self.up_op = self.op
            self.kw_dict_down = {
                "stride": stride,
                "padding": (0,) * len(k_size),
                "dilation": (1,) * len(k_size),
                "groups": 1,
            }
            self.kw_dict_up = {
                "stride": (1,) * len(k_size),
                "padding": (0,) * len(k_size),
                "dilation": (1,) * len(k_size),
                "groups": 1,
            }
        else:
            self.isconv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            flat_in_dim = in_dim

            self.q_layer = nn.Linear(in_dim, lora_dim, bias=False)
            self.p_layer = nn.Linear(lora_dim, out_dim, bias=False)
            self.down_op = F.linear
            self.up_op = F.linear

        # Learnable singular values
        self.lambda_layer = nn.Parameter(torch.ones(1, lora_dim))

        # SVD-based orthogonal initialization
        self._initialize_from_svd(org_module, in_dim, out_dim, flat_in_dim)

        # Store frozen base state for residual subtraction
        self.register_buffer("base_q", self.q_layer.weight.data.clone())
        self.register_buffer("base_p", self.p_layer.weight.data.clone())
        self.register_buffer("base_lambda", self.lambda_layer.data.clone())

        # Alpha scaling (same as standard LoRA)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Scalar: learnable global magnitude. Initialized to 1.0 (not 0.0)
        # because T-LoRA's residual subtraction already ensures zero init;
        # scalar=0.0 would create dead gradients.
        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

    def _initialize_from_svd(
        self,
        org_module: nn.Module,
        in_dim: int,
        out_dim: int,
        flat_in_dim: int,
    ) -> None:
        """Initialize Q and P layers using SVD."""
        if self.use_data_init:
            # SVD of original weights (data-dependent init)
            weight = org_module.weight.data.float()
            if self.isconv:
                # Reshape conv weight to 2D: (out_dim, in_dim * kernel_size)
                weight = weight.reshape(out_dim, -1)

            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        else:
            # SVD of random matrix (data-independent init)
            rand_weight = torch.normal(
                mean=0,
                std=1 / self.lora_dim,
                size=(out_dim, flat_in_dim),
            )
            u, s, vh = torch.linalg.svd(rand_weight, full_matrices=False)

        # Select singular vectors based on sig_type
        if self.sig_type == "principal":
            # Use top singular vectors (largest singular values)
            q_init = vh[:self.lora_dim]  # (lora_dim, in_dim)
            p_init = u[:, :self.lora_dim]  # (out_dim, lora_dim)
            lambda_init = s[:self.lora_dim]
        elif self.sig_type == "last":
            # Use bottom singular vectors (smallest singular values)
            q_init = vh[-self.lora_dim:]
            p_init = u[:, -self.lora_dim:]
            lambda_init = s[-self.lora_dim:]
        elif self.sig_type == "middle":
            # Use middle singular vectors
            start_q = (vh.shape[0] - self.lora_dim) // 2
            start_p = (u.shape[1] - self.lora_dim) // 2
            start_s = (s.shape[0] - self.lora_dim) // 2
            q_init = vh[start_q:start_q + self.lora_dim]
            p_init = u[:, start_p:start_p + self.lora_dim]
            lambda_init = s[start_s:start_s + self.lora_dim]
        else:
            raise ValueError(f"Unknown sig_type: {self.sig_type}")

        # Handle case where rank is larger than available singular values
        if q_init.shape[0] < self.lora_dim:
            pad_size = self.lora_dim - q_init.shape[0]
            q_init = F.pad(q_init, (0, 0, 0, pad_size))
            p_init = F.pad(p_init, (0, pad_size))
            lambda_init = F.pad(lambda_init, (0, pad_size), value=1e-6)

        # Assign to layers
        if self.isconv:
            # For conv, q_layer is 1x1 conv: weight shape (lora_dim, in_channels, 1, ...)
            # We need to reshape q_init from (lora_dim, in_dim) to conv format
            kernel_ones = [1] * (len(self.conv_shape) - 2)
            self.q_layer.weight.data = q_init[:, :self.shape[1]].reshape(
                self.lora_dim, self.shape[1], *kernel_ones
            ).contiguous()
            self.p_layer.weight.data = p_init[:self.shape[0], :].reshape(
                self.shape[0], self.lora_dim, *kernel_ones
            ).contiguous()
        else:
            # For linear: q_layer.weight is (lora_dim, in_features)
            # p_layer.weight is (out_features, lora_dim)
            self.q_layer.weight.data = q_init.contiguous()
            self.p_layer.weight.data = p_init.contiguous()

        self.lambda_layer.data = lambda_init.unsqueeze(0).contiguous()

        # Cleanup
        del u, s, vh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_mask(self, device: torch.device) -> torch.Tensor:
        """Get the current timestep mask, defaulting to all ones.

        Note: rank_dropout is intentionally NOT applied here. T-LoRA's
        timestep mask is a structured rank selection (coarse-to-fine);
        random rank dropout would undermine this and can zero out all
        active ranks at high-noise timesteps.
        """
        mask = get_timestep_mask(self.mask_group_id)
        if mask is None:
            mask = torch.ones(1, self.lora_dim)
        # Ensure mask covers our rank (in case max_rank > lora_dim)
        if mask.shape[1] > self.lora_dim:
            mask = mask[:, :self.lora_dim]
        elif mask.shape[1] < self.lora_dim:
            # Pad with ones if mask is smaller
            mask = F.pad(mask, (0, self.lora_dim - mask.shape[1]), value=1.0)
        return mask.to(device)

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        q_weight: torch.Tensor,
        p_weight: torch.Tensor,
        lambda_weight: torch.Tensor,
        alpha: torch.Tensor,
        base_q: Optional[torch.Tensor] = None,
        base_p: Optional[torch.Tensor] = None,
        base_lambda: Optional[torch.Tensor] = None,
    ):
        """Reconstruct module from saved state dict."""
        lora_dim = q_weight.shape[0] if q_weight.dim() == 2 else q_weight.shape[0]
        module = cls(
            lora_name,
            orig_module,
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=float(alpha),
            use_data_init=True,
        )
        module.q_layer.weight.data.copy_(q_weight)
        module.p_layer.weight.data.copy_(p_weight)
        module.lambda_layer.data.copy_(lambda_weight)

        if base_q is not None:
            module.base_q.copy_(base_q)
        if base_p is not None:
            module.base_p.copy_(base_p)
        if base_lambda is not None:
            module.base_lambda.copy_(base_lambda)
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

    def custom_state_dict(self):
        """Return state dict for saving.

        Scalar is baked into both lambda_layer and base_lambda so that
        s*(P@diag(λm)@Q - P_base@diag(λ_base*m)@Q_base) is preserved
        when scalar resets to 1.0 on load.
        """
        scalar = self.scalar.to(device=self.lambda_layer.device, non_blocking=True)
        return {
            "q_layer.weight": self.q_layer.weight,
            "p_layer.weight": self.p_layer.weight,
            "lambda_layer": self.lambda_layer * scalar,
            "alpha": self.alpha,
            "base_q": self.base_q,
            "base_p": self.base_p,
            "base_lambda": self.base_lambda * scalar,
        }

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        """
        Compute the weight difference: current - base.

        For T-LoRA: ΔW = P @ diag(λ*mask) @ Q - P_base @ diag(λ_base*mask) @ Q_base
        """
        if device is None:
            device = self.q_layer.weight.device

        mask = self._get_mask(device)

        # Current weights
        q = self.q_layer.weight.to(device)  # (lora_dim, in_dim) or conv shape
        p = self.p_layer.weight.to(device)  # (out_dim, lora_dim) or conv shape
        lam = self.lambda_layer.to(device) * mask  # (1, lora_dim)

        # Base weights
        q_base = self.base_q.to(device)
        p_base = self.base_p.to(device)
        lam_base = self.base_lambda.to(device) * mask

        if self.isconv:
            # For conv, reshape to 2D for matmul
            # q: (lora_dim, in_ch, 1, ...) -> (lora_dim, in_ch)
            # p: (out_ch, lora_dim, 1, ...) -> (out_ch, lora_dim)
            q_2d = q.reshape(self.lora_dim, -1)
            p_2d = p.reshape(self.shape[0], self.lora_dim)
            q_base_2d = q_base.reshape(self.lora_dim, -1)
            p_base_2d = p_base.reshape(self.shape[0], self.lora_dim)

            # Current: P @ diag(λ) @ Q
            # diag(λ) @ Q = λ.T * Q (broadcasting)
            curr = p_2d @ (lam.T * q_2d)  # (out_ch, in_ch)
            base = p_base_2d @ (lam_base.T * q_base_2d)

            # Reshape back to conv shape with 1x1 kernel
            kernel_ones = [1] * (len(self.shape) - 2)
            diff = (curr - base).reshape(self.shape[0], self.shape[1], *kernel_ones)
        else:
            # For linear: P @ diag(λ) @ Q
            # p: (out_features, lora_dim), λ: (1, lora_dim), q: (lora_dim, in_features)
            curr = p @ (lam.T * q)  # (out_features, in_features)
            base = p_base @ (lam_base.T * q_base)
            diff = curr - base

        diff = diff * self.scalar.to(device) * self.scale * multiplier

        if shape is not None:
            diff = diff.view(shape)

        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        """Get original weight + LoRA delta."""
        diff, _ = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)
        weight = self.get_org_weight_for_compute(diff.device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)

        # For conv with non-1x1 kernel, we need to handle shape mismatch
        if self.isconv and diff.shape != weight.shape:
            # diff is 1x1, weight has kernel - can't directly add
            # Place the 1x1 diff at the padding offset in the kernel so that
            # conv(x, merged, stride=s, padding=p) = conv(x, orig, stride=s, padding=p) + conv(x, diff_1x1, stride=s, padding=0)
            kernel_size = weight.shape[2:]
            if all(k == 1 for k in kernel_size):
                merged = weight + diff
            else:
                diff_expanded = torch.zeros_like(weight)
                padding = self.kw_dict.get("padding", tuple(k // 2 for k in kernel_size))
                center_slices = (slice(None), slice(None)) + tuple(
                    slice(p, p + 1) for p in padding
                )
                diff_expanded[center_slices] = diff
                merged = weight + diff_expanded
        else:
            merged = weight + diff

        return merged, None

    def orthogonality_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthogonality regularization loss.

        Encourages P and Q to remain orthogonal:
        L_p = ||P^T @ P - I||_F^2
        L_q = ||Q @ Q^T - I||_F^2

        Returns:
            (p_reg, q_reg): Regularization losses for P and Q
        """
        device = self.q_layer.weight.device

        if self.isconv:
            q = self.q_layer.weight.reshape(self.lora_dim, -1)
            p = self.p_layer.weight.reshape(self.shape[0], self.lora_dim)
        else:
            q = self.q_layer.weight
            p = self.p_layer.weight

        # P^T @ P should be identity (lora_dim x lora_dim)
        p_reg = torch.sum(
            (p.T @ p - torch.eye(self.lora_dim, device=device)) ** 2
        )

        # Q @ Q^T should be identity (lora_dim x lora_dim)
        q_reg = torch.sum(
            (q @ q.T - torch.eye(self.lora_dim, device=device)) ** 2
        )

        return p_reg, q_reg

    @staticmethod
    def _reshape_lam_for_broadcast(lam: torch.Tensor, q_out: torch.Tensor) -> torch.Tensor:
        """Reshape lam to broadcast with q_out.

        lam is (1, rank) or (B, rank).  q_out is (B, rank, *spatial) for conv
        or (B, [seq,] rank) for linear.  We need lam in the rank-dim position
        with singleton dims for everything else.

        For conv: q_out is (B, rank, H, W, ...) -> lam needs (B_or_1, rank, 1, 1, ...)
        For linear 2-D: q_out is (B, rank) -> lam is already fine as (B_or_1, rank)
        For linear 3-D: q_out is (B, seq, rank) -> lam needs (B_or_1, 1, rank)
        """
        if q_out.dim() == lam.dim():
            # Both 2-D: (B, rank) * (B_or_1, rank) — works directly
            return lam

        if q_out.dim() > lam.dim():
            # q_out has extra dims. Figure out where rank sits.
            # Conv: rank is dim 1, extra spatial dims follow -> pad trailing
            # Linear 3-D: rank is dim -1, seq is dim 1 -> pad middle
            rank_dim = lam.shape[-1]
            if q_out.shape[1] == rank_dim:
                # Conv layout: (B, rank, *spatial)
                return lam.view(*lam.shape, *([1] * (q_out.dim() - lam.dim())))
            else:
                # Linear layout: (B, seq, rank)
                # Insert singleton dims between batch and rank
                n_middle = q_out.dim() - lam.dim()
                shape = (lam.shape[0],) + (1,) * n_middle + (lam.shape[1],)
                return lam.view(*shape)

        return lam

    def bypass_forward_diff(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Compute LoRA contribution in bypass mode.

        Supports both (1, rank) and (B, rank) masks for per-sample timestep masking.
        """
        device = x.device
        dtype = x.dtype
        mask = self._get_mask(device)

        lam = self.lambda_layer.to(device) * mask       # (1, rank) or (B, rank)
        lam_base = self.base_lambda.to(device) * mask   # (1, rank) or (B, rank)

        if self.isconv:
            # Current path: x -> Q -> scale by λ -> P
            q_out = self.down_op(x, self.q_layer.weight.to(dtype), None, **self.kw_dict_down)
            q_out_scaled = q_out * self._reshape_lam_for_broadcast(lam, q_out)
            curr_out = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None, **self.kw_dict_up)

            # Base path
            q_base_out = self.down_op(x, self.base_q.to(dtype), None, **self.kw_dict_down)
            q_base_scaled = q_base_out * self._reshape_lam_for_broadcast(lam_base, q_base_out)
            base_out = self.up_op(q_base_scaled, self.base_p.to(dtype), None, **self.kw_dict_up)
        else:
            # Current path
            q_out = self.down_op(x, self.q_layer.weight.to(dtype), None)
            q_out_scaled = q_out * self._reshape_lam_for_broadcast(lam, q_out)
            curr_out = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None)

            # Base path
            q_base_out = self.down_op(x, self.base_q.to(dtype), None)
            q_base_scaled = q_base_out * self._reshape_lam_for_broadcast(lam_base, q_base_out)
            base_out = self.up_op(q_base_scaled, self.base_p.to(dtype), None)

        diff = curr_out - base_out
        return self.drop(diff * self.scalar.to(device) * self.scale * scale)

    def bypass_forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Forward with bypass mode (compute LoRA separately)."""
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def _has_batched_mask(self) -> bool:
        """Check if the current timestep mask is per-sample (B > 1)."""
        mask = get_timestep_mask(self.mask_group_id)
        return mask is not None and mask.shape[0] > 1

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with timestep-dependent rank masking."""
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        # For conv with non-1x1 kernel or groups > 1, always use bypass mode
        # since diff is computed with 1x1 groups=1 shape.
        # Also use bypass when a per-sample (batched) mask is active, because
        # get_diff_weight produces a single weight matrix that can't vary per sample.
        if self.bypass_mode or self._has_batched_mask() or (
            self.isconv
            and (any(k != 1 for k in self.shape[2:]) or self.kw_dict.get("groups", 1) > 1)
        ):
            return self.bypass_forward(x, scale=self.multiplier)

        # Standard forward: org_forward(x) + delta
        base = self.org_forward(x, *args, **kwargs)
        diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=x.device)
        diff_weight = diff_weight.to(self.dtype)

        delta = self.op(x, diff_weight, None, **self.kw_dict)
        return base + delta

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None):
        """Apply max norm regularization to weights."""
        diff, _ = self.get_diff_weight(multiplier=1.0, device=device)
        orig_norm = diff.norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio
