# abba.py
import math
from functools import cache
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger

from typing import Optional

@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA) will cause network dropout to be applied to the forward input, "
        "instead of to the layers, as per the DoRA paper."
    )

class AbbaModule(LycorisBaseModule):
    """
    Implementation of the ABBA (Hadamard Product Adaptation) module.
    This module reparameterizes the weight update as a Hadamard product of two
    independent low-rank matrices: ΔW = s * ((B1*A1) * (B2*A2)).
    """

    name = "abba"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    # Define the weights to be saved in the state dictionary
    weight_list = [
        "lora_up1.weight",
        "lora_down1.weight",
        "lora_up2.weight",
        "lora_down2.weight",
        "alpha",
        "dora_scale",
    ]
    # Use a deterministic set of weights to identify this module type from a state dict
    weight_list_det = ["lora_up1.weight", "lora_up2.weight"]

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
        weight_decompose=False,
        wd_on_output=True,
        bypass_mode=None,
        rs_lora=False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        """
        Initializes the ABBA module.
        If alpha is 0 or None, it is set to lora_dim.
        """
        # Pass only relevant kwargs to parent
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
            ggpo_conv_weight_sample_size
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in ABBA algo.")

        self.lora_dim = lora_dim
        # Split rank for the two adapter pairs, as suggested by the paper for fair comparison
        self.r1 = lora_dim // 2
        self.r2 = lora_dim - self.r1

        if self.r1 == 0 or self.r2 == 0:
            logger.warning(
                f"ABBA requires a rank of at least 2. "
                f"Got lora_dim={lora_dim}, which results in r1={self.r1}, r2={self.r2}. "
                f"The module will be inactive."
            )

        if self.module_type.startswith("conv"):
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.down_op = self.op
            self.up_op = self.op
            # Following LoCon, the 'down' layers have the full kernel, 'up' are 1x1 convs
            self.lora_down1 = self.module(in_dim, self.r1, k_size, stride, padding, bias=False)
            self.lora_up1 = self.module(self.r1, out_dim, 1, bias=False)
            self.lora_down2 = self.module(in_dim, self.r2, k_size, stride, padding, bias=False)
            self.lora_up2 = self.module(self.r2, out_dim, 1, bias=False)

        elif self.module_type == "linear":
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down1 = nn.Linear(in_dim, self.r1, bias=False)
            self.lora_up1 = nn.Linear(self.r1, out_dim, bias=False)
            self.lora_down2 = nn.Linear(in_dim, self.r2, bias=False)
            self.lora_up2 = nn.Linear(self.r2, out_dim, bias=False)
            # Cache for Khatri-Rao factors for efficient forward pass
            self.A_star = None
            self.B_star = None

        else:
            raise NotImplementedError
        
        self.wd = weight_decompose
        self.wd_on_output = wd_on_output
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_output:
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

        
        if dropout and self.wd:
            log_wd()

        # In ABBA, alpha is used to derive a single scaling factor.

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        # Scaling factor from ABBA Paper, Theorem 2
        # s_ABBA = alpha_LORA**2 / sqrt(r1 * r2)
        if self.r1 > 0 and self.r2 > 0:
            self.scale = alpha**2 / math.sqrt(self.r1 * self.r2)
        else:
            self.scale = 0.0
        
        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
        
        # Initialize weights according to the ABBA paper's strategy
        self.initialize_weights()
        
        self.init_ggpo()

    def initialize_weights(self):
        """
        Initialize weights using the hybrid SVD strategy from the ABBA paper.
        Pair 1 (A1, B1) is initialized from the SVD of the original weight.
        Pair 2 (A2, B2) is initialized with Kaiming uniform and zeros.
        """
        if self.r1 == 0 or self.r2 == 0:
            # If ranks are zero, initialize all to zero to be safe.
            torch.nn.init.constant_(self.lora_down1.weight, 0)
            torch.nn.init.constant_(self.lora_up1.weight, 0)
            torch.nn.init.constant_(self.lora_down2.weight, 0)
            torch.nn.init.constant_(self.lora_up2.weight, 0)
            return

        W0 = self.org_module[0].weight.data.clone().to(dtype=torch.float32)

        # Initialize the first pair (A1, B1) using SVD
        # U has shape (out_dim, r1), S has shape (r1,), V has shape (in_dim, r1)
        U, S, V = torch.svd_lowrank(W0.view(W0.size(0), -1), q=self.r1, niter=10)
        S_sqrt = torch.sqrt(S)

        B1_data = U * S_sqrt.unsqueeze(0)  # B1 is the 'up' weight
        A1_data = S_sqrt.unsqueeze(1) * V.T  # A1 is the 'down' weight

        # Reshape and assign to conv/linear layers
        self.lora_up1.weight.data.copy_(B1_data.view(self.lora_up1.weight.shape))
        self.lora_down1.weight.data.copy_(A1_data.view(self.lora_down1.weight.shape))

        # Initialize the second pair (A2, B2)
        # A2 with Kaiming uniform, B2 with zeros
        torch.nn.init.kaiming_uniform_(self.lora_down2.weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.lora_up2.weight, 0)

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, up1, down1, up2, down2, alpha, dora_scale):
        """
        Creates an AbbaModule from a state dictionary.
        """
        lora_dim = down1.size(0) + down2.size(0)
        module = cls(
            lora_name,
            orig_module,
            1.0,
            lora_dim=lora_dim,
            alpha=float(alpha),
            use_tucker=False,
            weight_decompose=dora_scale is not None,
        )
        # Manually copy weights instead of re-initializing
        module.lora_up1.weight.data.copy_(up1)
        module.lora_down1.weight.data.copy_(down1)
        module.lora_up2.weight.data.copy_(up2)
        module.lora_down2.weight.data.copy_(down2)

        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        """
        Handles loading state dicts that may not have a 'scalar' key.
        This is for compatibility with older saved models.
        """
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        
        # If scalar was baked into the weights, reset the module's scalar to 1.0
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.fill_(1.0)
        elif hasattr(self, "scalar"):
            self.scalar.fill_(1.0)
        else:
            # Fallback if scalar doesn't exist for some reason
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

    def custom_state_dict(self):
        """
        Returns a custom state dictionary for saving.
        The 'scalar' is baked into the lora_up weights for simplicity and symmetry.
        """
        destination = {}

        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha
        
        # Symmetrically bake the sqrt of the scalar into both up weights.
        # This is mathematically equivalent to multiplying the final output by the scalar.
        sqrt_scalar = torch.sqrt(self.scalar.to(device=self.lora_up1.weight.device, non_blocking=True))
        destination["lora_up1.weight"] = self.lora_up1.weight * sqrt_scalar
        destination["lora_down1.weight"] = self.lora_down1.weight
        destination["lora_up2.weight"] = self.lora_up2.weight * sqrt_scalar
        destination["lora_down2.weight"] = self.lora_down2.weight
        return destination

    @staticmethod
    def _khatri_rao(B1, B2):
        """Row-wise Khatri-Rao product for B matrices."""
        d_out, r1 = B1.shape
        _, r2 = B2.shape
        return (B1.unsqueeze(2) * B2.unsqueeze(1)).reshape(d_out, r1 * r2)

    @staticmethod
    def _khatri_rao_A(A1, A2):
        """Column-wise Khatri-Rao product for A matrices."""
        r1, d_in = A1.shape
        r2, _ = A2.shape
        return (A1.unsqueeze(1) * A2.unsqueeze(0)).reshape(r1 * r2, d_in)

    def _rebuild_khatri_rao_factors(self, device):
        """Recomputes and caches the Khatri-Rao factors for linear layers."""
        A1 = self.lora_down1.weight.to(device)
        B1 = self.lora_up1.weight.to(device)
        A2 = self.lora_down2.weight.to(device)
        B2 = self.lora_up2.weight.to(device)
        
        self.A_star = self._khatri_rao_A(A1, A2)
        self.B_star = self._khatri_rao(B1, B2)

    def make_weight(self, device=None):
        """
        Computes the weight update matrix ΔW.
        ΔW = (B1 @ A1) * (B2 @ A2)
        """
        if self.r1 == 0 or self.r2 == 0:
            return torch.zeros(self.shape, device=device)

        # Get weights for both pairs
        wa1 = self.lora_up1.weight.to(device)
        wb1 = self.lora_down1.weight.to(device)
        wa2 = self.lora_up2.weight.to(device)
        wb2 = self.lora_down2.weight.to(device)

        # For conv layers, we need to reshape the weights to perform matrix multiplication
        if self.isconv:
            wa1 = wa1.view(wa1.size(0), -1)
            wb1 = wb1.view(wb1.size(0), -1)
            wa2 = wa2.view(wa2.size(0), -1)
            wb2 = wb2.view(wb2.size(0), -1)

        # Calculate the two separate low-rank updates
        delta_w1 = wa1 @ wb1
        delta_w2 = wa2 @ wb2
        
        # Combine with Hadamard product
        delta_w = delta_w1 * delta_w2
        delta_w = delta_w.view(self.shape)

        if self.training and self.rank_dropout:
            drop = (torch.rand(delta_w.size(0), device=device) > self.rank_dropout).to(delta_w.dtype)
            drop = drop.view(-1, *[1] * len(delta_w.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            delta_w *= drop

        return delta_w * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        """
        Returns the scaled difference weight.
        This is used for merging the weights into the base model.
        """
        diff = self.make_weight(device=device) * self.scale * multiplier
        if shape is not None:
            diff = diff.view(shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        """
        Returns the full merged weight (original + scaled difference).
        """
        diff, _ = self.get_diff_weight(multiplier, shape, device)
        weight = self.get_org_weight_for_compute(device).to(self.dtype, non_blocking=True)
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    def bypass_forward_diff(self, x, scale=1):
        """
        Efficient forward pass for the difference weight (ΔW * x).
        Uses Khatri-Rao factorization for linear layers.
        Falls back to materializing the weight for convolutional layers.
        """
        if self.r1 == 0 or self.r2 == 0:
            return 0
        
        current_scale = self.scale * scale
        
        # Efficient path for Linear layers using Khatri-Rao factorization
        if not self.isconv:
            if self.A_star is None or self.B_star is None:
                self._rebuild_khatri_rao_factors(x.device)
            
            # Apply the two sequential linear operations
            mid = F.linear(x, self.A_star)
            up = F.linear(mid, self.B_star)
            
            return self.drop(up * self.scalar * current_scale)

        # Inefficient path for Convolutional layers
        else:
            if not hasattr(self, '_warned_conv'):
                logger.warning("ABBA for convolution layers is inefficient in bypass mode as it materializes the weight matrix.")
                self._warned_conv = True

            diff_weight = self.make_weight(device=x.device)
            
            bias = self.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(x.dtype, non_blocking=True)

            return self.drop(self.op(x, diff_weight * current_scale, bias, **self.kw_dict))

    def forward(self, x):
        """
        Standard forward pass. Handles module dropout and bypass/rebuild modes.
        """
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)

        if self.bypass_mode:
            return self.org_forward(x) + self.bypass_forward_diff(x, self.multiplier)
        else:
            if self.wd:
                # Rebuild mode: merge weights on-the-fly
                merged_weight, _ = self.get_merged_weight(self.multiplier, device=x.device)

                bias = self.get_org_bias_for_compute(x.device)
                if bias is not None:
                    bias = bias.to(x.dtype, non_blocking=True)
                    
                return self.op(self.drop(x), merged_weight, bias, **self.kw_dict)
            elif self.dropout:
                # 1. Get the original module's output
                # self.org_forward(x) correctly calls the original layer's forward pass.
                base_output = self.org_forward(x)

                # 2. Calculate the LoRA delta output
                diff_weight = self.make_weight(x.device).to(self.dtype)
                
                # The delta path has no bias.
                delta_output = self.op(x, diff_weight, None, **self.kw_dict)

                # 3. Apply scaling, multiplier, and network dropout to the delta output
                # This mirrors the logic in bypass_forward_diff
                final_delta = self.drop(delta_output * self.scale * self.multiplier)
                
                # 4. Return the sum
                return base_output + final_delta
            else:
                # Rebuild mode: merge weights on-the-fly
                merged_weight, _ = self.get_merged_weight(self.multiplier, device=x.device)

                bias = self.get_org_bias_for_compute(x.device)
                if bias is not None:
                    bias = bias.to(x.dtype, non_blocking=True)

                return self.op(x, merged_weight, bias, **self.kw_dict)

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
        unscaled_norm = self.make_weight(device).norm() * self.scale
        return unscaled_norm
    
    def apply_weight_decompose(self, weight, multiplier=1):
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_output:
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
    
    def ggpo_pertubation(self, x):
        # Optimized perturbation generation based on module type
        if self.module_type == "linear":
            # More efficient scale calculation
            perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (self.ggpo_beta * (self.grad_norms**2))
            perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)
            
            # For linear layers, use efficient matrix multiplication
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            perturbation = perturbation * perturbation_scale_factor.view(-1, 1)
            return x @ perturbation.T
        elif self.module_type.startswith("conv") and self.ggpo_conv:
            # More efficient scale calculation
            perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (self.ggpo_beta * (self.grad_norms**2))
            perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)

            # For convolution layers, generate efficient perturbation
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            
            # Apply scaling with efficient broadcasting
            view_shape = [perturbation.shape[0]] + [1] * (len(perturbation.shape) - 1)
            perturbation = perturbation * perturbation_scale_factor.view(*view_shape)
            
            # Use the appropriate convolution operation
            return self.op(x, perturbation, None, **self.kw_dict)
        else:
            return None