# OCGOptV2 from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
import math

from .utils import copy_stochastic_
import logging
from functools import lru_cache
from typing import Tuple

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
"""
GRAM_NEWTON_SCHULZ_2STEP_COEFFS1 = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]

@torch.no_grad()
def gram_newton_schulz_2step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    2-step Gram Newton-Schulz with pre-optimized unconstrained coefficients.

    Uses 2-step accumulated iteration with coefficients derived from pure
    optimization (no h(1)=1 or h'(1)=-0.5 constraints enforced).

    Coefficients optimized for spectral range [0.2, 1.8] with convergence
    target ||r_final - 1|| < 1e-4.

    Args:
        M: Input matrix [n, m] to orthogonalize
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality after odd iterations
        tolerance: Orthonormality error threshold for early stopping

    Returns:
        Orthonormal matrix [n, m]
    """
    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL-Gram folding
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s
    R = s * A * s.mT

    n, m = X.shape
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    # Q = I is safe here: Q = Q @ z rebinds Q to a new tensor (not in-place),
    # so I is never mutated.
    Q = I

    # 5-step iteration with pre-optimized coefficients
    for step_idx, (a, b, c) in enumerate(GRAM_NEWTON_SCHULZ_2STEP_COEFFS1):
        # Cubic polynomial on Gram matrix
        R2 = R @ R
        z = a * I + b * R + c * R2

        # Accumulated updates
        Q = Q @ z
        R = z @ R @ z

    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)

@torch.no_grad()
@lru_cache(maxsize=64)
def _get_filter_weights(shape: Tuple[int, ...], device_str: str, fft_alpha: float) -> torch.Tensor:
    """Cached FFT low-pass filter weights. Depends only on shape/device/alpha, not gradient values.
    Avoids recomputing freq coords, meshgrid, and radius every step for each parameter."""
    device = torch.device(device_str)
    freq_dims = [torch.fft.fftfreq(s, device=device) for s in shape]
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    return torch.exp(-fft_alpha * (radius ** 2))

def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')

    # 2. Apply cached radial low-pass filter
    filter_weights = _get_filter_weights(grad.shape, str(grad.device), fft_alpha)

    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights

    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')

    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

def create_gaussian_mask(shape, sigma=1.0, device='cpu'):
    """
    Creates a n-dimensional Gaussian mask, centered for use with fftshift.
    """
    freq_dims = [torch.fft.fftfreq(s, device=device) for s in shape]
    # Center the grid for radial calculation
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]

    # Create a meshgrid of coordinates
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))

    # Calculate the radial distance (L2 norm) from the center (zero frequency)
    # Normalize by the max possible frequency radius for scale invariance
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius

    # Create a Gaussian low-pass filter.
    # Higher alpha means sharper decay, i.e., more aggressive filtering
    filter_weights = torch.exp(-sigma * (radius ** 2))
    return filter_weights

def similarity_fft(grad, prev_grad, sigma=0.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    prev_grad_freq = torch.fft.fftn(prev_grad, norm='ortho')

    grad_freq_shifted = torch.fft.fftshift(grad_freq)
    prev_grad_freq_shifted = torch.fft.fftshift(prev_grad_freq)

    agreement_mask = grad_freq_shifted.abs() * prev_grad_freq_shifted.abs().conj()

    mask_max = torch.max(agreement_mask.abs())
    if mask_max > 1e-16:
        agreement_mask /= mask_max

    new_grad_fft = grad_freq_shifted * agreement_mask.real

    if sigma != 0:
        gaussian_mask = create_gaussian_mask(grad.shape, sigma=sigma, device=grad.device)
        new_grad_fft = new_grad_fft * gaussian_mask

    new_grad_fft = torch.fft.ifftshift(new_grad_fft)

    new_grad = torch.fft.ifftn(new_grad_fft, norm='ortho').real

    return new_grad


def _reshape_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D: [N, -1] for >2D, [1, -1] for 1D, identity for 2D."""
    if t.ndim > 2:
        return t.reshape(len(t), -1)
    if t.ndim < 2:
        return t.reshape(1, -1)
    return t


class OCGOptV2(Optimizer):
    r"""
    OCGOptV2: Orthogonal Centralized Gradient Optimization.

    Separates momentum states into full gradient and centralized gradient for smoother and faster descent. Featuring orthogonalization, RMS normalization, cautious stepping, and dual-normed adaptive update magnitudes.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the centralized momentum, full gradient momentum (used for centering), and the long-term squared running average (default: 0.95, 0.9999999, 0.9999999).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step - Visualization: https://www.desmos.com/calculator/ipgbjovebr - (default: 0.995).
        centralization (float):
            Subtract the full gradient momentum from the current gradient at this ratio (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase) (default: True).
        spectral_clip_dtype (torch.dtype in string format):
            Sets the dtype of spectral clipping calculation. Recommended to use torch.float32 (or leave at default of None) (default: None, which results in torch.float32).
        adaptive (bool):
            Scale the full step to the momentumized average gradient, always utilizes RMS normalization on the gradient if True, otherwise caps RMS at 1.0 (default: True).
        adaptive_min (float):
            Minimum multiplier for the adaptive scale (default: -1.0).
        adaptive_max (float):
            Maximum multiplier for the adaptive scale (default: 1.0).
        input_norm (bool):
            Normalizes with RMS on the input feature dimensions instead of utilizing gradient-wise RMS normalization (default: False).
        aol (bool):
            Use RMS variant of AOL (Almost-Orthogonal-Layer) preconditioning on the input gradient instead of regular RMS normalization. Computes Gram matrix and rescales rows by inverse sqrt of absolute row sums (default: False).
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        sim_match (bool):
            Filters the frequencies of the running average with the gradient of the current step's frequencies (default: False).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
        compile_step (bool):
            Compile the entire per-parameter step function with torch.compile(fullgraph=True, dynamic=False).
            When True, the full momentum+spectral_clip+update pipeline is fused into a single
            compiled graph, subsuming spectral_clip_compile.  Requires PyTorch 2.x with dynamo support.
            Mutually exclusive with ``foreach`` (compile_step takes priority).
            (default: False).
        foreach (bool):
            Use ``torch._foreach_*`` operations to batch element-wise tensor
            operations (momentum update, weight decay, parameter update) across
            all eligible parameters per group, reducing GPU kernel-launch
            overhead.  Per-tensor operations (FFT low-pass filter, RMS
            normalization / AOL preconditioning, spectral clipping, atan2
            normalization, cautious stepping, adaptive scaling) remain
            sequential.  Scalar (0-dim) parameters are processed per-tensor
            via an inline native-like path.  Mutually exclusive with
            ``compile_step`` (compile_step takes priority).
            (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: float = (0.95, 0.9975, 0.9999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        centralization: float = 1.0,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        adaptive: bool = False,
        adaptive_min: float = -1.,
        adaptive_max: float = 1.,
        input_norm: bool = False,
        aol: bool = False,
        lowpass_grad: float = 0.0,
        cautious_min: float = 0.0,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        foreach: bool = False,
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        self._init_lr = lr
        self._compile_step = compile_step
        self._foreach = foreach
        self._scalar_tensors = {}
        self._srng_buf = None  # Reusable stochastic-rounding scratch buffer (lazy init)

        if compile_step:
            # Spectral clipping will be inlined into the compiled full-step graph;
            # no need for a separately compiled spectral_clip wrapper.
            self.clip_func = None
        elif spectral_clip_compile:
            self.clip_func = torch.compile(gram_newton_schulz_2step, dynamic=True, mode="default")
        else:
            self.clip_func = gram_newton_schulz_2step

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.bfloat16

        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split('.')[-1] # Gets "float16"
            spectral_clip_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            centralization = centralization,
            spectral_adaptive = spectral_adaptive,
            spectral_clip_compile = spectral_clip_compile,
            spectral_clip_dtype = spectral_clip_dtype,
            adaptive = adaptive,
            adaptive_min = adaptive_min,
            adaptive_max = adaptive_max,
            input_norm = input_norm,
            aol = aol,
            lowpass_grad = lowpass_grad,
            cautious_min = cautious_min,
            stochastic_fp = stochastic_fp,
            foreach = foreach,
        )

        super(OCGOptV2, self).__init__(params, defaults)

        # Set up the step function (compiled or uncompiled)
        self._compiled_step = self._ocgoptv2_step_fp32
        if compile_step:
            self._compile_core_fns()

    # ------------------------------------------------------------------
    # Compile helpers
    # ------------------------------------------------------------------

    def _compile_core_fns(self) -> None:
        r"""Lazily compile the full per-parameter step with torch.compile."""
        try:
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64
            )
            with torch._dynamo.utils.disable_cache_limit():
                self._compiled_step = torch.compile(
                    self._ocgoptv2_step_fp32, fullgraph=True, dynamic=False
                )
            logging.info(
                "OCGOptV2 full step compiled with torch.compile(fullgraph=True, dynamic=False)."
            )
        except Exception as e:
            logging.warning(
                f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. "
                f"Falling back to uncompiled step."
            )
            self._compiled_step = self._ocgoptv2_step_fp32

    def _get_scalar_tensors(self, device: torch.device) -> dict:
        r"""Return pre-allocated FP32 scalar tensors for *device*, creating them on first access.

        The caller should ``.fill_(value)`` each tensor before passing to the compiled step
        so that torch.compile does not re-specialize on changing Python float values.
        """
        if device not in self._scalar_tensors:
            self._scalar_tensors[device] = {
                'lr_t':           torch.tensor(0.0, dtype=torch.float32, device=device),
                'slow_beta1_t':   torch.tensor(0.0, dtype=torch.float32, device=device),
                'slow_beta2_t':   torch.tensor(0.0, dtype=torch.float32, device=device),
                'slow_beta3_t':   torch.tensor(0.0, dtype=torch.float32, device=device),
                'wd_mul_t':       torch.tensor(0.0, dtype=torch.float32, device=device),
                'centralization_t': torch.tensor(0.0, dtype=torch.float32, device=device),
                'step_clamp_t':   torch.tensor(0.0, dtype=torch.float32, device=device),
                'adaptive_min_t': torch.tensor(0.0, dtype=torch.float32, device=device),
                'adaptive_max_t': torch.tensor(0.0, dtype=torch.float32, device=device),
            }
        return self._scalar_tensors[device]

    def _get_srng_buf(self, like_tensor: torch.Tensor) -> torch.Tensor:
        r"""Get a reusable int32 scratch buffer for stochastic rounding noise.

        Returns a view of a single shared buffer sized to the largest parameter
        encountered. The buffer is reused across all parameters within a step,
        eliminating per-parameter int32 allocations. Content is NOT preserved
        across calls — callers must refill before each use.
        """
        n = like_tensor.numel()
        if (
            self._srng_buf is None
            or self._srng_buf.device != like_tensor.device
            or self._srng_buf.numel() < n
        ):
            self._srng_buf = torch.empty(n, dtype=torch.int32, device=like_tensor.device)
        return self._srng_buf[:n].view(like_tensor.shape)

    # ------------------------------------------------------------------
    # Compiled core step (static — no self / dict access)
    # ------------------------------------------------------------------

    @staticmethod
    def _ocgoptv2_step_fp32(
        # --- Tensor inputs (all FP32, modified in-place where noted) ---
        p_data:                torch.Tensor,  # [in-out] parameter clone in working precision
        grad:                  torch.Tensor,  # gradient in working precision (read-only)
        value_momentum:        torch.Tensor,  # [in-out] value momentum state clone
        centralized_momentum:  torch.Tensor,  # [in-out] centralized momentum state clone
        denom:                 torch.Tensor,  # [in-out] denom state clone (dummy for non-scalar)
        filter_weights:        torch.Tensor,  # pre-computed FFT low-pass weights (dummy if unused)
        # --- Scalar tensors (avoid Python-float specialization) ---
        lr_t:                  torch.Tensor,  # scalar float32
        slow_beta1_t:          torch.Tensor,  # scalar float32, averaged beta1
        slow_beta2_t:          torch.Tensor,  # scalar float32, averaged beta2
        slow_beta3_t:          torch.Tensor,  # scalar float32, averaged beta3
        wd_mul_t:              torch.Tensor,  # scalar float32, pre-computed wd * wd_rate**step
        centralization_t:      torch.Tensor,  # scalar float32
        step_clamp_t:          torch.Tensor,  # scalar float32, step count for ADOPT-style clamping
        adaptive_min_t:        torch.Tensor,  # scalar float32
        adaptive_max_t:        torch.Tensor,  # scalar float32
        # --- Compile-time constants (Python scalars → resolved at trace time) ---
        do_lowpass:            bool,          # True when dimcount > 0 AND lowpass_grad != 0
        do_aol:                bool,          # True when aol AND dimcount >= 1
        do_input_norm:         bool,          # True when input_norm AND dimcount >= 1
        do_adaptive:           bool,          # True when adaptive is enabled
        do_weight_decay:       bool,          # True when weight_decay != 0
        spectral_adaptive:     bool,          # adaptive spectral clipping flag
        is_scalar:             bool,          # True when dimcount < 1
        is_1d:                 bool,          # True when dimcount == 1
        needs_reshape:         bool,          # True when dimcount > 2 (conv / 3-D+ tensors)
        ortho_dtype,                        # torch.dtype for spectral clip computation
    ) -> None:
        r"""Core per-parameter step in working precision.

        All tensor inputs live on the compute device.  Boolean / int / dtype
        arguments are compile-time constants resolved during tracing so each
        distinct combination produces a separate compiled graph (bounded by
        the number of parameter-group configurations).

        This function is designed to be called from ``torch.compile`` with
        ``fullgraph=True, dynamic=False``.  It calls ``gram_newton_schulz_2step``
        directly (the uncompiled pure-tensor-math version) which the compiler
        traces through and inlines into a single fused graph.
        """

        # ---- 1. ADOPT-style clamping for early stability ----------------
        grad = grad.clamp(-step_clamp_t, step_clamp_t)

        # ---- 2. Low-pass filter via FFT --------------------------------
        if do_lowpass:
            grad_sign = grad.sign()
            grad_freq = torch.fft.fftn(grad, norm='ortho')
            filtered = torch.fft.ifftn(grad_freq * filter_weights, norm='ortho').real
            grad = filtered.abs().mul_(grad_sign)

        # ---- 3. RMS normalization / AOL preconditioning -----------------
        if do_aol:
            # AOL-RMS: Gram matrix row rescaling
            if is_1d:
                grad_2d = grad.reshape(1, -1)
            elif needs_reshape:
                grad_2d = grad.reshape(grad.shape[0], -1)
            else:
                grad_2d = grad
            A = grad_2d @ grad_2d.mT
            rescaling = A.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
            grad_2d = grad_2d * rescaling.rsqrt()
            grad = grad_2d.reshape_as(grad)

        if do_input_norm:
            # Feature-wise RMS normalization
            if is_1d:
                grad_2d = grad.reshape(1, -1)
            elif needs_reshape:
                grad_2d = grad.reshape(grad.shape[0], -1)
            else:
                grad_2d = grad
            rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
            grad = grad_2d.div(rms).reshape_as(grad)
        else:
            # Global RMS normalization (also used for scalar tensors)
            rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
            grad = grad.div(rms)

        # ---- 4. Pre-compute denom for scalar tensors --------------------
        if is_scalar:
            current_denom = denom.sqrt()

        # ---- 5. Centralized gradient (uses old value_momentum) ----------
        centralized_grad = grad - value_momentum * centralization_t

        # ---- 6. Pre-compute 1 - slow_beta for lerp weights -------------
        one_minus_sb1 = 1.0 - slow_beta1_t
        one_minus_sb2 = 1.0 - slow_beta2_t
        one_minus_sb3 = 1.0 - slow_beta3_t

        # ---- 7. Momentum updates (in-place to propagate to caller) ------
        centralized_momentum.lerp_(centralized_grad, weight=one_minus_sb1)
        value_momentum.lerp_(grad, weight=one_minus_sb2)

        # ---- 8. Exponential average (combines centralized + full) -------
        exp_avg = centralized_grad.lerp(centralized_momentum, weight=slow_beta1_t).add_(
            grad.lerp(value_momentum, weight=slow_beta2_t) * centralization_t
        )

        # ---- 9. Update denom for scalar tensors -------------------------
        if is_scalar:
            denom.lerp_(centralized_grad.pow(2), weight=one_minus_sb3)

        # ---- 10. Spectral clipping or atan2 normalization ---------------
        if not is_scalar:
            if is_1d:
                exp_avg_2d = exp_avg.reshape(1, -1)
            elif needs_reshape:
                exp_avg_2d = exp_avg.reshape(exp_avg.shape[0], -1)
            else:
                exp_avg_2d = exp_avg

            flip = exp_avg_2d.shape[0] > exp_avg_2d.shape[1]
            if flip:
                exp_avg_2d = exp_avg_2d.T

            # Call uncompiled gram_newton_schulz_2step (inlined by compiler)
            exp_avg_2d_o = gram_newton_schulz_2step(exp_avg_2d, ortho_dtype=ortho_dtype)

            if spectral_adaptive:
                scale_factor = (exp_avg_2d_o * exp_avg_2d).sum()
                exp_avg_2d_o = exp_avg_2d_o * scale_factor

            if flip:
                exp_avg_2d_o = exp_avg_2d_o.T

            full_step = exp_avg_2d_o.reshape_as(exp_avg)
            full_step = full_step.div(full_step.pow(2).mean().sqrt_().clamp_min_(1))
        else:
            full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

        # ---- 11. Cautious update: mask where gradient disagrees ---------
        mask = (grad * full_step > 0).to(full_step.dtype)
        num_agree = mask.sum()
        dim = full_step.numel()
        alpha = dim / (num_agree + 1.0)
        full_step = full_step * mask * alpha

        # ---- 12. Adaptive scaling ---------------------------------------
        if do_adaptive:
            scale_factor = (
                exp_avg.pow(2).mean().sqrt_() * full_step.pow(2).mean().sqrt_()
            ).mean().clamp(adaptive_min_t, adaptive_max_t)
            full_step = scale_factor * full_step

        # ---- 13. Decoupled weight decay ---------------------------------
        if do_weight_decay:
            full_step = full_step + p_data * wd_mul_t

        # ---- 14. Parameter update ---------------------------------------
        p_data.sub_(full_step * lr_t)

    # ------------------------------------------------------------------
    # Compiled step path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_compiled(self) -> None:
        r"""Full-step compiled path.

        For each parameter: prepares FP32 working copies, updates scalar
        tensors, calls the (potentially compiled) ``_ocgoptv2_step_fp32``,
        then copies results back with optional stochastic rounding.
        """
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"][0], group["betas"][1], group["betas"][2]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]
            stochastic_fp = group["stochastic_fp"]
            lowpass_grad = group["lowpass_grad"]
            spectral_adaptive = group["spectral_adaptive"]
            spectral_clip_dtype = group["spectral_clip_dtype"]
            adaptive = group["adaptive"]
            adaptive_min = group["adaptive_min"]
            adaptive_max = group["adaptive_max"]
            input_norm = group["input_norm"]
            aol = group["aol"]

            step = group['step']

            # Pre-compute slow betas (constant for all params in this group)
            b1p = beta1 ** step
            slow_beta1 = (b1p - beta1) / (b1p - 1.0)
            b2p = beta2 ** step
            slow_beta2 = (b2p - beta2) / (b2p - 1.0)
            b3p = beta3 ** step
            slow_beta3 = (b3p - beta3) / (b3p - 1.0)

            # Pre-compute weight decay multiplier
            wd_mul = weight_decay * weight_decay_rate ** step if weight_decay != 0 else 0.0

            # Resolve ortho_dtype: None → float32 for the compiled path
            ortho_dtype = spectral_clip_dtype if spectral_clip_dtype is not None else torch.float32

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data
                dimcount = grad.ndim
                is_scalar = dimcount < 1

                # ---- State initialization --------------------------------
                if len(state) == 0:
                    if is_scalar:
                        state["denom"] = torch.ones_like(grad)
                    state["value_momentum"] = torch.zeros_like(grad)
                    state["centralized_momentum"] = torch.zeros_like(grad)

                # ---- Prepare FP32 working copies -------------------------
                use_fp32 = p.dtype in {torch.bfloat16} and stochastic_fp
                if use_fp32:
                    grad_work = grad.to(torch.float32)
                    p_work = p.detach().clone().to(torch.float32)
                    value_momentum_work = state["value_momentum"].detach().to(torch.float32)
                    centralized_momentum_work = state["centralized_momentum"].detach().to(torch.float32)
                    denom_work = (
                        state["denom"].detach().to(torch.float32)
                        if is_scalar
                        else torch.empty(0, dtype=torch.float32, device=p.device)
                    )
                else:
                    grad_work = grad
                    p_work = p.detach().clone()
                    value_momentum_work = state["value_momentum"].detach().clone()
                    centralized_momentum_work = state["centralized_momentum"].detach().clone()
                    denom_work = (
                        state["denom"].detach().clone()
                        if is_scalar
                        else torch.empty(0, dtype=p.dtype, device=p.device)
                    )

                # ---- Pre-compute FFT filter weights (cached) -------------
                do_lowpass = dimcount > 0 and lowpass_grad != 0.0
                if do_lowpass:
                    filter_weights = _get_filter_weights(grad.shape, str(grad.device), lowpass_grad)
                else:
                    filter_weights = torch.empty(0, dtype=grad_work.dtype, device=p.device)

                # ---- Update scalar tensors (avoids recompilation) ---------
                st = self._get_scalar_tensors(p.device)
                st['lr_t'].fill_(lr)
                st['slow_beta1_t'].fill_(slow_beta1)
                st['slow_beta2_t'].fill_(slow_beta2)
                st['slow_beta3_t'].fill_(slow_beta3)
                st['wd_mul_t'].fill_(wd_mul)
                st['centralization_t'].fill_(centralization)
                st['step_clamp_t'].fill_(step)
                st['adaptive_min_t'].fill_(adaptive_min)
                st['adaptive_max_t'].fill_(adaptive_max)

                # ---- Call compiled / uncompiled core step -----------------
                self._compiled_step(
                    p_work, grad_work, value_momentum_work, centralized_momentum_work,
                    denom_work, filter_weights,
                    # Scalar tensors
                    st['lr_t'], st['slow_beta1_t'], st['slow_beta2_t'], st['slow_beta3_t'],
                    st['wd_mul_t'], st['centralization_t'], st['step_clamp_t'],
                    st['adaptive_min_t'], st['adaptive_max_t'],
                    # Compile-time constants
                    do_lowpass,
                    aol and not is_scalar,       # do_aol
                    input_norm and not is_scalar, # do_input_norm
                    adaptive,                     # do_adaptive
                    weight_decay != 0,            # do_weight_decay
                    spectral_adaptive,
                    is_scalar,
                    dimcount == 1,                # is_1d
                    dimcount > 2,                 # needs_reshape
                    ortho_dtype,
                )

                # ---- Copy back with optional stochastic rounding ---------
                if use_fp32:
                    if is_scalar:
                        copy_stochastic_(state["denom"], denom_work)
                    copy_stochastic_(state["value_momentum"], value_momentum_work)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum_work)
                    copy_stochastic_(p, p_work)
                else:
                    if is_scalar:
                        state["denom"].copy_(denom_work)
                    state["value_momentum"].copy_(value_momentum_work)
                    state["centralized_momentum"].copy_(centralized_momentum_work)
                    p.copy_(p_work)

    # ------------------------------------------------------------------
    # Foreach step path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_foreach(self) -> None:
        r"""Foreach step path.

        Uses ``torch._foreach_*`` operations to batch element-wise tensor
        operations across all eligible parameters per group, reducing GPU
        kernel-launch overhead.  The following operations are batched:

        * **FFT filter helpers**: ``_foreach_sign``, ``_foreach_abs_``,
          ``_foreach_mul_`` for the sign-preservation and magnitude ops
          surrounding the per-tensor FFT roundtrip.
        * **Centralized gradient**: ``_foreach_sub``.
        * **Momentum updates**: ``_foreach_lerp_``.
        * **Exponential average**: ``_foreach_lerp``, ``_foreach_add``.
        * **Weight decay / param update**: ``_foreach_add_``.
        * **Non-stochastic write-back**: ``_foreach_copy_``.

        Per-tensor operations that *cannot* be foreach'd (FFT roundtrip,
        RMS normalization / AOL preconditioning, spectral clipping via
        Newton-Schulz, atan2 normalization, cautious stepping, adaptive
        scaling) remain sequential.

        Scalar (0-dim) parameters are processed per-tensor via an inline
        native-like path since they use atan2 normalization with a running
        denominator instead of spectral clipping.
        """
        _FOUR_OVER_PI = 1.27323954474

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # ---- Hoist group-level scalars --------------------------------
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"][0], group["betas"][1], group["betas"][2]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]
            stochastic_fp = group["stochastic_fp"]
            lowpass_grad = group["lowpass_grad"]
            spectral_adaptive = group["spectral_adaptive"]
            spectral_clip_dtype = group["spectral_clip_dtype"]
            adaptive = group["adaptive"]
            adaptive_min = group["adaptive_min"]
            adaptive_max = group["adaptive_max"]
            input_norm = group["input_norm"]
            aol = group["aol"]

            step = group['step']

            # Pre-compute slow betas (constant for all params in this group)
            b1p = beta1 ** step
            slow_beta1 = (b1p - beta1) / (b1p - 1.0)
            b2p = beta2 ** step
            slow_beta2 = (b2p - beta2) / (b2p - 1.0)
            b3p = beta3 ** step
            slow_beta3 = (b3p - beta3) / (b3p - 1.0)

            # Pre-compute weight decay multiplier
            wd_mul = weight_decay * weight_decay_rate ** step if weight_decay != 0 else 0.0

            # Resolve ortho_dtype
            ortho_dtype = spectral_clip_dtype if spectral_clip_dtype is not None else torch.float32

            # ---- Collect eligible params, separating scalar from non-scalar
            foreach_params: list = []
            scalar_params: list = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    grad = p.grad.data
                    dimcount = grad.ndim
                    if dimcount < 1:
                        state["denom"] = torch.ones_like(grad)
                    state["value_momentum"] = torch.zeros_like(grad)
                    state["centralized_momentum"] = torch.zeros_like(grad)
                if p.grad.data.ndim < 1:
                    scalar_params.append(p)
                else:
                    foreach_params.append(p)

            # ==== Process scalar params per-tensor (native-like) ==========
            for p in scalar_params:
                state = self.state[p]
                grad = p.grad.data

                use_fp32 = p.dtype in {torch.bfloat16} and stochastic_fp
                if use_fp32:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.detach().to(torch.float32)
                    denom = state["denom"].detach().to(torch.float32)
                    value_momentum = state["value_momentum"].detach().to(torch.float32)
                    centralized_momentum = state["centralized_momentum"].detach().to(torch.float32)
                else:
                    p_fp32 = p.detach().clone()
                    denom = state["denom"].detach().clone()
                    value_momentum = state["value_momentum"].detach().clone()
                    centralized_momentum = state["centralized_momentum"].detach().clone()

                # ADOPT-style clamping
                grad = grad.clamp(-step, step)

                # Global RMS normalization (scalar path — no AOL / input_norm)
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                grad = grad.div(rms)

                # Denom for scalar
                current_denom = denom.sqrt()

                # Centralized gradient
                centralized_grad = grad.sub(value_momentum, alpha=centralization)

                # Momentum updates
                centralized_momentum = centralized_momentum.lerp(centralized_grad, weight=1. - slow_beta1)
                value_momentum = value_momentum.lerp(grad, weight=1. - slow_beta2)

                # Exponential average
                exp_avg = centralized_grad.lerp(centralized_momentum, weight=slow_beta1).add_(
                    grad.lerp(value_momentum, weight=slow_beta2), alpha=centralization
                )

                # Denom update
                denom = denom.lerp(centralized_grad.pow(2), weight=1. - slow_beta3)

                # atan2 normalization (scalar path)
                full_step = exp_avg.atan2(current_denom).mul_(_FOUR_OVER_PI)

                # Cautious update
                mask = (grad * full_step > 0).to(full_step.dtype)
                num_agree = mask.sum()
                dim = full_step.numel()
                alpha = dim / (num_agree + 1.)
                full_step.mul_(mask).mul_(alpha)

                # Adaptive scaling
                if adaptive:
                    scale_factor = (
                        exp_avg.pow(2).mean().sqrt_() * full_step.pow(2).mean().sqrt_()
                    ).mean().clamp(adaptive_min, adaptive_max)
                    full_step = scale_factor * full_step

                # Weight decay
                if wd_mul != 0:
                    full_step.add_(p_fp32.data, alpha=wd_mul)

                # Parameter update
                p_fp32.data.add_(full_step, alpha=-lr)

                # Write-back
                if use_fp32:
                    copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["denom"].copy_(denom)
                    state["value_momentum"].copy_(value_momentum)
                    state["centralized_momentum"].copy_(centralized_momentum)
                    p.copy_(p_fp32)

            # ==== Process non-scalar params via foreach ====================
            if not foreach_params:
                continue

            # Determine compute device (GPU preferred for foreach throughput)
            first_device = foreach_params[0].device
            if first_device.type == "cpu" and torch.cuda.is_available():
                compute_device = torch.cuda.current_device()
            else:
                compute_device = first_device

            n = len(foreach_params)

            # ==== Collect phase: build FP32 tensor lists on compute device
            p_fp32_list = [None] * n
            grad_list = [None] * n
            value_momentum_list = [None] * n
            centralized_momentum_list = [None] * n
            filter_weights_list = [None] * n
            use_fp32_list = [False] * n
            state_list = [None] * n
            param_list = [None] * n

            for idx, p in enumerate(foreach_params):
                state = self.state[p]
                state_list[idx] = state
                param_list[idx] = p

                grad = p.grad.data

                use_fp32 = p.dtype in {torch.bfloat16} and stochastic_fp
                use_fp32_list[idx] = use_fp32

                target_dtype = torch.float32 if use_fp32 else p.dtype

                grad_list[idx] = grad.to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                p_fp32_list[idx] = p.detach().to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                value_momentum_list[idx] = state["value_momentum"].to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                centralized_momentum_list[idx] = state["centralized_momentum"].to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )

                # Pre-compute FFT filter weights (cached per shape/device/alpha)
                dimcount = grad.ndim
                do_lowpass = dimcount > 0 and lowpass_grad != 0.0
                if do_lowpass:
                    compute_device_str = (
                        f"cuda:{compute_device}"
                        if isinstance(compute_device, int)
                        else str(compute_device)
                    )
                    filter_weights_list[idx] = _get_filter_weights(
                        grad.shape, compute_device_str, lowpass_grad
                    )

            # ==== Per-tensor: ADOPT-style clamping ============================
            for idx in range(n):
                grad_list[idx] = grad_list[idx].clamp(-step, step)

            # ==== Per-tensor: FFT low-pass filter ============================
            # (with foreach sign/abs/mul batching around the per-tensor FFT)
            filter_indices = [
                idx for idx in range(n) if filter_weights_list[idx] is not None
            ]
            if filter_indices:
                grads_to_filter = [grad_list[idx] for idx in filter_indices]
                # BATCH: sign() across all filtered grads (saves N-1 kernel launches)
                grad_signs = torch._foreach_sign(grads_to_filter)
                # Per-tensor: FFT roundtrip (shape-dependent, cannot be foreach'd)
                filtered_list = [None] * len(filter_indices)
                for i, idx in enumerate(filter_indices):
                    grad_freq = torch.fft.fftn(grads_to_filter[i], norm='ortho')
                    filtered_list[i] = torch.fft.ifftn(
                        grad_freq * filter_weights_list[idx], norm='ortho'
                    ).real
                # BATCH: abs() and mul_() across all filtered grads
                torch._foreach_abs_(filtered_list)
                torch._foreach_mul_(filtered_list, grad_signs)
                for i, idx in enumerate(filter_indices):
                    grad_list[idx] = filtered_list[i]

            # ==== Per-tensor: RMS normalization / AOL preconditioning ========
            # (shape-dependent, involves matrix ops for AOL; cannot be foreach'd)
            for idx in range(n):
                grad = grad_list[idx]

                if aol:
                    grad_2d = _reshape_to_2d(grad)
                    # AOL-RMS: Compute Gram matrix and rescale rows
                    A = grad_2d @ grad_2d.mT
                    rescaling = A.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
                    grad_2d = grad_2d * rescaling.rsqrt()
                    grad = grad_2d.view_as(grad)

                if input_norm:
                    grad_2d = _reshape_to_2d(grad)
                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                grad_list[idx] = grad

            # ==== BATCH: Centralized gradient ================================
            # centralized_grad = grad - value_momentum * centralization
            centralized_grad_list = torch._foreach_sub(
                grad_list, value_momentum_list, alpha=centralization
            )

            # ==== BATCH: Momentum updates (in-place) =========================
            # centralized_momentum.lerp_(centralized_grad, weight=1-slow_beta1)
            torch._foreach_lerp_(
                centralized_momentum_list, centralized_grad_list,
                weight=1.0 - slow_beta1,
            )
            # value_momentum.lerp_(grad, weight=1-slow_beta2)
            torch._foreach_lerp_(
                value_momentum_list, grad_list,
                weight=1.0 - slow_beta2,
            )

            # ==== BATCH: Exponential average ================================
            # exp_avg = centralized_grad.lerp(centralized_momentum, weight=slow_beta1)
            #       + grad.lerp(value_momentum, weight=slow_beta2) * centralization
            term1_list = torch._foreach_lerp(
                centralized_grad_list, centralized_momentum_list,
                weight=slow_beta1,
            )
            term2_list = torch._foreach_lerp(
                grad_list, value_momentum_list,
                weight=slow_beta2,
            )
            exp_avg_list = torch._foreach_add(
                term1_list, term2_list, alpha=centralization
            )

            # ==== Per-tensor: Spectral clipping + RMS normalization ==========
            # Newton-Schulz iterations involve per-tensor matrix products and
            # cannot be foreach'd.  The final RMS normalization is also per-tensor.
            full_step_list = [None] * n
            for idx in range(n):
                exp_avg = exp_avg_list[idx]

                exp_avg_2d = _reshape_to_2d(exp_avg)
                flip = exp_avg_2d.shape[0] > exp_avg_2d.shape[1]
                if flip:
                    exp_avg_2d = exp_avg_2d.T

                exp_avg_2d_o = self.clip_func(exp_avg_2d, ortho_dtype=ortho_dtype)

                if spectral_adaptive:
                    scale_factor = (exp_avg_2d_o * exp_avg_2d).sum()
                    exp_avg_2d_o = exp_avg_2d_o * scale_factor

                if flip:
                    exp_avg_2d_o = exp_avg_2d_o.T

                full_step = exp_avg_2d_o.view_as(exp_avg)
                full_step = full_step.div(full_step.pow(2).mean().sqrt_().clamp_min_(1))

                full_step_list[idx] = full_step

            # ==== Per-tensor: Cautious update ================================
            # Mask where gradient disagrees with update direction.
            # Involves per-tensor sum/numel for the normalization scalar.
            for idx in range(n):
                full_step = full_step_list[idx]
                grad = grad_list[idx]

                mask = (grad * full_step > 0).to(full_step.dtype)
                num_agree = mask.sum()
                dim = full_step.numel()
                alpha = dim / (num_agree + 1.0)
                full_step_list[idx] = full_step * mask * alpha

            # ==== Per-tensor: Adaptive scaling ===============================
            if adaptive:
                for idx in range(n):
                    exp_avg = exp_avg_list[idx]
                    full_step = full_step_list[idx]
                    scale_factor = (
                        exp_avg.pow(2).mean().sqrt_() * full_step.pow(2).mean().sqrt_()
                    ).mean().clamp(adaptive_min, adaptive_max)
                    full_step_list[idx] = scale_factor * full_step

            # ==== BATCH: Decoupled weight decay ==============================
            if wd_mul != 0:
                torch._foreach_add_(full_step_list, p_fp32_list, alpha=wd_mul)

            # ==== BATCH: Parameter update ====================================
            torch._foreach_add_(p_fp32_list, full_step_list, alpha=-lr)

            # ==== Write-back phase ===========================================
            # Split into stochastic (per-tensor, custom rounding) and
            # non-stochastic (batched via foreach_copy_) paths to minimize
            # kernel-launch overhead for the common fp32 case.
            stoch_indices = [
                idx for idx in range(n) if use_fp32_list[idx]
            ]
            non_stoch_indices = [
                idx for idx in range(n) if not use_fp32_list[idx]
            ]

            # Stochastic rounding path (per-tensor, uses shared scratch buffer)
            for idx in stoch_indices:
                scratch = self._get_srng_buf(p_fp32_list[idx])
                copy_stochastic_(
                    state_list[idx]["value_momentum"], value_momentum_list[idx],
                    scratch=scratch,
                )
                copy_stochastic_(
                    state_list[idx]["centralized_momentum"],
                    centralized_momentum_list[idx],
                    scratch=scratch,
                )
                copy_stochastic_(param_list[idx], p_fp32_list[idx], scratch=scratch)

            # Non-stochastic path — batched via torch._foreach_copy_
            if non_stoch_indices:
                ns_vm_dst = [
                    state_list[idx]["value_momentum"] for idx in non_stoch_indices
                ]
                ns_vm_src = [value_momentum_list[idx] for idx in non_stoch_indices]
                torch._foreach_copy_(ns_vm_dst, ns_vm_src)
                ns_cm_dst = [
                    state_list[idx]["centralized_momentum"]
                    for idx in non_stoch_indices
                ]
                ns_cm_src = [
                    centralized_momentum_list[idx] for idx in non_stoch_indices
                ]
                torch._foreach_copy_(ns_cm_dst, ns_cm_src)
                ns_p_dst = [param_list[idx] for idx in non_stoch_indices]
                ns_p_src = [p_fp32_list[idx] for idx in non_stoch_indices]
                torch._foreach_copy_(ns_p_dst, ns_p_src)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Step — public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._compile_step:
            self._step_compiled()
        elif self._foreach:
            self._step_foreach()
        else:
            self._step_native()

        return loss

    # ------------------------------------------------------------------
    # Native (uncompiled) step path — original behaviour preserved
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_native(self) -> None:
        r"""Original per-parameter step without full-step compilation.

        Uses ``self.clip_func`` (optionally compiled spectral clip) for the
        spectral clipping sub-step.  This path is taken when
        ``compile_step=False``.
        """
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"][0], group["betas"][1], group["betas"][2]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]

            step = group['step']

            # Pre-compute weight decay multiplier once per group (avoids
            # recomputing weight_decay_rate**step for every parameter).
            _wd_mul = weight_decay * weight_decay_rate ** step if weight_decay != 0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                dimcount = grad.ndim

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    if dimcount < 1:
                        state["denom"] = torch.ones_like(grad)
                    state["value_momentum"] = torch.zeros_like(grad)
                    state["centralized_momentum"] = torch.zeros_like(grad)

                # Detach and unpack state into working copies.
                # When the parameter is bf16 with stochastic rounding, cast directly
                # to fp32 via .to() which already creates a new tensor on dtype change,
                # avoiding wasteful intermediate bf16 clones.
                if p.dtype in {torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.detach().to(torch.float32)
                    if dimcount < 1:
                        denom = state["denom"].detach().to(torch.float32)
                    value_momentum = state["value_momentum"].detach().to(torch.float32)
                    centralized_momentum = state["centralized_momentum"].detach().to(torch.float32)
                else:
                    p_fp32 = p.detach().clone()
                    if dimcount < 1:
                        denom = state["denom"].detach().clone()
                    value_momentum = state["value_momentum"].detach().clone()
                    centralized_momentum = state["centralized_momentum"].detach().clone()

                # Averaged beta (step 1 = 0, step 2 = 0.5, step 3 = 0.6667, step 4 = 0.75...)
                # Cache beta**step to avoid computing it twice per beta (6 → 3 pow calls).
                b1p = beta1 ** step
                slow_beta1 = (b1p - beta1) / (b1p - 1.0)
                b2p = beta2 ** step
                slow_beta2 = (b2p - beta2) / (b2p - 1.0)
                b3p = beta3 ** step
                slow_beta3 = (b3p - beta3) / (b3p - 1.0)

                # ADOPT-style clamping for early stability / to prevent NaNs
                grad = grad.clamp(-step, step)

                # Low-pass filter via FFT, maintains direction
                if dimcount > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                # Move RMS to 1.0, input-feature-wise if 2D or larger, otherwise utilize standard gradient-wide RMS normalization.
                # AOL preconditioning variant: uses Gram matrix row rescaling instead of RMS
                if dimcount >= 1 and group["aol"]:
                    grad_2d = _reshape_to_2d(grad)

                    # AOL-RMS: Compute Gram matrix and rescale rows by inverse sqrt of absolute row sums
                    A = grad_2d @ grad_2d.mT
                    rescaling = A.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
                    grad_2d = grad_2d * rescaling.rsqrt()

                    grad = grad_2d.view_as(grad)

                if dimcount >= 1 and group["input_norm"]:
                    grad_2d = _reshape_to_2d(grad)

                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16) # Cap at RMS of 1.0

                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16) # Cap at RMS of 1.0
                    grad = grad.div(rms)

                # ADOPT-style denominator update (un-updated denom)
                if dimcount < 1:
                    current_denom = denom.sqrt()

                # Centralize gradient by removing running average
                centralized_grad = grad.sub(value_momentum, alpha=centralization)

                # Momentumize the centralized gradient
                centralized_momentum = centralized_momentum.lerp(centralized_grad, weight=1. - slow_beta1)

                # Update full momentum
                value_momentum = value_momentum.lerp(grad, weight=1. - slow_beta2)

                # Add back full momentum to the centralized gradient
                exp_avg = centralized_grad.lerp(centralized_momentum, weight=slow_beta1).add_(grad.lerp(value_momentum, weight=slow_beta2), alpha=centralization)

                # Update denominator with either centralized gradient, or its mean when utilizing a sign-based gradient
                if dimcount < 1:
                    denom = denom.lerp(centralized_grad.pow(2), weight=1. - slow_beta3)

                # Spectral Clipping / Newton Schulz iters
                if dimcount >= 1:
                    exp_avg_2d = _reshape_to_2d(exp_avg)

                    flip = exp_avg_2d.shape[0] > exp_avg_2d.shape[1]
                    if flip:
                        exp_avg_2d = exp_avg_2d.T # Flip if first dim is larger

                    exp_avg_2d_o = self.clip_func(exp_avg_2d, ortho_dtype=group["spectral_clip_dtype"])

                    if group["spectral_adaptive"]:
                        scale_factor = (exp_avg_2d_o * exp_avg_2d).sum()
                        exp_avg_2d_o.mul_(scale_factor)

                    if flip:
                        exp_avg_2d_o = exp_avg_2d_o.T

                    full_step = exp_avg_2d_o.view_as(exp_avg)

                    full_step = full_step.div(full_step.pow(2).mean().sqrt_().clamp_min_(1))
                else:
                    full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

                # Cautious update: zero-out update where the update isn't in the direction of the current gradient
                mask = (grad * full_step > 0).to(full_step.dtype)
                num_agree = mask.sum()
                dim = full_step.numel()
                alpha = dim / (num_agree + 1.)
                full_step.mul_(mask).mul_(alpha)

                # Scale the full step with the gradient
                if group["adaptive"]:
                    scale_factor = (exp_avg.pow(2).mean().sqrt_() * full_step.pow(2).mean().sqrt_()).mean().clamp(group["adaptive_min"], group["adaptive_max"])
                    full_step = scale_factor * full_step

                # Perform weight decay (using pre-computed group-level multiplier)
                if _wd_mul != 0:
                    full_step.add_(p_fp32.data, alpha=_wd_mul)

                p_fp32.data.add_(full_step, alpha=-lr)

                # Stochastic update
                if p.dtype in {torch.bfloat16} and group["stochastic_fp"]:
                    if dimcount < 1:
                        copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    if dimcount < 1:
                        state["denom"].copy_(denom)
                    state["value_momentum"].copy_(value_momentum)
                    state["centralized_momentum"].copy_(centralized_momentum)
                    p.copy_(p_fp32)
