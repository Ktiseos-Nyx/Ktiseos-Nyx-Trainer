# OCGOpt from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

from .utils import copy_stochastic_

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
"""
"""
NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]
"""
# New coeffs from https://kexue.fm/archives/11059, may enable later.

NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]

@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None, adaptive=False) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = M.dtype
        M = M.to(ortho_dtype)
    if adaptive:
        M_orig = M.clone()
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T.contiguous()
    I = torch.eye(M.shape[1], dtype=M.dtype, device=M.device)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        M = M / (torch.linalg.norm(M).clamp_min_(1e-8))
        A = M.T @ M
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T.contiguous()
    if adaptive:
        M = M * (M_orig.type_as(M) * M).sum()
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M

@torch.no_grad()
def orthogonalize_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def orthogonalize_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

@lru_cache(maxsize=128)
def _compute_gaussian_weights(shape: tuple, strength: float, device_str: str) -> torch.Tensor:
    r"""Compute and cache Gaussian low-pass filter weights for a given shape and strength.

    The frequency grid and radial distance are constant for a given tensor shape
    and device, so caching eliminates redundant meshgrid/norm computations across
    steps when parameter shapes are static (the typical case during training).

    Args:
        shape: Tuple of tensor dimension sizes.
        strength: Gaussian decay rate (``fft_alpha`` for filter_grad, ``sigma`` for create_gaussian_mask).
        device_str: String representation of the torch device (e.g. ``'cuda:0'``).
    Returns:
        Gaussian filter weight tensor of the given shape on the given device.
    """
    device = torch.device(device_str)
    freq_dims = [torch.fft.fftfreq(s, device=device) for s in shape]
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    return torch.exp(-strength * (radius ** 2))

@torch.no_grad()
def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    
    # 2. Get cached radial low-pass filter weights
    filter_weights = _compute_gaussian_weights(tuple(grad.shape), fft_alpha, str(grad.device))
    
    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights
    
    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    
    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

def create_gaussian_mask(shape, sigma=1.0, device='cpu'):
    """
    Creates a n-dimensional Gaussian mask, centered for use with fftshift.
    Returns a cached result when called repeatedly with the same shape/sigma/device.
    """
    return _compute_gaussian_weights(tuple(shape), sigma, str(device))

@torch.no_grad()
def similarity_fft(grad, prev_grad, sigma=0.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    prev_grad_freq = torch.fft.fftn(prev_grad, norm='ortho')

    # Agreement mask: |X| * |Y*| is shift-invariant, no fftshift needed
    agreement_mask = grad_freq.abs() * prev_grad_freq.abs()

    mask_max = torch.max(agreement_mask.abs())
    if mask_max > 1e-16:
        agreement_mask /= mask_max
    
    new_grad_fft = grad_freq * agreement_mask.real

    if sigma != 0:
        gaussian_mask = create_gaussian_mask(grad.shape, sigma=sigma, device=grad.device)
        new_grad_fft = new_grad_fft * gaussian_mask

    new_grad = torch.fft.ifftn(new_grad_fft, norm='ortho').real

    return new_grad


class OCGOpt(Optimizer):
    r"""
    OCGOpt: Orthogonal Centralized Gradient Optimization.

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
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        sim_match (bool):
            Filters the frequencies of the running average with the gradient of the current step's frequencies (default: False).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        cautious_weight_decay (bool):
            Applies weight decay only to parameter coordinates whose signs align with the gradient direction, avoiding weight decay that would work against the optimization step. (default: False).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
        kahan_summation (bool):
            Utilize Kahan Summation for the parameter update. This maintains a high-precision error buffer to effectively double the precision of the accumulation step.
            Excellent for bfloat16 training, compatible with stochastic rounding. (default: False).
        foreach (bool):
            Use torch._foreach_* operations for parameters for better GPU utilization (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: float = (0.95, 0.9999999, 0.9999999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        centralization: float = 1.0,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        adaptive: bool = True,
        adaptive_min: float = -1.,
        adaptive_max: float = 1.,
        input_norm: bool = False,
        lowpass_grad: float = 0.0,
        sim_match: bool = False,
        cautious_min: float = 0.0,
        cautious_weight_decay: bool = False,
        stochastic_fp: bool = True,
        kahan_summation: bool = False,
        foreach: bool = False,
        sync_chunk_size: int = 256,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        compile_step: bool = False,
        num_ns_steps: int = len(NS_COEFFS),
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        if isinstance(state_storage_dtype, str):
            normalized_str_dtype = state_storage_dtype.strip().lower()
            if normalized_str_dtype == "float32":
                final_dtype = torch.float32
            elif normalized_str_dtype == "float16":
                final_dtype = torch.float16
            elif normalized_str_dtype == "bfloat16":
                final_dtype = torch.bfloat16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        # Single shared int32 scratch buffer for stochastic rounding.
        # Grows to the largest parameter size encountered; reused across all params each step.
        self._srng_buf: torch.Tensor | None = None

        self._init_lr = lr

        self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func

        # Compiled step callables (lazily compiled on first step() call)
        self._compiled_ge1d = None
        self._compiled_0d = None

        # Cache for placeholder tensors (device -> empty(0) tensor)
        self._empty_tensor_cache: dict = {}

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32

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
            lowpass_grad = lowpass_grad,
            sim_match = sim_match,
            cautious_min = cautious_min,
            cautious_weight_decay = cautious_weight_decay,
            stochastic_fp = stochastic_fp,
            kahan_summation = kahan_summation,
            foreach = foreach,
            sync_chunk_size = sync_chunk_size,
            state_storage_dtype = final_dtype,
            state_storage_device = state_storage_device,
            compile_step = compile_step,
            num_ns_steps = num_ns_steps,
        )

        super(OCGOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        r"""Reset all optimizer state buffers to their initial values.

        Clears momentum accumulators, 0-D denominators, and Kahan summation
        compensation tensors so that subsequent ``step()`` calls start fresh.
        """
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                state["value_momentum"].zero_()
                state["centralized_momentum"].zero_()
                if "denom" in state:
                    state["denom"].fill_(1.0)
                if "kahan_comp" in state:
                    state["kahan_comp"].zero_()

    def _get_srng_buf(self, like_tensor: torch.Tensor) -> torch.Tensor:
        r"""Get a reusable int32 scratch buffer for stochastic rounding noise.

        Returns a view of a single shared buffer sized to the largest parameter
        encountered. The buffer is reused across all parameters within a step,
        eliminating per-parameter int32 allocations. Content is NOT preserved
        across calls — callers must refill before each use.
        """
        n = like_tensor.numel()
        if self._srng_buf is None or self._srng_buf.device != like_tensor.device or self._srng_buf.numel() < n:
            self._srng_buf = torch.empty(n, dtype=torch.int32, device=like_tensor.device)
        return self._srng_buf[:n].view(like_tensor.shape)

    def _get_empty_tensor(self, device: torch.device) -> torch.Tensor:
        r"""Get or create cached empty tensor for optional state placeholder."""
        if device not in self._empty_tensor_cache:
            self._empty_tensor_cache[device] = torch.empty(0, device=device)
        return self._empty_tensor_cache[device]

    def _init_param_state(self, p, state, dimcount, use_kahan):
        r"""Initialize optimizer state tensors for a parameter.

        Creates ``value_momentum``, ``centralized_momentum``, and (for 0-D params)
        ``denom`` buffers on ``self.state_storage_device``.  Optionally creates a
        ``kahan_comp`` buffer when *use_kahan* is True and the parameter is
        low-precision (bf16/fp16).  Pinned memory is used when the storage device
        is CPU to enable efficient async device transfers.
        """
        is_cpu = self.state_storage_device == "cpu"

        if dimcount < 1:
            state["denom"] = torch.ones_like(
                p.data, dtype=self.state_storage_dtype,
                device=self.state_storage_device
            )
            if is_cpu:
                state["denom"] = state["denom"].pin_memory()

        state["value_momentum"] = torch.zeros_like(
            p.data, dtype=self.state_storage_dtype,
            device=self.state_storage_device
        )
        state["centralized_momentum"] = torch.zeros_like(
            p.data, dtype=self.state_storage_dtype,
            device=self.state_storage_device
        )
        if is_cpu:
            state["value_momentum"] = state["value_momentum"].pin_memory()
            state["centralized_momentum"] = state["centralized_momentum"].pin_memory()

        if use_kahan and p.dtype in {torch.bfloat16, torch.float16}:
            state["kahan_comp"] = torch.zeros_like(
                p.data, dtype=torch.float32,
                device=self.state_storage_device
            )
            if is_cpu:
                state["kahan_comp"] = state["kahan_comp"].pin_memory()

    def _late_init_kahan_comp(self, p, state):
        r"""Lazily create kahan_comp state for a parameter that was added after initialisation."""
        state["kahan_comp"] = torch.zeros_like(
            p.data, dtype=torch.float32,
            device=self.state_storage_device
        )
        if self.state_storage_device == "cpu":
            state["kahan_comp"] = state["kahan_comp"].pin_memory()

    @staticmethod
    def _simulate_kahan_rounding(p_fp32, p_dtype, stochastic_fp, scratch=None):
        r"""Simulate low-precision rounding for Kahan summation compensation.

        Returns ``(rounded_as_fp32, compensation)`` where
        ``compensation = rounded - exact``.

        For bfloat16 with stochastic rounding enabled, the rounding is simulated
        via mantissa-level stochastic noise.  For all other cases, deterministic
        round-to-nearest is used.

        Args:
            p_fp32: Exact FP32 parameter values.
            p_dtype: Target low-precision dtype.
            stochastic_fp: Whether to use stochastic rounding for bf16.
            scratch: Optional pre-allocated int32 scratch buffer for stochastic noise.
        """
        if stochastic_fp and p_dtype == torch.bfloat16:
            p_simulated = p_fp32.clone()
            sim_int = p_simulated.view(dtype=torch.int32)
            if scratch is not None:
                scratch_view = scratch[:p_simulated.numel()].view(p_simulated.shape)
                scratch_view.random_(0, 1 << 16)
                sim_int.add_(scratch_view)
            else:
                sim_int.add_(torch.randint_like(
                    sim_int, dtype=torch.int32, low=0, high=(1 << 16)
                ))
            sim_int.bitwise_and_(-65536)
        else:
            # Deterministic rounding (round to nearest)
            p_simulated = p_fp32.to(p_dtype).to(torch.float32)

        new_comp = p_simulated.sub(p_fp32)
        return p_simulated, new_comp

    @staticmethod
    @torch.no_grad()
    def _core_ge1d_step_fp32(
        grad: torch.Tensor,
        p_fp32: torch.Tensor,
        value_momentum: torch.Tensor,
        centralized_momentum: torch.Tensor,
        beta_t: float,
        slow_beta2_t: float,
        centralization_t: float,
        step_t: int,
        cautious_min_t: float,
        adaptive_min_t: float,
        adaptive_max_t: float,
        lr_t: float,
        weight_decay_t: float,
        wd_rate_t: float,
        input_norm: bool,
        adaptive: bool,
        spectral_adaptive: bool,
        ortho_dtype: torch.dtype,
        num_ns_steps: int,
        cautious_weight_decay: bool = False,
    ) -> None:
        r"""Core per-parameter step for dimcount >= 1 (with inlined Newton-Schulz clipping).

        All inputs are FP32 tensors on the compute device.
        Modifies all tensor arguments in-place.
        Branch booleans are compile-time constants resolved during tracing.
        """
        dimcount = grad.ndim

        # 1. Gradient clamping (ADOPT-style)
        grad.clamp_(-step_t, step_t)

        # 2. RMS normalization
        if input_norm:
            if dimcount > 2:
                grad_2d = grad.reshape(len(grad), -1)
            elif dimcount < 2:
                grad_2d = grad.reshape(1, -1)
            else:
                grad_2d = grad
            rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
            grad_2d.div_(rms)
        else:
            rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
            grad.div_(rms)

        # 3. Centralize gradient by removing running average
        centralized_grad = grad - centralization_t * value_momentum

        # 4. Momentumize the centralized gradient
        centralized_momentum.lerp_(centralized_grad, weight=1.0 - beta_t)

        # 5. Update full momentum
        value_momentum.lerp_(grad, weight=1.0 - slow_beta2_t)

        # 6. Combine: exp_avg = centralized_grad.lerp(cm, beta) + centralization * grad.lerp(vm, slow_beta2)
        exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta_t)
        exp_avg.add_(grad.lerp(value_momentum, weight=slow_beta2_t), alpha=centralization_t)

        # 7. Spectral clipping: inlined Newton-Schulz iteration
        # Reshape to 2D
        if dimcount > 2:
            exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
        elif dimcount < 2:
            exp_avg_2d = exp_avg.reshape(1, -1)
        else:
            exp_avg_2d = exp_avg

        # Ortho dtype conversion
        if ortho_dtype is not None:
            orig_dtype = exp_avg_2d.dtype
            exp_avg_2d = exp_avg_2d.to(ortho_dtype)
        else:
            orig_dtype = exp_avg_2d.dtype

        # Save original for adaptive einsum
        if spectral_adaptive:
            M_orig = exp_avg_2d.clone()

        # Flip if first dim is smaller (more rows than cols for NS iteration)
        flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
        if flip:
            exp_avg_2d = exp_avg_2d.T

        # Newton-Schulz iteration (5th order)
        I = torch.eye(exp_avg_2d.shape[1], dtype=exp_avg_2d.dtype, device=exp_avg_2d.device)
        for a, b, c in NS_COEFFS[:num_ns_steps]:
            exp_avg_2d = exp_avg_2d / (torch.linalg.norm(exp_avg_2d).clamp_min_(1e-8))
            A = exp_avg_2d.T @ exp_avg_2d
            exp_avg_2d = exp_avg_2d @ (a * I + b * A + c * A @ A)

        if flip:
            exp_avg_2d = exp_avg_2d.T

        if spectral_adaptive:
            exp_avg_2d = exp_avg_2d * (M_orig.type_as(exp_avg_2d) * exp_avg_2d).sum()

        if ortho_dtype is not None:
            exp_avg_2d = exp_avg_2d.to(orig_dtype)

        # View back to original shape
        full_step = exp_avg_2d.view_as(exp_avg)

        # 8. Post-clip RMS normalization (cap at 1.0)
        full_step.div_(full_step.pow(2).mean().sqrt_().clamp_min_(1))

        # 9. Cautious mask: zero-out where update doesn't align with gradient direction
        aligned = (grad * full_step > 0).to(full_step.dtype)
        mask = aligned.mul_(1.0 - cautious_min_t).add_(cautious_min_t)
        mask.div_(mask.mean().clamp_min_(1e-3))
        full_step.mul_(mask)

        # 10. Adaptive scaling
        if adaptive:
            if input_norm:
                if dimcount > 2:
                    fs_2d = full_step.reshape(len(full_step), -1)
                    ea_2d = exp_avg.reshape(len(exp_avg), -1)
                elif dimcount < 2:
                    fs_2d = full_step.reshape(1, -1)
                    ea_2d = exp_avg.reshape(1, -1)
                else:
                    fs_2d = full_step
                    ea_2d = exp_avg
                scale = (ea_2d * fs_2d).sum(dim=1, keepdim=True).clamp(adaptive_min_t, adaptive_max_t)
                full_step.copy_((fs_2d * scale).view_as(full_step))
            else:
                scale = (exp_avg * full_step).sum().clamp(adaptive_min_t, adaptive_max_t)
                full_step.mul_(scale)

        # 11. Weight decay
        if cautious_weight_decay:
            # Apply weight decay only where gradient and parameter agree in sign
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - weight_decay_t * lr_t * wd_rate_t * cwd_mask)
        else:
            # Standard decoupled weight decay (always applied; alpha is 0 when weight_decay=0)
            full_step.add_(p_fp32, alpha=weight_decay_t * wd_rate_t)

        # 12. Parameter update
        p_fp32.add_(full_step, alpha=-lr_t)

    @staticmethod
    @torch.no_grad()
    def _core_0d_step_fp32(
        grad: torch.Tensor,
        p_fp32: torch.Tensor,
        value_momentum: torch.Tensor,
        centralized_momentum: torch.Tensor,
        denom: torch.Tensor,
        beta_t: float,
        slow_beta2_t: float,
        slow_beta3_t: float,
        centralization_t: float,
        step_t: int,
        cautious_min_t: float,
        adaptive_min_t: float,
        adaptive_max_t: float,
        lr_t: float,
        weight_decay_t: float,
        wd_rate_t: float,
        adaptive: bool,
        cautious_weight_decay: bool = False,
    ) -> None:
        r"""Core per-parameter step for dimcount == 0 (scalar parameters with denom/atan2).

        Scalar hyperparameters are passed as Python floats/ints (not 1-element GPU tensors)
        to avoid per-step CUDA kernel launches from ``Tensor.fill_()``.
        Modifies all tensor arguments in-place.
        Branch booleans are compile-time constants resolved during tracing.
        """
        # 1. Gradient clamping
        grad.clamp_(-step_t, step_t)

        # 2. Centralize gradient
        centralized_grad = grad - centralization_t * value_momentum

        # 3. Momentumize the centralized gradient
        centralized_momentum.lerp_(centralized_grad, weight=1.0 - beta_t)

        # 4. Update full momentum
        value_momentum.lerp_(grad, weight=1.0 - slow_beta2_t)

        # 5. Combine: exp_avg
        exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta_t)
        exp_avg.add_(grad.lerp(value_momentum, weight=slow_beta2_t), alpha=centralization_t)

        # 6. ADOPT-style denominator: snapshot OLD denom before update, then lerp
        current_denom = denom.sqrt()
        denom.lerp_(centralized_grad.pow(2), weight=1.0 - slow_beta3_t)

        # 7. 0D step via atan2: full_step = atan2(exp_avg, sqrt(old_denom)) * (2/pi)
        full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)  # 4/pi ≈ 1.273

        # 8. Cautious mask
        aligned = (grad * full_step > 0).to(full_step.dtype)
        mask = aligned.mul_(1.0 - cautious_min_t).add_(cautious_min_t)
        mask.div_(mask.mean().clamp_min_(1e-3))
        full_step.mul_(mask)

        # 9. Adaptive scaling
        if adaptive:
            scale = (exp_avg * full_step).sum().clamp(adaptive_min_t, adaptive_max_t)
            full_step.mul_(scale)

        # 10. Weight decay
        if cautious_weight_decay:
            # Apply weight decay only where gradient and parameter agree in sign
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - weight_decay_t * lr_t * wd_rate_t * cwd_mask)
        else:
            # Standard decoupled weight decay (always applied; alpha is 0 when weight_decay=0)
            full_step.add_(p_fp32, alpha=weight_decay_t * wd_rate_t)

        # 11. Parameter update
        p_fp32.add_(full_step, alpha=-lr_t)

    def _compile_core_fns(self) -> None:
        r"""Lazily compile the core step functions with torch.compile."""
        if self.defaults.get('compile_step', False):
            try:
                # Raise recompile limit to accommodate diverse parameter shapes
                # Typical training setups can have 100+ unique parameter shapes;
                # set generous limits to avoid FailOnRecompileLimitHit.
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 256
                )
                torch._dynamo.config.cache_size_limit = max(
                    torch._dynamo.config.cache_size_limit, 256
                )
                self._compiled_ge1d = torch.compile(
                    self._core_ge1d_step_fp32, fullgraph=True, dynamic=True
                )
                self._compiled_0d = torch.compile(
                    self._core_0d_step_fp32, fullgraph=True, dynamic=True
                )
                logger.info("OCGOpt core functions compiled with torch.compile(fullgraph=True, dynamic=True).")
            except Exception as e:
                logger.warning(f"torch.compile(fullgraph=True, dynamic=True) failed: {e}. Falling back to uncompiled step.")
                self._compiled_ge1d = self._core_ge1d_step_fp32
                self._compiled_0d = self._core_0d_step_fp32
        else:
            self._compiled_ge1d = self._core_ge1d_step_fp32
            self._compiled_0d = self._core_0d_step_fp32

    @torch.no_grad()
    def _foreach_step(
        self,
        group,
        active_params: list,
        beta: float,
        beta2: float,
        beta3: float,
        slow_beta2: float,
        slow_beta3: float,
        compute_device: torch.device,
    ) -> None:
        r"""Foreach step for 1D parameters (ndim == 1, numel >= 16).

        Batches operations using ``torch._foreach_*`` for better GPU utilization.
        Handles the full OCGOpt pipeline: gradient clamping, RMS normalization,
        centralization, momentum updates, spectral clipping, cautious stepping,
        adaptive scaling, weight decay, and parameter update with optional
        Kahan summation and stochastic rounding.

        Excludes params with lowpass_grad or sim_match enabled (those fall through
        to the standard per-parameter loop).
        """
        # Extract group options
        spectral_adaptive = group['spectral_adaptive']
        spectral_clip_dtype = group['spectral_clip_dtype']
        adaptive = group['adaptive']
        adaptive_min = group['adaptive_min']
        adaptive_max = group['adaptive_max']
        input_norm = group['input_norm']
        cautious_min = group['cautious_min']
        stochastic_fp = group['stochastic_fp']
        use_kahan = group.get('kahan_summation', False)
        lr = group['lr']
        weight_decay = group['weight_decay']
        weight_decay_rate = group['weight_decay_rate']
        centralization = group['centralization']
        step = group['step']

        n = len(active_params)

        # ========= Collect phase: build FP32 tensor lists on compute device =========
        p_fp32_list = [None] * n
        grad_list = [None] * n
        value_momentum_list = [None] * n
        centralized_momentum_list = [None] * n
        kahan_comp_list = [None] * n if use_kahan else None
        state_list = [None] * n
        param_kahan_flags = [False] * n
        param_list = [None] * n

        for idx, p in enumerate(active_params):
            state = self.state[p]
            state_list[idx] = state
            param_list[idx] = p

            value_momentum_list[idx] = state["value_momentum"].to(
                compute_device, non_blocking=True, dtype=torch.float32
            )
            centralized_momentum_list[idx] = state["centralized_momentum"].to(
                compute_device, non_blocking=True, dtype=torch.float32
            )
            grad_list[idx] = p.grad.data.to(
                compute_device, dtype=torch.float32, non_blocking=True
            )
            p_fp32_list[idx] = p.to(
                compute_device, dtype=torch.float32, non_blocking=True
            )

            param_kahan = use_kahan and p.dtype in {torch.float16, torch.bfloat16}
            param_kahan_flags[idx] = param_kahan
            if param_kahan:
                kahan_comp_list[idx] = state['kahan_comp'].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )

        # ========= Kahan pre-compensation =========
        if use_kahan:
            for idx in range(n):
                if param_kahan_flags[idx]:
                    p_fp32_list[idx].add_(kahan_comp_list[idx])

        # ========= BATCH: gradient clamping =========
        torch._foreach_clamp_(grad_list, -step, step)

        # ========= BATCH: RMS normalization =========
        # For 1D params, input_norm and gradient-wide RMS produce the same result
        # (reshape to (1, -1) with per-row RMS is equivalent to scalar RMS for 1 row)
        if input_norm:
            for idx in range(n):
                g = grad_list[idx]
                g_2d = g.reshape(1, -1)
                rms = g_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                g_2d.div_(rms)
        else:
            rms_list = []
            for idx in range(n):
                rms = grad_list[idx].pow(2).mean().sqrt_().clamp_min_(1e-16)
                rms_list.append(rms)
            torch._foreach_div_(grad_list, rms_list)

        # ========= PER-TENSOR: centralization, momentum updates, exp_avg combine =========
        exp_avg_list = [None] * n
        for idx in range(n):
            g = grad_list[idx]
            vm = value_momentum_list[idx]
            cm = centralized_momentum_list[idx]

            # Centralize gradient by removing running average
            centralized_grad = g.sub(vm, alpha=centralization)

            # Momentumize the centralized gradient
            cm.lerp_(centralized_grad, weight=1.0 - beta)

            # Update full momentum
            vm.lerp_(g, weight=1.0 - slow_beta2)

            # exp_avg = centralized_grad.lerp(cm, beta) + centralization * grad.lerp(vm, slow_beta2)
            exp_avg = centralized_grad.lerp(cm, weight=beta)
            exp_avg.add_(g.lerp(vm, weight=slow_beta2), alpha=centralization)
            exp_avg_list[idx] = exp_avg

        # ========= PER-TENSOR: spectral clipping (Newton-Schulz orthogonalization) =========
        full_step_list = [None] * n
        for idx in range(n):
            exp_avg = exp_avg_list[idx]
            # For 1D: reshape to (1, -1), always flips since 1 < N for numel >= 2
            exp_avg_2d = exp_avg.reshape(1, -1)

            flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
            if flip:
                exp_avg_2d = exp_avg_2d.T

            exp_avg_2d = self.clip_func(
                exp_avg_2d, sigma_min=0., sigma_max=0.,
                adaptive=spectral_adaptive, ortho_dtype=spectral_clip_dtype
            )

            if flip:
                exp_avg_2d = exp_avg_2d.T

            full_step = exp_avg_2d.view_as(exp_avg)
            full_step_list[idx] = full_step

        # ========= BATCH: post-clip RMS normalization =========
        rms_list = []
        for idx in range(n):
            rms = full_step_list[idx].pow(2).mean().sqrt_().clamp_min_(1)
            rms_list.append(rms)
        torch._foreach_div_(full_step_list, rms_list)

        # ========= PER-TENSOR: cautious mask =========
        mask_list = [None] * n
        for idx in range(n):
            aligned = (grad_list[idx] * full_step_list[idx] > 0).to(full_step_list[idx].dtype)
            mask = aligned.mul_(1.0 - cautious_min).add_(cautious_min)
            mask = mask.div(mask.mean().clamp_min_(1e-3))
            mask_list[idx] = mask

        # ========= BATCH: apply cautious mask =========
        torch._foreach_mul_(full_step_list, mask_list)

        # ========= PER-TENSOR: adaptive scale compute, BATCH: apply =========
        if adaptive:
            scale_list = [None] * n
            if input_norm:
                for idx in range(n):
                    fs_2d = full_step_list[idx].reshape(1, -1)
                    ea_2d = exp_avg_list[idx].reshape(1, -1)
                    scale = (ea_2d * fs_2d).sum(dim=1, keepdim=True).clamp(
                        adaptive_min, adaptive_max
                    )
                    scale_list[idx] = scale
            else:
                for idx in range(n):
                    scale = (exp_avg_list[idx] * full_step_list[idx]).sum().clamp(
                        adaptive_min, adaptive_max
                    )
                    scale_list[idx] = scale
            torch._foreach_mul_(full_step_list, scale_list)

        # ========= BATCH: weight decay =========
        cautious_weight_decay = group.get('cautious_weight_decay', False)
        if weight_decay != 0:
            if cautious_weight_decay:
                wd_factor = lr * weight_decay * weight_decay_rate ** step
                for idx in range(n):
                    cwd_mask = (grad_list[idx] * p_fp32_list[idx] >= 0).to(p_fp32_list[idx].dtype)
                    p_fp32_list[idx].mul_(1.0 - wd_factor * cwd_mask)
            else:
                wd_alpha = weight_decay * weight_decay_rate ** step
                torch._foreach_add_(full_step_list, p_fp32_list, alpha=wd_alpha)

        # ========= BATCH: LR scale and parameter update =========
        torch._foreach_mul_(full_step_list, -lr)
        torch._foreach_add_(p_fp32_list, full_step_list)

        # ========= Write-back phase =========
        for idx in range(n):
            p = param_list[idx]
            state = state_list[idx]
            p_fp32 = p_fp32_list[idx]
            device = p.device
            srng = self._get_srng_buf(value_momentum_list[idx])

            # Parameter write-back (with optional Kahan compensation)
            if param_kahan_flags[idx]:
                kahan_comp = kahan_comp_list[idx]

                if p.dtype == torch.bfloat16:
                    # Simulate stochastic rounding
                    kahan_sim = p_fp32.clone()
                    sim_int = kahan_sim.view(dtype=torch.int32)
                    if srng is not None:
                        srng.random_(0, 1 << 16)
                        sim_int.add_(srng)
                    else:
                        sim_int.add_(torch.randint_like(sim_int, 0, 1 << 16))
                    sim_int.bitwise_and_(-65536)
                else:
                    # fp16: simulate deterministic rounding
                    kahan_sim = p_fp32.to(p.dtype).to(torch.float32)

                # Write rounded value to parameter
                if device.type == "cpu":
                    p.data.copy_(kahan_sim)
                else:
                    p.data.copy_(kahan_sim, non_blocking=True)

                # Compute new compensation: rounded - exact
                kahan_sim.sub_(p_fp32)

                # Store compensation back to state
                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state['kahan_comp'], kahan_sim, scratch=srng)
                else:
                    state['kahan_comp'].copy_(kahan_sim, non_blocking=True)
            else:
                # Standard parameter write-back
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16 and stochastic_fp:
                        copy_stochastic_(p.data, p_fp32, scratch=srng)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    if p.dtype == torch.bfloat16 and stochastic_fp:
                        copy_stochastic_(p, p_fp32, scratch=srng)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

            # State write-back
            if self.state_storage_dtype == torch.bfloat16:
                copy_stochastic_(
                    state["value_momentum"], value_momentum_list[idx], scratch=srng
                )
                copy_stochastic_(
                    state["centralized_momentum"],
                    centralized_momentum_list[idx],
                    scratch=srng,
                )
            else:
                state["value_momentum"].copy_(
                    value_momentum_list[idx], non_blocking=True
                )
                state["centralized_momentum"].copy_(
                    centralized_momentum_list[idx], non_blocking=True
                )

            # Sync chunking
            if (idx + 1) % group.get('sync_chunk_size', 256) == 0:
                torch.cuda.synchronize()

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta, beta2, beta3 = group["betas"][0], group["betas"][1], group["betas"][2]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]
            stochastic_fp = group["stochastic_fp"]
            kahan_sum = group["kahan_summation"]
            compile_step = group.get('compile_step', False)

            step = group['step']

            # ========= Precompute group-level scalars (avoids redundant pow() per parameter) =========
            slow_beta2 = ((beta2 ** step - beta2) / (beta2 ** step - 1.0))
            slow_beta3 = ((beta3 ** step - beta3) / (beta3 ** step - 1.0))
            wd_rate_step = weight_decay_rate ** step

            # ========= Foreach bucketing: collect 1D params for batched processing =========
            use_foreach = group.get('foreach', False)
            foreach_params = []
            if use_foreach:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # 1D params only (0D has special denom/atan2, 2D+ has shape-specific spectral clipping)
                    if p.grad.ndim == 1 and p.numel() >= 16:
                        # Exclude params with per-tensor FFT filters
                        if group.get('lowpass_grad', 0) == 0 and not group.get('sim_match', False):
                            # Lazy-init state if needed (foreach params skip the per-param init below)
                            state = self.state[p]
                            if len(state) == 0:
                                self._init_param_state(p, state, dimcount=1, use_kahan=kahan_sum)
                            elif kahan_sum and "kahan_comp" not in state:
                                self._late_init_kahan_comp(p, state)

                            foreach_params.append(p)

                if foreach_params:
                    first_device = foreach_params[0].device
                    foreach_compute_device = (
                        torch.cuda.current_device() if first_device.type == "cpu" else first_device
                    )

                    self._foreach_step(
                        group, foreach_params, beta, beta2, beta3,
                        slow_beta2, slow_beta3, foreach_compute_device
                    )
                    torch.cuda.synchronize()

            # Lazily compile core functions on first step
            if self._compiled_ge1d is None:
                self._compile_core_fns()
            core_ge1d_fn = self._compiled_ge1d
            core_0d_fn = self._compiled_0d

            processed_count = 0

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # Skip 1D params already handled by foreach
                if use_foreach and p.grad.ndim == 1 and p.numel() >= 16:
                    if group.get('lowpass_grad', 0) == 0 and not group.get('sim_match', False):
                        continue
                state = self.state[p]
                device = p.device

                grad = p.grad.data

                dimcount = grad.ndim

                use_kahan = kahan_sum and (p.dtype == torch.bfloat16 or p.dtype == torch.float16)

                if len(state) == 0:
                    self._init_param_state(p, state, dimcount, use_kahan)
                elif use_kahan and "kahan_comp" not in state:
                    self._late_init_kahan_comp(p, state)

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                if dimcount < 1:
                    denom = state["denom"].to(
                        compute_device, 
                        non_blocking=True, 
                        dtype=torch.float32
                    )
                value_momentum = state["value_momentum"].to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                centralized_momentum = state["centralized_momentum"].to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                grad = grad.to(compute_device, dtype=torch.float32, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                # ========= Determine if compiled path is usable (bypass for FFT-filtered params) =========
                use_compiled_step = (compile_step and
                               not (dimcount > 0 and group["lowpass_grad"] != 0) and
                               not (dimcount > 0 and group["sim_match"]))

                if use_compiled_step:
                    # --- Compiled path: pass Python floats to avoid Tensor.fill_() CUDA kernel launches ---
                    betas = group['betas']

                    # Kahan pre-compensation: p_fp32 -= kahan_comp
                    if use_kahan:
                        kahan_comp = state["kahan_comp"].to(compute_device, non_blocking=True, dtype=torch.float32)
                        p_fp32.sub_(kahan_comp)

                    if dimcount >= 1:
                        core_ge1d_fn(
                            grad, p_fp32,
                            value_momentum, centralized_momentum,
                            betas[0], slow_beta2,
                            group['centralization'], step,
                            group['cautious_min'], group['adaptive_min'],
                            group['adaptive_max'], group['lr'],
                            group['weight_decay'], wd_rate_step,
                            group['input_norm'], group['adaptive'],
                            group['spectral_adaptive'],
                            group['spectral_clip_dtype'],
                            group.get('num_ns_steps', len(NS_COEFFS)),
                            group.get('cautious_weight_decay', False),
                        )
                    else:
                        core_0d_fn(
                            grad, p_fp32,
                            value_momentum, centralized_momentum,
                            denom,
                            betas[0], slow_beta2,
                            slow_beta3, group['centralization'],
                            step,
                            group['cautious_min'], group['adaptive_min'],
                            group['adaptive_max'], group['lr'],
                            group['weight_decay'], wd_rate_step,
                            group['adaptive'],
                            group.get('cautious_weight_decay', False),
                        )
                else:
                    # --- Original uncompiled path ---
                    # ADOPT-style clamping for early stability / to prevent NaNs
                    grad.clamp_(-step, step)

                    # Low-pass filter via FFT, maintains direction
                    if dimcount > 0 and group["lowpass_grad"] != 0:
                        grad = torch.copysign(filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs(), grad)

                    # Move RMS to 1.0, input-feature-wise if 2D or larger, otherwise utilize standard gradient-wide RMS normalization.
                    if dimcount >= 1 and group["input_norm"]:
                        if dimcount > 2:
                            grad_2d = grad.reshape(len(grad), -1) # Make 2D if conv or 1 dim
                        elif dimcount < 2:
                            grad_2d = grad.reshape(1, -1) # Make 2D if conv or 1 dim
                        else:
                            grad_2d = grad

                        rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16) # Cap at RMS of 1.0
                        grad_2d.div_(rms)
                    else:
                        rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16) # Cap at RMS of 1.0
                        grad.div_(rms)

                    # ADOPT-style denominator update (un-updated denom)
                    if dimcount < 1:
                        current_denom = denom.sqrt()

                    # Centralize gradient by removing running average
                    centralized_grad = grad.sub(value_momentum, alpha=centralization)

                    # Momentumize the centralized gradient
                    centralized_momentum.lerp_(centralized_grad, weight=1. - beta)

                    # Update full momentum
                    value_momentum.lerp_(grad, weight=1. - slow_beta2)

                    # Add back full momentum to the centralized gradient
                    exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta).add_(grad.lerp(value_momentum, weight=slow_beta2), alpha=centralization)

                    # Update denominator with either centralized gradient, or its mean when utilizing a sign-based gradient
                    if dimcount < 1:
                        denom.lerp_(centralized_grad.pow(2), weight=1. - slow_beta3)

                    # Frequency matching the momentumized update with the current step's gradient
                    if dimcount > 0 and group["sim_match"]:
                        exp_avg = similarity_fft(exp_avg, grad)

                    # Spectral Clipping / Newton Schulz iters
                    if dimcount >= 1:
                        if dimcount > 2:
                            exp_avg_2d = exp_avg.reshape(len(exp_avg), -1) # Make 2D if conv or 1 dim
                        elif dimcount < 2:
                            exp_avg_2d = exp_avg.reshape(1, -1) # Make 2D if conv or 1 dim
                        else:
                            exp_avg_2d = exp_avg

                        flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
                        if flip:
                            exp_avg_2d = exp_avg_2d.T # Flip if first dim is larger

                        exp_avg_2d = self.clip_func(exp_avg_2d, sigma_min=0., sigma_max=0., adaptive=group["spectral_adaptive"], ortho_dtype=group["spectral_clip_dtype"])

                        if flip:
                            exp_avg_2d = exp_avg_2d.T

                        full_step = exp_avg_2d.view_as(exp_avg)

                        full_step.div_(full_step.pow(2).mean().sqrt_().clamp_min_(1))
                    else:
                        full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

                    # Cautious update (zero-out update where the update isn't in the direction of the current gradient)
                    aligned = (grad * full_step > 0).to(full_step.dtype)
                    scale_factor_mask = aligned.mul_(1.0 - group["cautious_min"]).add_(group["cautious_min"])
                    scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))

                    # Apply Cautious update
                    full_step.mul_(scale_factor_mask)

                    # Scale the full step with the gradient
                    if group["adaptive"]:
                        if dimcount >= 1 and group["input_norm"]:
                            if dimcount > 2:
                                full_step_2d = full_step.reshape(len(full_step), -1) # Make 2D if conv or 1 dim
                                exp_avg_2d = exp_avg.reshape(len(exp_avg), -1) # Make 2D if conv or 1 dim
                            elif dimcount < 2:
                                full_step_2d = full_step.reshape(1, -1) # Make 2D if conv or 1 dim
                                exp_avg_2d = exp_avg.reshape(1, -1) # Make 2D if conv or 1 dim
                            else:
                                full_step_2d = full_step
                                exp_avg_2d = exp_avg

                            scale_factor = (exp_avg_2d * full_step_2d).sum(dim=1, keepdim=True).clamp(group["adaptive_min"], group["adaptive_max"])

                            full_step = (full_step_2d * scale_factor).view_as(full_step)
                        else:
                            scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])
                            full_step.mul_(scale_factor)

                    # Perform weight decay
                    if weight_decay != 0:
                        if group.get('cautious_weight_decay', False):
                            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
                            p_fp32.mul_(1.0 - lr * weight_decay * wd_rate_step * cwd_mask)
                        else:
                            full_step.add_(p_fp32, alpha=weight_decay * wd_rate_step)

                    # Apply Update (with optional Kahan Summation)
                    update_step = full_step.mul(-lr)

                srng = self._get_srng_buf(p_fp32)

                if use_compiled_step:
                    # --- Compiled path: parameter already updated in p_fp32 by core fn ---
                    if use_kahan:
                        # Kahan post-compensation: simulate rounding, compute new compensation
                        p_simulated, new_comp = self._simulate_kahan_rounding(
                            p_fp32, p.dtype, stochastic_fp, scratch=srng
                        )
                        if device.type == "cpu":
                            p.data.copy_(p_simulated)
                        else:
                            p.data.copy_(p_simulated, non_blocking=True)

                        # Store the new compensation
                        if self.state_storage_dtype == torch.bfloat16:
                            copy_stochastic_(state["kahan_comp"], new_comp, scratch=srng)
                        else:
                            state["kahan_comp"].copy_(new_comp, non_blocking=True)
                    else:
                        # Standard parameter write-back (p_fp32 already updated by compiled fn)
                        if device.type == "cpu":
                            if p.dtype == torch.bfloat16 and stochastic_fp:
                                copy_stochastic_(p.data, p_fp32, scratch=srng)
                            else:
                                p.data.copy_(p_fp32)
                        else:
                            if p.dtype == torch.bfloat16 and stochastic_fp:
                                copy_stochastic_(p, p_fp32, scratch=srng)
                            else:
                                p.data.copy_(p_fp32, non_blocking=True)
                else:
                    # --- Original uncompiled parameter update + write-back ---
                    if use_kahan:
                        # Kahan Summation Logic
                        # Kahan effectively works by subtracting the error ("compensation") from the *input* of the summation.
                        # This compensation accumulates the bits that were too small to be added to the weight in previous steps.
                        
                        kahan_comp = state["kahan_comp"].to(compute_device, non_blocking=True, dtype=torch.float32)
                        
                        # 1. Adjust the update by the compensation
                        update_step.sub_(kahan_comp)
                        
                        # 2. Add adjusted update to the high-precision weight
                        p_fp32.add_(update_step)

                        # 3. Simulate the lossy update to calculate new compensation
                        p_simulated, new_comp = self._simulate_kahan_rounding(
                            p_fp32, p.dtype, stochastic_fp, scratch=srng
                        )
                        
                        # Apply to actual parameter
                        if device.type == "cpu":
                            p.data.copy_(p_simulated)
                        else:
                            p.data.copy_(p_simulated, non_blocking=True)

                        # Store the new compensation
                        if self.state_storage_dtype == torch.bfloat16:
                            copy_stochastic_(state["kahan_comp"], new_comp, scratch=srng)
                        else:
                            state["kahan_comp"].copy_(new_comp, non_blocking=True)

                    else:
                        # Standard Update Path
                        p_fp32.add_(update_step)
        
                        # Device-to-Host copy (with stochastic rounding for bf16)
                        if device.type == "cpu":
                            if p.dtype == torch.bfloat16 and stochastic_fp:
                                copy_stochastic_(p.data, p_fp32, scratch=srng)
                            else:
                                p.data.copy_(p_fp32)
                        else:
                            if p.dtype == torch.bfloat16 and stochastic_fp:
                                copy_stochastic_(p, p_fp32, scratch=srng)
                            else:
                                p.data.copy_(p_fp32, non_blocking=True)

                # Store State
                if self.state_storage_dtype == torch.bfloat16:
                    if dimcount < 1:
                        copy_stochastic_(state["denom"], denom, scratch=srng)
                    copy_stochastic_(state["value_momentum"], value_momentum, scratch=srng)
                    copy_stochastic_(
                        state["centralized_momentum"], centralized_momentum, scratch=srng)
                else:
                    if dimcount < 1:
                        state["denom"].copy_(denom, non_blocking=True)
                    state["value_momentum"].copy_(value_momentum, non_blocking=True)
                    state["centralized_momentum"].copy_(centralized_momentum, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                processed_count += 1
                if processed_count % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()
            
        return loss