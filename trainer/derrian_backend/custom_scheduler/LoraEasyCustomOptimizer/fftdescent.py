# FFTDescent from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math
import logging
from functools import lru_cache

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
NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]
# New coeffs from https://kexue.fm/archives/11059, may enable later.
"""
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]
"""
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
    M = M / (torch.linalg.norm(M) + 1e-20)
    # Pre-allocate identity matrix once — shape (M.shape[1], M.shape[1]) is constant across NS iterations
    n = M.shape[1]
    I = torch.eye(n, dtype=M.dtype, device=M.device)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T.contiguous()
    if adaptive:
        M = torch.einsum('ij,ij,ab->ab', M_orig.type_as(M), M, M)
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M

@torch.no_grad()
def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if adaptive:
        W_orig = W.clone()
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = orthogonalize(W, num_ns_steps)
    eye_m = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    result = 0.5 * (
        (sigma_min + sigma_max) * eye_m
        + (sigma_min * OW - W) @ orthogonalize(sigma_min * OW - W, num_ns_steps).T
        - (sigma_max * OW - W) @ orthogonalize(sigma_max * OW - W, num_ns_steps).T
    ) @ OW
    if adaptive:
        result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
    return result.to(orig_dtype)

@torch.no_grad()
def spectral_clip_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=None, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def spectral_clip_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=None, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

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

@torch.no_grad()
def filter_grad(grad, fft_alpha=1.0):
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    filter_weights = _get_filter_weights(grad.shape, str(grad.device), fft_alpha)
    filtered_grad_freq = grad_freq * filter_weights
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    return modified_grad.real

class FFTDescent(Optimizer):
    r"""
    FFTDescent: ***TEMPORARY NAME***

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        beta (float, float, float):
            Coefficient used for computing the running average (default: 0.95)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: True, recommended to keep on True if possible / slowdown is negligible).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase). (default: True).
        spectral_clip_dtype (torch.dtype or None):
            Compute spectral clipping in this dtype. (default: None, is determined based on spectral_clip_compile (float16 if uncompiled, float32 if compiled)).
        spectral_min (float):
            The minimum value of the spectral magnitude. Ought to be lower than spectral_max. (default: -1.0).
        spectral_max (float):
            The maximum value of the spectral magnitude. (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: False).
        lowpass_grad (float):
            Pre-condition the gradient with a lowpass filter via FFT (default: 1.0).
        sign_momentum (float):
            Decouple the momentum from the sign/direction, value is the coefficient used for computing the sign's running average (default: 0.9).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
        compile_step (bool):
            Compile the entire per-parameter step function with torch.compile(fullgraph=True, dynamic=False).
            When True, the full FFT+momentum+spectral_clip+update pipeline is fused into a single
            compiled graph, subsuming spectral_clip_compile.  Requires PyTorch 2.x with dynamo support.
            Mutually exclusive with ``foreach`` (compile_step takes priority).
            (default: False).
        foreach (bool):
            Use ``torch._foreach_*`` operations to batch element-wise tensor
            operations (momentum update, weight decay, parameter update) across
            all eligible parameters per group, reducing GPU kernel-launch
            overhead.  Per-tensor operations (FFT low-pass filter, spectral
            clipping, atan2 normalization) remain sequential.  Mutually exclusive
            with ``compile_step`` (compile_step takes priority).
            (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        spectral_min: float = -1.,
        spectral_max: float = 1.,
        spectral_adaptive: bool = False,
        lowpass_grad: float = 1.0,
        sign_momentum: float = 0.9,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        foreach: bool = False,
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Optimizer argument '{key}' passed into FFTDescent. It will be ignored."
            )


        self._init_lr = lr
        self._compile_step = compile_step
        self._foreach = foreach
        self._srng_buf = None  # Reusable stochastic-rounding scratch buffer (lazy init)

        if spectral_clip:
            if compile_step:
                # Spectral clipping will be inlined into the compiled full-step graph;
                # no need for a separately compiled spectral_clip wrapper.
                self.clip_func = None
            else:
                self.clip_func = spectral_clip_compiled_func if spectral_clip_compile else spectral_clip_func

        defaults = dict(
            lr = lr,
            beta = beta,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            spectral_clip = spectral_clip,
            spectral_clip_compile = spectral_clip_compile,
            spectral_clip_dtype = spectral_clip_dtype,
            spectral_min = spectral_min,
            spectral_max = spectral_max,
            spectral_adaptive = spectral_adaptive,
            lowpass_grad = lowpass_grad,
            sign_momentum = sign_momentum,
            stochastic_fp = stochastic_fp,
            foreach = foreach,
        )

        super(FFTDescent, self).__init__(params, defaults)

        # Pre-allocated scalar tensors per device (lazily populated by _get_scalar_tensors)
        self._scalar_tensors = {}

        # Set up the step function (compiled or uncompiled)
        self._compiled_step = self._fftdescent_step_fp32
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
                    self._fftdescent_step_fp32, fullgraph=True, dynamic=False
                )
            logging.info(
                "FFTDescent full step compiled with torch.compile(fullgraph=True, dynamic=False)."
            )
        except Exception as e:
            logging.warning(
                f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. "
                f"Falling back to uncompiled step."
            )
            self._compiled_step = self._fftdescent_step_fp32

    def _get_scalar_tensors(self, device: torch.device) -> dict:
        r"""Return pre-allocated FP32 scalar tensors for *device*, creating them on first access.

        The caller should ``.fill_(value)`` each tensor before passing to the compiled step
        so that torch.compile does not re-specialize on changing Python float values.
        """
        if device not in self._scalar_tensors:
            self._scalar_tensors[device] = {
                'lr_t':           torch.tensor(0.0, dtype=torch.float32, device=device),
                'beta_t':         torch.tensor(0.0, dtype=torch.float32, device=device),
                'wd_scaled_t':    torch.tensor(0.0, dtype=torch.float32, device=device),
                'sign_mom_coeff_t': torch.tensor(0.0, dtype=torch.float32, device=device),
                'spectral_min_t': torch.tensor(0.0, dtype=torch.float32, device=device),
                'spectral_max_t': torch.tensor(0.0, dtype=torch.float32, device=device),
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
    def _fftdescent_step_fp32(
        # --- Tensor inputs (all FP32, modified in-place where noted) ---
        p_data:            torch.Tensor,  # [in-out] parameter clone in working precision
        grad:              torch.Tensor,  # gradient in working precision (read-only)
        momentum:          torch.Tensor,  # [in-out] momentum state clone
        sign_momentum:     torch.Tensor,  # [in-out] sign-momentum state clone (dummy if unused)
        filter_weights:    torch.Tensor,  # pre-computed FFT low-pass weights (dummy if unused)
        # --- Scalar tensors (avoid Python-float specialization) ---
        lr_t:              torch.Tensor,  # scalar float32
        beta_t:            torch.Tensor,  # scalar float32
        wd_scaled_t:       torch.Tensor,  # scalar float32, pre-computed wd * wd_rate**step
        sign_mom_coeff_t:  torch.Tensor,  # scalar float32
        spectral_min_t:    torch.Tensor,  # scalar float32
        spectral_max_t:    torch.Tensor,  # scalar float32
        # --- Compile-time constants (Python scalars → resolved at trace time) ---
        do_spectral_clip:  bool,          # True when dimcount >= 2 AND spectral_clip enabled
        use_sign_momentum: bool,          # True when sign_mom_coeff != 0
        do_lowpass:        bool,          # True when dimcount > 0 AND lowpass_grad != 0
        spectral_adaptive: bool,          # adaptive spectral clipping flag
        has_weight_decay:  bool,          # True when weight_decay != 0
        step_is_one:       bool,          # True on the very first optimization step
        needs_reshape:     bool,          # True when dimcount > 2 (conv / 3-D+ tensors)
        num_ns_steps:      int,           # number of Newton-Schulz iterations (len(NS_COEFFS))
        ortho_dtype,                     # torch.dtype for spectral clip computation
    ) -> None:
        r"""Core per-parameter step in working precision.

        All tensor inputs live on the compute device.  Boolean / int / dtype
        arguments are compile-time constants resolved during tracing so each
        distinct combination produces a separate compiled graph (bounded by
        the number of parameter-group configurations).

        This function is designed to be called from ``torch.compile`` with
        ``fullgraph=True, dynamic=False``.  It calls ``_spectral_clip`` and
        ``orthogonalize`` directly (the uncompiled pure-tensor-math versions)
        which the compiler traces through and inlines into a single fused graph.
        """

        # ---- 1. Low-pass filter via FFT --------------------------------
        # When do_lowpass is True, filter the gradient magnitude through a
        # Gaussian low-pass in the frequency domain while preserving the
        # original gradient's sign.
        if do_lowpass:
            grad_sign = grad.sign()
            grad_freq = torch.fft.fftn(grad, norm='ortho')
            filtered = torch.fft.ifftn(grad_freq * filter_weights, norm='ortho').real
            g = filtered.abs().mul_(grad_sign)
        else:
            g = grad

        # ---- 2. Zero gradient on first step ----------------------------
        # On step 1 the gradient is zeroed so momentum starts from zero.
        if step_is_one:
            g = torch.zeros_like(g)

        # ---- 3. Momentum update ----------------------------------------
        # When using sign-momentum the momentum tracks |grad| (magnitude only);
        # otherwise it tracks the raw gradient.
        # NOTE: We pre-multiply by (1 - coeff) and use .add_(tensor) without
        # alpha= because torch.compile(fullgraph=True) cannot extract a Python
        # scalar from a tensor via _local_scalar_dense (data-dependent operator).
        one_minus_beta = 1.0 - beta_t
        if use_sign_momentum:
            momentum.mul_(beta_t).add_(g.abs() * one_minus_beta)
        else:
            momentum.mul_(beta_t).add_(g * one_minus_beta)

        # ---- 4. Nesterov-like look-ahead --------------------------------
        # Expand lerp manually: a.lerp(b, w) = a*(1-w) + b*w
        one_minus_sign = 1.0 - sign_mom_coeff_t
        if use_sign_momentum:
            sign_momentum.mul_(sign_mom_coeff_t).add_(g.sign() * one_minus_sign)
            c_t = g.abs() * one_minus_beta + momentum * beta_t
        else:
            c_t = g * one_minus_beta + momentum * beta_t

        # ---- 5. Spectral clipping or atan2 normalization ----------------
        if do_spectral_clip:
            # Reshape conv / high-dim tensors to 2-D for matrix operations
            if needs_reshape:
                c_t_2d = c_t.reshape(c_t.shape[0], -1).contiguous()
            else:
                c_t_2d = c_t

            # Transpose so the smaller dim is rows (Newton-Schulz prefers tall-skinny)
            flip = c_t_2d.shape[0] > c_t_2d.shape[1]
            if flip:
                c_t_2d = c_t_2d.T.contiguous()

            # Call the uncompiled spectral clip (pure tensor math, inlined by the compiler)
            full_step = _spectral_clip(
                c_t_2d,
                sigma_min=spectral_min_t,
                sigma_max=spectral_max_t,
                ortho_dtype=ortho_dtype,
                num_ns_steps=num_ns_steps,
                adaptive=spectral_adaptive,
            )

            if flip:
                full_step = full_step.T.contiguous()

            # Normalize via atan2 (bounded [-1, 1] × 4/π) using momentum as denominator
            full_step = full_step.view_as(c_t).contiguous().atan2(momentum.abs()).mul_(1.27323954474)
        else:
            # No spectral clipping — direct atan2 normalization
            full_step = c_t.atan2(momentum.abs()).mul_(1.27323954474)

        # ---- 6. Apply sign-momentum direction --------------------------
        if use_sign_momentum:
            full_step = full_step.mul(sign_momentum)

        # ---- 7. Decoupled weight decay ---------------------------------
        if has_weight_decay:
            full_step = full_step + p_data * wd_scaled_t

        # ---- 8. Parameter update ---------------------------------------
        p_data.sub_(full_step * lr_t)

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
    # Compiled step path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_compiled(self) -> None:
        r"""Full-step compiled path.

        For each parameter: prepares FP32 working copies, updates scalar
        tensors, calls the (potentially compiled) ``_fftdescent_step_fp32``,
        then copies results back with optional stochastic rounding.
        """
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']
            # Hoist constant group lookups to locals
            sign_mom_coeff = group["sign_momentum"]
            stochastic_fp = group["stochastic_fp"]
            lowpass_grad = group["lowpass_grad"]
            do_spectral_clip = group["spectral_clip"]
            spectral_min = group["spectral_min"]
            spectral_max = group["spectral_max"]
            spectral_adaptive = group["spectral_adaptive"]
            spectral_clip_dtype = group["spectral_clip_dtype"]
            # Precompute weight decay scale factor (constant for all params in this group)
            wd_scaled = weight_decay * (weight_decay_rate ** step) if weight_decay != 0 else 0.0
            use_sign_momentum = sign_mom_coeff != 0
            # Resolve ortho_dtype: None → float32 for the compiled path
            ortho_dtype = spectral_clip_dtype if spectral_clip_dtype is not None else torch.float32

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data
                dimcount = grad.ndim

                # ---- State initialization --------------------------------
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(grad)
                    if use_sign_momentum:
                        state["sign_momentum"] = torch.zeros_like(grad)

                # ---- Prepare FP32 working copies -------------------------
                use_fp32 = p.dtype in {torch.float16, torch.bfloat16} and stochastic_fp
                if use_fp32:
                    grad_work = grad.to(torch.float32)
                    p_work = p.detach().clone().to(torch.float32)
                    momentum_work = state["momentum"].detach().clone().to(torch.float32)
                    sign_momentum_work = (
                        state["sign_momentum"].detach().clone().to(torch.float32)
                        if use_sign_momentum
                        else torch.empty(0, dtype=torch.float32, device=p.device)
                    )
                else:
                    grad_work = grad
                    p_work = p.detach().clone()
                    momentum_work = state["momentum"].detach().clone()
                    sign_momentum_work = (
                        state["sign_momentum"].detach().clone()
                        if use_sign_momentum
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
                st['beta_t'].fill_(beta)
                st['wd_scaled_t'].fill_(wd_scaled)
                st['sign_mom_coeff_t'].fill_(sign_mom_coeff)
                st['spectral_min_t'].fill_(spectral_min)
                st['spectral_max_t'].fill_(spectral_max)

                # ---- Call compiled / uncompiled core step -----------------
                self._compiled_step(
                    p_work, grad_work, momentum_work, sign_momentum_work, filter_weights,
                    # Scalar tensors
                    st['lr_t'], st['beta_t'], st['wd_scaled_t'],
                    st['sign_mom_coeff_t'], st['spectral_min_t'], st['spectral_max_t'],
                    # Compile-time constants
                    do_spectral_clip and dimcount >= 2,  # do_spectral_clip
                    use_sign_momentum,
                    do_lowpass,
                    spectral_adaptive,
                    weight_decay != 0,     # has_weight_decay
                    step == 1,             # step_is_one
                    dimcount > 2,          # needs_reshape
                    len(NS_COEFFS),        # num_ns_steps
                    ortho_dtype,
                )

                # ---- Copy back with optional stochastic rounding ---------
                if use_fp32:
                    copy_stochastic_(state["momentum"], momentum_work)
                    if use_sign_momentum:
                        copy_stochastic_(state["sign_momentum"], sign_momentum_work)
                    copy_stochastic_(p, p_work)
                else:
                    state["momentum"].copy_(momentum_work)
                    if use_sign_momentum:
                        state["sign_momentum"].copy_(sign_momentum_work)
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
        * **First-step gradient zeroing**: ``_foreach_zero_``.
        * **Momentum update**: ``_foreach_mul_``, ``_foreach_add_``.
        * **Nesterov look-ahead / sign-momentum**: ``_foreach_sign``,
          ``_foreach_mul_``, ``_foreach_add_``, ``_foreach_lerp_`` /
          ``_foreach_lerp``.
        * **Pre-computed |momentum|**: ``_foreach_abs`` before the
          per-tensor atan2 normalization.
        * **4/pi scaling**: ``_foreach_mul_`` extracted from the per-tensor
          loop.
        * **Sign-momentum direction / weight decay / param update**:
          ``_foreach_mul_``, ``_foreach_add_``.
        * **Non-stochastic write-back**: ``_foreach_copy_``.

        Per-tensor operations that *cannot* be foreach'd (FFT roundtrip,
        spectral clipping via Newton-Schulz, atan2 normalization) remain
        sequential since they depend on tensor shape or involve matrix ops.

        Eligible parameters: ``numel >= 16``.  Parameters below this
        threshold fall through to the native per-parameter path on the
        next ``step()`` call (the native path is always available as a
        fallback when ``foreach=False``).
        """
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # ---- Hoist group-level scalars --------------------------------
            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']
            sign_mom_coeff = group["sign_momentum"]
            stochastic_fp = group["stochastic_fp"]
            lowpass_grad = group["lowpass_grad"]
            do_spectral_clip = group["spectral_clip"]
            spectral_min = group["spectral_min"]
            spectral_max = group["spectral_max"]
            spectral_adaptive = group["spectral_adaptive"]
            spectral_clip_dtype = group["spectral_clip_dtype"]

            wd_scaled = (
                weight_decay * (weight_decay_rate ** step)
                if weight_decay != 0
                else 0.0
            )
            use_sign_momentum = sign_mom_coeff != 0

            # ---- Collect eligible params (numel >= 16) --------------------
            foreach_params: list = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.numel() >= 16:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p.grad)
                        if use_sign_momentum:
                            state["sign_momentum"] = torch.zeros_like(p.grad)
                    foreach_params.append(p)

            if not foreach_params:
                continue

            # Determine compute device (GPU)
            first_device = foreach_params[0].device
            compute_device = (
                torch.cuda.current_device()
                if first_device.type == "cpu"
                else first_device
            )

            n = len(foreach_params)

            # ==== Collect phase: build FP32 tensor lists on compute device =
            p_fp32_list = [None] * n
            grad_list = [None] * n
            momentum_list = [None] * n
            sign_momentum_list = [None] * n if use_sign_momentum else None
            state_list = [None] * n
            param_list = [None] * n
            filter_weights_list = [None] * n
            dimcount_list = [0] * n
            do_spectral_clip_list = [False] * n
            use_fp32_list = [False] * n

            for idx, p in enumerate(foreach_params):
                state = self.state[p]
                state_list[idx] = state
                param_list[idx] = p

                grad = p.grad.data
                dimcount = grad.ndim
                dimcount_list[idx] = dimcount
                do_spectral_clip_list[idx] = do_spectral_clip and dimcount >= 2

                use_fp32 = (
                    p.dtype in {torch.float16, torch.bfloat16} and stochastic_fp
                )
                use_fp32_list[idx] = use_fp32

                target_dtype = torch.float32 if use_fp32 else p.dtype

                grad_list[idx] = grad.to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                p_fp32_list[idx] = p.detach().to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                momentum_list[idx] = state["momentum"].to(
                    compute_device, dtype=target_dtype, non_blocking=True
                )
                if use_sign_momentum:
                    sign_momentum_list[idx] = state["sign_momentum"].to(
                        compute_device, dtype=target_dtype, non_blocking=True
                    )

                # Pre-compute FFT filter weights (cached per shape/device/alpha)
                # Use compute_device so weights are on the same device as grad_list[idx]
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

            # ==== Per-tensor: FFT low-pass filter ==========================
            # Split into FFT (per-tensor, shape-dependent) and element-wise
            # ops (batched via foreach to reduce kernel-launch overhead).
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

            # ==== BATCH: zero gradient on first step =======================
            if step == 1:
                torch._foreach_zero_(grad_list)

            # ==== BATCH: momentum update ===================================
            # When using sign-momentum the momentum tracks |grad| (magnitude
            # only); otherwise it tracks the raw gradient.
            torch._foreach_mul_(momentum_list, beta)
            if use_sign_momentum:
                # momentum = beta * momentum + (1 - beta) * |grad|
                g_abs_list = torch._foreach_abs(grad_list)
                torch._foreach_add_(momentum_list, g_abs_list, alpha=1.0 - beta)
            else:
                torch._foreach_add_(momentum_list, grad_list, alpha=1.0 - beta)

            # ==== Nesterov look-ahead + sign momentum ======================
            if use_sign_momentum:
                # sign_momentum = coeff * sign_momentum + (1-coeff) * grad.sign()
                grad_sign_list = torch._foreach_sign(grad_list)
                torch._foreach_mul_(sign_momentum_list, sign_mom_coeff)
                torch._foreach_add_(
                    sign_momentum_list, grad_sign_list,
                    alpha=1.0 - sign_mom_coeff,
                )

                # c_t = g.abs().lerp(momentum, weight=beta)
                # Reuse g_abs_list already computed above
                torch._foreach_lerp_(g_abs_list, momentum_list, weight=beta)
                c_t_list = g_abs_list
            else:
                # c_t = g.lerp(momentum, weight=beta)
                c_t_list = torch._foreach_lerp(
                    grad_list, momentum_list, weight=beta
                )

            # ==== BATCH: pre-compute |momentum| for atan2 normalization ====
            # Batches N individual abs() calls into fewer foreach kernel
            # launches.  Used by both spectral-clip and non-clip paths.
            momentum_abs_list = torch._foreach_abs(momentum_list)

            # ==== Per-tensor: spectral clipping or atan2 normalization =====
            # Spectral clipping (Newton-Schulz iterations) involves per-tensor
            # matrix products and cannot be foreach'd.  atan2 has no foreach
            # variant.  The final 4/π scaling is extracted and batched below.
            _FOUR_OVER_PI = 1.27323954474
            full_step_list = [None] * n
            for idx in range(n):
                c_t = c_t_list[idx]

                if do_spectral_clip_list[idx]:
                    dimcount = dimcount_list[idx]
                    # Reshape conv / high-dim tensors to 2-D for matrix ops
                    if dimcount > 2:
                        c_t_2d = c_t.reshape(c_t.shape[0], -1)
                    else:
                        c_t_2d = c_t

                    # Transpose so smaller dim is rows (NS prefers tall-skinny)
                    flip = c_t_2d.shape[0] > c_t_2d.shape[1]
                    if flip:
                        c_t_2d = c_t_2d.T

                    full_step = self.clip_func(
                        c_t_2d,
                        sigma_min=spectral_min,
                        sigma_max=spectral_max,
                        adaptive=spectral_adaptive,
                        ortho_dtype=spectral_clip_dtype,
                    )

                    if flip:
                        full_step = full_step.T

                    # atan2 normalization (bounded [-1, 1] × 4/π)
                    full_step = (
                        full_step.view_as(c_t)
                        .atan2(momentum_abs_list[idx])
                    )
                else:
                    # No spectral clipping — direct atan2 normalization
                    full_step = c_t.atan2(momentum_abs_list[idx])

                full_step_list[idx] = full_step

            # BATCH: scale by 4/π (extracted from per-tensor loop to reduce
            # N individual mul_ kernel launches to a single foreach call)
            torch._foreach_mul_(full_step_list, _FOUR_OVER_PI)

            # ==== BATCH: apply sign-momentum direction =====================
            if use_sign_momentum:
                torch._foreach_mul_(full_step_list, sign_momentum_list)

            # ==== BATCH: decoupled weight decay ============================
            if weight_decay != 0:
                torch._foreach_add_(full_step_list, p_fp32_list, alpha=wd_scaled)

            # ==== BATCH: parameter update ==================================
            torch._foreach_add_(p_fp32_list, full_step_list, alpha=-lr)

            # ==== Write-back phase =========================================
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
                momentum_fp32 = momentum_list[idx]
                scratch = self._get_srng_buf(momentum_fp32)
                copy_stochastic_(
                    state_list[idx]["momentum"], momentum_fp32, scratch=scratch
                )
                if use_sign_momentum:
                    copy_stochastic_(
                        state_list[idx]["sign_momentum"],
                        sign_momentum_list[idx],
                        scratch=scratch,
                    )
                copy_stochastic_(param_list[idx], p_fp32_list[idx], scratch=scratch)

            # Non-stochastic path — batched via torch._foreach_copy_
            if non_stoch_indices:
                ns_mom_dst = [
                    state_list[idx]["momentum"] for idx in non_stoch_indices
                ]
                ns_mom_src = [momentum_list[idx] for idx in non_stoch_indices]
                torch._foreach_copy_(ns_mom_dst, ns_mom_src)
                if use_sign_momentum:
                    ns_sm_dst = [
                        state_list[idx]["sign_momentum"]
                        for idx in non_stoch_indices
                    ]
                    ns_sm_src = [
                        sign_momentum_list[idx] for idx in non_stoch_indices
                    ]
                    torch._foreach_copy_(ns_sm_dst, ns_sm_src)
                ns_p_dst = [param_list[idx] for idx in non_stoch_indices]
                ns_p_src = [p_fp32_list[idx] for idx in non_stoch_indices]
                torch._foreach_copy_(ns_p_dst, ns_p_src)

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
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']
            # Hoist constant group lookups to locals for faster access in the per-parameter inner loop
            sign_mom_coeff = group["sign_momentum"]
            stochastic_fp = group["stochastic_fp"]
            lowpass_grad = group["lowpass_grad"]
            do_spectral_clip = group["spectral_clip"]
            spectral_min = group["spectral_min"]
            spectral_max = group["spectral_max"]
            spectral_adaptive = group["spectral_adaptive"]
            spectral_clip_dtype = group["spectral_clip_dtype"]
            # Precompute weight decay scale factor (constant for all params in this group)
            wd_scaled = weight_decay * (weight_decay_rate ** step) if weight_decay != 0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                dimcount = grad.ndim

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["momentum"] = torch.zeros_like(grad)
                    # Exponential moving average of sign
                    if sign_mom_coeff != 0:
                        state["sign_momentum"] = torch.zeros_like(grad)

                # Detach and clone once at the correct dtype — avoids redundant allocation
                # when stochastic_fp is enabled for half-precision parameters (previously
                # cloned at original dtype then immediately re-cloned at float32)
                use_fp32 = p.dtype in {torch.float16, torch.bfloat16} and stochastic_fp
                if use_fp32:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)
                    momentum = state["momentum"].detach().clone().to(torch.float32)
                    if sign_mom_coeff != 0:
                        sign_momentum = state["sign_momentum"].detach().clone().to(torch.float32)
                else:
                    p_fp32 = p.detach().clone()
                    momentum = state["momentum"].detach().clone()
                    if sign_mom_coeff != 0:
                        sign_momentum = state["sign_momentum"].detach().clone()

                # Low-pass filter via FFT
                if dimcount > 0:
                    grad = filter_grad(grad, fft_alpha=lowpass_grad).abs().mul_(grad.sign())

                if step == 1:
                    grad.zero_()

                # Decouple momentum from direction if using sign_momentum parameter (highly recommended)
                if sign_mom_coeff != 0:
                    momentum = momentum.mul(beta).add_(grad.abs(), alpha=1. - beta)
                else:
                    momentum = momentum.mul(beta).add_(grad, alpha=1. - beta)

                # Update sign momentum
                if sign_mom_coeff != 0:
                    sign_momentum = sign_momentum.mul(sign_mom_coeff).add_(grad.sign(), alpha=1 - sign_mom_coeff)
                    c_t = grad.abs().lerp(momentum, weight=beta) # Nesterov-like momentum
                else:
                    c_t = grad.lerp(momentum, weight=beta) # Nesterov-like momentum

                # Spectral Clipping / Newton Schulz iters or RMS normalization
                if dimcount >= 2 and do_spectral_clip:
                    if dimcount > 2:
                        c_t_2d = c_t.reshape(len(c_t), -1) # Make 2D if conv or 1 dim
                    else:
                        c_t_2d = c_t

                    flip = c_t_2d.shape[0] > c_t_2d.shape[1]
                    if flip:
                        c_t_2d = c_t_2d.T # Flip if first dim is larger

                    full_step = self.clip_func(c_t_2d, sigma_min=spectral_min, sigma_max=spectral_max, adaptive=spectral_adaptive, ortho_dtype=spectral_clip_dtype)

                    if flip:
                        full_step = full_step.T

                    full_step = full_step.view_as(c_t).atan2(momentum.abs()).mul_(1.27323954474)
                else:
                    # Utilize momentum as denom with atan2
                    full_step = c_t.atan2(momentum.abs()).mul_(1.27323954474)

                # Apply sign if using sign_momentum
                if sign_mom_coeff != 0:
                    full_step = full_step.mul(sign_momentum)

                # Perform weight decay
                if weight_decay != 0:
                    full_step = full_step.add(p_fp32, alpha=wd_scaled)

                p_fp32.add_(full_step, alpha=-lr)

                if use_fp32:
                    copy_stochastic_(state["momentum"], momentum)
                    if sign_mom_coeff != 0:
                        copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["momentum"].copy_(momentum)
                    if sign_mom_coeff != 0:
                        state["sign_momentum"].copy_(sign_momentum)
                    p.copy_(p_fp32)
