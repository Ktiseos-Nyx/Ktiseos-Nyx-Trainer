# ABMOG from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math
import collections

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
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        M = M / (torch.linalg.norm(M).clamp_min_(1e-8))
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T.contiguous()
    if adaptive:
        M = torch.einsum('ij,ij,ab->ab', M_orig.type_as(M), M, M)
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

def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    
    # 2. Create a radial low-pass filter
    freq_dims = [torch.fft.fftfreq(s, device=grad.device) for s in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius ** 2))
    
    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights
    
    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    
    return modified_grad.real

def create_gaussian_mask(shape, sigma=1.0, device='cpu'):
    freq_dims = [torch.fft.fftfreq(s, device=device) for s in shape]
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-sigma * (radius ** 2))
    return filter_weights

def similarity_fft(grad, prev_grad, sigma=0.0):
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

def reshape_to_2d(grad):
    dimcount = len(grad.shape)
    if dimcount > 2:
        grad_2d = grad.reshape(len(grad), -1)
    elif dimcount < 2:
        grad_2d = grad.reshape(1, -1)
    else:
        grad_2d = grad
    return grad_2d

class ABMOG(Optimizer):
    r"""
    ABMOG: Adams-Bashforth-Moulton Orthogonal Gradient

    A Muon-styled optimizer which incorporates an Adams-Bashforth predictor and Adams-Moulton corrector
    step to refine the gradient based on its history, accelerating convergence. Now includes bonus goodies (bcos, cautious, dual-norm gradient)

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the Nesterov-styled momentum, the long-term squared mean running average, and the running average grad norm for the adaptive learning-rate ratio (default: 0.95, 0.99, 0.999).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step - Visualization: https://www.desmos.com/calculator/ipgbjovebr - (default: 0.995).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase) (default: True).
        spectral_clip_dtype (torch.dtype in string format):
            Sets the dtype of spectral clipping calculation. Recommended to use torch.float32 (or leave at default of None) (default: None, which results in torch.float32).
        adaptive (bool):
            Scale the full step to the momentumized average gradient (default: True).
        adaptive_min (float):
            Minimum multiplier for the adaptive scale (default: -1.0).
        adaptive_max (float):
            Maximum multiplier for the adaptive scale (default: 1.0).
        input_norm (bool):
            Normalizes with RMS on the input feature dimensions instead of utilizing gradient-wise RMS normalization (default: True).
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        bcos (bool):
            Uses a conditional estimator from facebookresearch's bcos as the denominator - https://github.com/facebookresearch/bcos (default: True).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        sgd_nesterov (bool):
            Utilizes SGD-like Nesterov momentum instead of current-gradient-focused momentum (default: True).
        abm_order (int):
            Order of the Adams-Bashforth-Moulton method. Uses abm_order gradients. Set abm_order to 1 to disable ABM extrapolation. (default: 4).
        abm_k (int):
            Do an Adams-Bashforth-Moulton extrapolation every abm_k steps. (default: 5).
        abm_cpu_storage (bool):
            Store ABM gradient history on CPU to save VRAM. (default: True).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: float = (0.95, 0.99),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None,
        adaptive: bool = True,
        adaptive_min: float = -1.,
        adaptive_max: float = 1.,
        input_norm: bool = True,
        lowpass_grad: float = 0.0,
        bcos: bool = True,
        cautious_min: float = 0.0,
        sgd_nesterov: bool = True,
        abm_order: int = 4,
        abm_k: int = 5,
        abm_cpu_storage: bool = True,
        stochastic_fp: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
    ):

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

        self._init_lr = lr

        self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32

        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split('.')[-1]
            spectral_clip_dtype = getattr(torch, dtype_name)
        
        # Coefficients for Adams-Bashforth (Predictor)
        # k=1 to 9. History is [g_n, g_{n-1}, ...]
        self.ab_coeffs = {
            1: [1.0],
            2: [1.5, -0.5],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24],
            5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
            6: [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440],
            7: [198721/60480, -447288/60480, 705549/60480, -688256/60480, 407139/60480, -134472/60480, 19087/60480],
            8: [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960],
            9: [14097241/3628800, -43448842/3628800, 98223681/3628800, -145788142/3628800, 143531169/3628800, -92956942/3628800, 38162241/3628800, -9124282/3628800, 959281/3628800],
            10: [29579241/7257600, -104829331/7257600, 276985582/7257600, -491429182/7257600, 608822461/7257600, -520448951/7257600, 296222582/7257600, -107198731/7257600, 22254361/7257600, -2043851/7257600],
        }
        
        # Coefficients for Adams-Moulton (Corrector)
        # k=1 to 9. History is [g_{n+1}_pred, g_n, g_{n-1}, ...]
        self.am_coeffs = {
            1: [1.0], # Using predicted gradient only as corrector
            2: [0.5, 0.5],
            3: [5/12, 8/12, -1/12],
            4: [9/24, 19/24, -5/24, 1/24],
            5: [251/720, 646/720, -264/720, 106/720, -19/720],
            6: [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440],
            7: [19087/60480, 65112/60480, -46461/60480, 37504/60480, -20211/60480, 6312/60480, -863/60480],
            8: [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960],
            9: [2043851/7257600, 8648118/7257600, -8526441/7257600, 10049438/7257600, -8507853/7257600, 4899438/7257600, -1818321/7257600, 392958/7257600, -37867/7257600],
            10: [37867/1451520, 196967/1451520, -214753/1451520, 290177/1451520, -289063/1451520, 199367/1451520, -89533/1451520, 24047/1451520, -2953/1451520],
        }

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            spectral_adaptive = spectral_adaptive,
            spectral_clip_compile = spectral_clip_compile,
            spectral_clip_dtype = spectral_clip_dtype,
            adaptive = adaptive,
            adaptive_min = adaptive_min,
            adaptive_max = adaptive_max,
            input_norm = input_norm,
            lowpass_grad = lowpass_grad,
            bcos = bcos,
            cautious_min = cautious_min,
            sgd_nesterov = sgd_nesterov,
            stochastic_fp = stochastic_fp,
            abm_order = abm_order,
            abm_k = abm_k,
            abm_cpu_storage = abm_cpu_storage,
            sync_chunk_size = sync_chunk_size,
            state_storage_dtype = final_dtype,
            state_storage_device = state_storage_device,
        )

        super(ABMOG, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        pass

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
            beta, beta2 = group["betas"][0], group["betas"][1]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            abm_order, abm_k = group["abm_order"], group["abm_k"]
            abm_cpu_storage = group["abm_cpu_storage"]

            step = group['step']

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                device = p.device

                grad = p.grad.data
                
                # State initialization
                if len(state) == 0:
                    if self.state_storage_device == "cpu":
                        if not group["bcos"]:
                            state["denom"] = torch.tensor(1.0, 
                                                          dtype=self.state_storage_dtype, 
                                                          device=self.state_storage_device).pin_memory()

                        state["value_momentum"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device
                        ).pin_memory()
                    else:
                        if not group["bcos"]:
                            state["denom"] = torch.tensor(1.0, 
                                                          dtype=self.state_storage_dtype, 
                                                          device=self.state_storage_device)

                        state["value_momentum"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device
                        )
                    if abm_order > 1:
                        # Use a deque to efficiently manage fixed-size history
                        state["p_history"] = collections.deque(maxlen=abm_order)

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                dimcount = grad.ndim

                # 1. Queue Host-to-Device copy
                if not group["bcos"]:
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
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                # Fast-to-slow beta (0 @ step 1, 0.5 @ step 2, 0.6667... @ step 3, repeating to a max of beta2)
                slow_beta2 = ((beta2**(step) - beta2) / (beta2**(step) - 1.0))

                # ADOPT-style clamp to prevent overshooting at the beginning
                grad = grad.clamp(-step, step)

                # Optional low-passing of gradient
                if dimcount > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                # Normalize the gradient per-channel (input_norm=True + dim > 0) or per-tensor
                if dimcount >= 1 and group["input_norm"]:
                    grad_2d = reshape_to_2d(grad)
                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                # SGD-Like Nesterov or Adam-like Nesterov
                if group["sgd_nesterov"]:
                    value_momentum = value_momentum.mul(beta).add_(grad)
                    exp_avg = value_momentum.mul(beta).add_(grad).mul(1. - beta)
                else:
                    value_momentum = value_momentum.lerp(grad, weight=1. - beta)
                    exp_avg = grad.lerp(value_momentum, weight=beta)

                # Get denom if not using bcos
                if not group["bcos"]:
                    current_denom = denom.sqrt()

                # Muon-styled spectral norming, with scalar denominator
                if dimcount >= 1:
                    exp_avg_2d = reshape_to_2d(exp_avg)

                    flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
                    if flip:
                        exp_avg_2d = exp_avg_2d.T

                    exp_avg_2d = self.clip_func(exp_avg_2d, sigma_min=0., sigma_max=0., adaptive=group["spectral_adaptive"], ortho_dtype=group["spectral_clip_dtype"])

                    if flip:
                        exp_avg_2d = exp_avg_2d.T
                    full_step = exp_avg_2d.view_as(exp_avg)
                    if not group["bcos"]:
                        denom = denom.lerp(full_step.pow(2).mean(), weight=1. - slow_beta2)
                    else:
                        current_denom = ((3 * beta**2 - 2 * beta**3) * full_step.square() + (1 - beta)**2 * grad.detach().square() + 2 * beta * (1-beta)**2 * full_step * grad.detach()).mean().sqrt()

                    full_step = full_step.div(current_denom.clamp_min(1.0))
                else:
                    if not group["bcos"]:
                        denom = denom.lerp(exp_avg.pow(2), weight=1. - slow_beta2)
                    else:
                        current_denom = ((3 * beta**2 - 2 * beta**3) * exp_avg.square() + (1 - beta)**2 * grad.detach().square() + 2 * beta * (1-beta)**2 * exp_avg * grad.detach()).mean().sqrt()
                    full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

                scale_factor_mask = torch.where(grad * full_step > 0, torch.ones_like(full_step), torch.ones_like(full_step) * group["cautious_min"]).to(full_step.dtype)
                scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))

                # Cautious masking
                full_step = full_step.mul(scale_factor_mask)

                # Dual-norm gradient
                if group["adaptive"]:
                    if dimcount >= 1 and group["input_norm"]:
                        if dimcount > 2:
                            full_step_2d = full_step.reshape(len(full_step), -1)
                            exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
                        elif dimcount < 2:
                            full_step_2d = full_step.reshape(1, -1)
                            exp_avg_2d = exp_avg.reshape(1, -1)
                        else:
                            full_step_2d = full_step
                            exp_avg_2d = exp_avg
                        scale_factor = (exp_avg_2d * full_step_2d).sum(dim=1, keepdim=True).clamp(group["adaptive_min"], group["adaptive_max"])
                        full_step = (full_step_2d * scale_factor).view_as(full_step)
                    else:
                        scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])
                        full_step = scale_factor * full_step

                if weight_decay != 0:
                    p_fp32.data = p_fp32.mul(1 - lr * weight_decay*weight_decay_rate**group["step"])

                # Add step
                p_fp32.data.add_(full_step, alpha=-lr)

                if abm_order > 1 and step % abm_k == 0:
                    # Store history on CPU
                    storage_device = 'cpu' if abm_cpu_storage else p_fp32.data.device

                    # Add current grad to history (left side is newest)
                    state["p_history"].appendleft(p_fp32.data.detach().to(storage_device))

                    history = list(state["p_history"])
                    current_k = len(history)

                    # Wait for history buffer to fill at least once
                    if current_k > 1:
                        # Bring history to calculation device
                        history_gpu = [g for g in history]

                        # Predictor (Adams-Bashforth)
                        ab_c = self.ab_coeffs[current_k]
                        p_pred = torch.zeros_like(state["p_history"][0])
                        for i in range(current_k):
                            p_pred.add_(history_gpu[i], alpha=ab_c[i])

                        # Corrector (Adams-Moulton)
                        am_c = self.am_coeffs[current_k]
                        # Use predicted grad as proxy for g_{n+1}
                        corrector_hist = [p_pred] + history_gpu[:-1]

                        p_corrected = torch.zeros_like(state["p_history"][0])
                        for i in range(current_k):
                            p_corrected.add_(corrector_hist[i], alpha=am_c[i])

                        p_fp32.data.copy_(p_corrected)

                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    # Original GPU path
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)
                if self.state_storage_dtype == torch.bfloat16:
                    if not group["bcos"]:
                        copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                else:
                    if not group["bcos"]:
                        state["denom"].copy_(denom, non_blocking=True)
                    state["value_momentum"].copy_(value_momentum, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss