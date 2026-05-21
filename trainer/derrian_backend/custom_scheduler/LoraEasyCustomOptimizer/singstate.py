# SingState from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math
import logging

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
    for a, b, c in NS_COEFFS[:num_ns_steps]:
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
def block_matmul(
    P1: torch.Tensor, Q1: torch.Tensor, R1: torch.Tensor,
    P2: torch.Tensor, Q2: torch.Tensor, R2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs block matrix multiplication elements of the (linear) sub-algebra
    of matrices of the form:
        [P   Q]
        [Q.T R]
    where Q is a MxN matrix, and P and R are symmetric matrices of size MxM and NxN respectively.
    """
    P = P1 @ P2   + Q1 @ Q2.T
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R

@torch.no_grad()
def newton_schulz_iter(
    P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    a: float, b: float, c: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """5th order blockwise Newton-Schulz iteration for orthogonalization."""
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    I_P = a * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    I_R = a * torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)

@torch.no_grad()
def orthogonalize_blockwise(
    W: torch.Tensor, ortho_dtype=torch.float32, num_ns_steps: int=len(NS_COEFFS)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonalize a matrix via 5th order blockwise Newton-Schulz iteration.

    Tighter spectral norm bound:
    => Matrices of the form [I_m, W; W.T, I_n] have spectral norm 1 + ||W||_2
    => We can estimate ||W||_2 via power iteration or Gram iteration.
    => However, we can also use the fact that ||W||_2 <= ||W||_F and the latter is much cheaper to compute.

    yeah this is 'translated' from jax to python via gemini
    in the name of PS: 'you can eat my entire ass' or something
    """
    orig_dtype = W.dtype
    m, n = W.shape
    I_m, I_n = torch.eye(m, device=W.device), torch.eye(n, device=W.device)
    # norm = 1 + _power_iterate(W, torch.manual_seed(0), num_iters=16)[1]
    norm = 1 + torch.linalg.norm(W)
    P = (I_m / (norm + 1e-12)).to(ortho_dtype)
    Q = (W   / (norm + 1e-12)).to(ortho_dtype)
    R = (I_n / (norm + 1e-12)).to(ortho_dtype)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        P, Q, R = newton_schulz_iter(P, Q, R, a=a, b=b, c=c)
    return P.to(orig_dtype), Q.to(orig_dtype), R.to(orig_dtype)

def _spectral_hardcap_blockwise(W: torch.Tensor, sigma_max=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    def _spectral_hardcap_blockwise_util(W: torch.Tensor):
        if adaptive:
            W_orig = W.clone()
        transpose = W.shape[0] > W.shape[1]
        if transpose:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        # _, Q, R = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        # result = Q + W @ R
        P, Q, _ = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        result = Q + P @ W
        if transpose:
            result = result.T
        if adaptive:
            result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
        return result.to(orig_dtype)
    return sigma_max * _spectral_hardcap_blockwise_util(W / sigma_max)

def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if adaptive:
        W_orig = W.clone()
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = orthogonalize(W, num_ns_steps)
    eye_m = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    result = (1/2) * (
        (sigma_min + sigma_max) * eye_m
        + (sigma_min * OW - W) @ orthogonalize(sigma_min * OW - W, num_ns_steps).T
        - (sigma_max * OW - W) @ orthogonalize(sigma_max * OW - W, num_ns_steps).T
    ) @ OW
    if adaptive:
        result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
    return result.to(orig_dtype)

@torch.no_grad()
def batch_project(M: torch.Tensor, project_fn: Callable) -> torch.Tensor:
    """Batch project tensors of shape [..., fanout, fanin] using vmap."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape(-1, *matrix_shape)

    M_projected = torch.vmap(project_fn)(M_flattened)

    return M_projected.reshape(M.shape) / len(M_flattened)

@torch.no_grad()
def spectral_clip_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def spectral_clip_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

@torch.no_grad()
def spectral_hardcap_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_hardcap_blockwise(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def spectral_hardcap_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_hardcap_blockwise(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch.no_grad()
def orthogonalize_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def orthogonalize_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

@torch.no_grad()
def separate_frequencies(
    grad: torch.Tensor, 
    cutoff_freq_ratio: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Separates a gradient tensor into low-frequency and high-frequency components
    using the Fast Fourier Transform (FFT).

    Args:
        grad (torch.Tensor): The input gradient tensor. Can be of any shape.
        cutoff_freq_ratio (float): A value between 0.0 and 1.0. It defines the
            radius of the low-pass filter in the frequency domain, as a ratio
            of the smallest dimension size. For example, a value of 0.1 means
            frequencies within a radius of 10% of the smallest dimension size
            are considered "low frequency".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - low_freq_component (torch.Tensor): The low-frequency part of the gradient.
            - high_freq_component (torch.Tensor): The high-frequency part of the gradient.
    """
    if not 0.0 <= cutoff_freq_ratio <= 1.0:
        raise ValueError("cutoff_freq_ratio must be between 0.0 and 1.0")

    if cutoff_freq_ratio == 1.0:
        return grad.clone(), torch.zeros_like(grad)
    if cutoff_freq_ratio == 0.0:
        return torch.zeros_like(grad), grad.clone()

    # 1. Perform n-dimensional FFT
    grad_fft = torch.fft.fftn(grad)

    # 2. Shift the zero-frequency component to the center for easier masking
    grad_fft_shifted = torch.fft.fftshift(grad_fft)

    # 3. Create a low-pass filter mask
    shape = grad.shape
    # The center of the n-dimensional FFT grid
    center_indices = [s // 2 for s in shape]
    # The radius for the low-pass filter cutoff
    # We use the smallest dimension to define the relative cutoff
    min_dim_size = min(shape)
    cutoff_radius = int(min_dim_size * cutoff_freq_ratio / 2)

    # Create coordinate grids for each dimension
    grid_coords = torch.meshgrid(
        *[torch.arange(s, device=grad.device) for s in shape], 
        indexing='ij'
    )
    
    # Calculate Euclidean distance from the center for each point in the grid
    dist_from_center_sq = torch.zeros_like(grad, dtype=torch.float32)
    for i, center_idx in enumerate(center_indices):
        dist_from_center_sq += (grid_coords[i] - center_idx)**2

    # The mask is True for frequencies within the cutoff radius
    low_pass_mask = dist_from_center_sq <= cutoff_radius**2
    
    # 4. Apply the mask
    low_freq_fft_shifted = grad_fft_shifted * low_pass_mask

    # 5. Inverse shift to move the zero-frequency component back
    low_freq_fft = torch.fft.ifftshift(low_freq_fft_shifted)

    # 6. Perform inverse FFT to get the low-frequency component in the spatial domain
    # The result of ifftn will be complex, but since the input was real, the
    # imaginary part should be negligible. We take the real part.
    low_freq_component = torch.fft.ifftn(low_freq_fft).real

    # 7. The high-frequency component is simply the original gradient minus the low-freq part.
    # This is more numerically stable than performing a second inverse FFT.
    high_freq_component = grad - low_freq_component

    return low_freq_component, high_freq_component

@torch.no_grad()
def freq_sep_func(W: torch.Tensor, cutoff_freq_ratio=0.1):
    return separate_frequencies(W, cutoff_freq_ratio=cutoff_freq_ratio)

def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    
    # 2. Create a radial low-pass filter
    # Create a grid of frequency coordinates
    freq_dims = [torch.fft.fftfreq(s, device=grad.device) for s in grad.shape]
    # Center the grid for radial calculation
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    
    # Create a meshgrid of coordinates
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    
    # Calculate the radial distance (L2 norm) from the center (zero frequency)
    # Normalize by the max possible frequency radius for scale invariance
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    
    # Create a Gaussian low-pass filter.
    # Higher alpha means sharper decay, i.e., more aggressive filtering
    filter_weights = torch.exp(-fft_alpha * (radius ** 2))
    
    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights
    
    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    
    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

def sym(A):
    """
    Computes the symmetric part of a square matrix A.
    sym(A) = (A + A.T) / 2
    """
    return 0.5 * (A + A.T)

def project_to_stiefel_tangent_space(X, delta_X):
    """
    Projects a matrix delta_X onto the tangent space of the Stiefel manifold at point X.

    Args:
        X (torch.Tensor): A point on the Stiefel manifold, i.e., an n x p matrix
                          such that X.T @ X = I. Shape: (n, p).
        delta_X (torch.Tensor): A matrix in the ambient space (the "gradient"),
                                to be projected. Shape: (n, p).

    Returns:
        torch.Tensor: The projection of delta_X onto the tangent space at X.
                      Shape: (n, p).
    """
    # The core projection formula from the JAX pseudo-code
    # This is the "normal component" of the gradient that gets subtracted.
    # It ensures the result is in the tangent space.
    return delta_X - X @ sym(X.T @ delta_X)

def steepest_descent_stiefel_manifold_heuristic(W, G, num_steps=3):
    assert num_steps > 0, "Number of steps must be positive"
    A_star = G
    for _ in range(num_steps):
        A_star = project_to_stiefel_tangent_space(W, A_star)
        A_star = orthogonalize(A_star)
    return A_star

class SingState(Optimizer):
    r"""
    SingState: Temporal Adaptation via Level and Orientation Normalization. 
    
    Cuts through noise by decoupling the gradient's sign and magnitude into two different momentum states, with a denominator for adaptive learning.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the sign momentum, running average, and the long-term squared running average (default: 0.9, 0.99, 0.9999999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        denom_atan2 (bool):
            Divide the smooth gradient using .atan2 instead of .div for stability and scale-invariance, removes epsilon/eps - https://arxiv.org/abs/2407.05872 (default: True).
        invariant (bool):
            Scale the latent into -1 to 1 space via .arctan().sin(), then later divide by the original grad's .arctan().cos(). Its been tested a bit, with the general result of speeding up descent. (default: False).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: True).
                * Set spectral_min and spectral_max to 0 to enable generic Newton-Schulz orthogonalization.
                * Set spectral_min to any value below -1000.0 to enable block-wise "spectral hardcapping" mode. Likely to be slower in this mode, but more stable.
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase). (default: True).
        spectral_min (float):
            The minimum value of the spectral magnitude. Ought to be lower than spectral_max. (default: -1.0).
        spectral_max (float):
            The maximum value of the spectral magnitude. (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: False).
        lowpass_grad (bofloatol):
            Pre-condition the gradient with a lowpass filter via FFT (default: 1.0).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_clip: bool = False,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        spectral_min: float = -1.,
        spectral_max: float = 1.,
        spectral_adaptive: bool = False,
        lowpass_grad: float = 1.0,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Optimizer argument '{key}' passed into SingState. It will be ignored."
            )

        self._init_lr = lr

        if spectral_clip:
            if spectral_min == 0 and spectral_max == 0:
                self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func
            else:
                self.clip_func = spectral_clip_compiled_func if spectral_clip_compile else spectral_clip_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32

        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split('.')[-1] # Gets "float16"
            spectral_clip_dtype = getattr(torch, dtype_name)

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
            stochastic_fp = stochastic_fp,
        )

        super(SingState, self).__init__(params, defaults)

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
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']

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

                # Detach
                p_fp32 = p.detach().clone()
                momentum = state["momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    momentum = state['momentum'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                if dimcount > 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                #rms = grad.pow(2).mean().sqrt_().clamp_min_(1.0)
                grad = grad.clamp(-step, step)

                denom = momentum.abs()

                momentum = momentum.lerp(grad.sign(), weight=1. - beta)#.abs_().lerp_(grad.sign(), weight=1. - beta)

                c_t = grad.abs().lerp(momentum.abs(), weight=beta)

                # Spectral Clipping / Newton Schulz iters or RMS normalization
                if dimcount >= 2 and group["spectral_clip"]:
                    if dimcount > 2:
                        c_t_2d = c_t.reshape(len(c_t), -1) # Make 2D if conv or 1 dim
                    else:
                        c_t_2d = c_t

                    flip = c_t_2d.shape[0] > c_t_2d.shape[1]
                    if flip:
                        c_t_2d = c_t_2d.T # Flip if first dim is larger

                    c_t_2d = self.clip_func(c_t_2d, sigma_min=group["spectral_min"], sigma_max=group["spectral_max"], adaptive=group["spectral_adaptive"], ortho_dtype=group["spectral_clip_dtype"])

                    if flip:
                        c_t_2d = c_t_2d.T

                    full_step = c_t_2d.view_as(c_t).atan2(denom).mul_(1.27323954474)
                else:
                    # Utilize momentum as denom with atan2
                    full_step = c_t.atan2(denom).mul_(1.27323954474)

                #rms = momentum.pow(2).mean().sqrt_().clamp_min_(1.0)
                nesterov_direction = grad.sign().lerp_(momentum, weight=beta)
                full_step = full_step.mul(nesterov_direction)

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])
                #print(full_step)
                p_fp32.data.add_(full_step, alpha=-lr)

                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["momentum"].copy_(momentum)
                    p.copy_(p_fp32)
        return loss