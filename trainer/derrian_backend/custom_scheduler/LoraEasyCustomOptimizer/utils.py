import torch
from typing import Tuple, Union, Type, Literal, Optional, Dict, Any, List
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import math
import inspect
import logging

OPTIMIZER = Type[Optimizer]

NORM_TYPE = Literal['unit','global','layer']

CLIP_TYPE = Literal['unit','layer','element']

STATE_PRECISION = Literal['parameter', 'q4bit', 'q8bit', 'qfp8']

UPDATE_STRATEGY = Literal['unmodified','cautious','grams','both']

def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)

def unit_norm_logging(x: torch.Tensor, norm: float = 2.0):
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    logging.info(f"unit_norm shape={str(x.shape)}")
    logging.info(f"unit_norm norms={str(torch.norm(x, p=norm, dim=dim, keepdim=keep_dim))}")

@torch.no_grad()
@torch.compiler.disable()
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor, scratch: Optional[torch.Tensor] = None):
    r"""Copy source to target with stochastic rounding for reduced-precision targets.

    :param target: torch.Tensor. destination tensor (e.g. bfloat16 on CPU).
    :param source: torch.Tensor. source tensor (float32 on compute device).
    :param scratch: Optional[torch.Tensor]. pre-allocated int32 scratch buffer matching source shape/device.
        When provided, avoids a per-call ``torch.randint_like`` allocation.  Pass ``None`` to fall back
        to the default allocation path.  The scratch buffer is modified in-place and must have the same
        shape and device as ``source``.
    """
    # Determine the intermediate FP32 tensor
    if source.dtype == torch.float64:
        src_fp32 = source.to(dtype=torch.float32)
    elif source.dtype == torch.float32:
        src_fp32 = source
    else:
        target.copy_(source.to(dtype=target.dtype))
        return

    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    # create a random 16 bit integer (reuse pre-allocated scratch when available)
    if scratch is not None:
        result = scratch
        result.random_(0, 1 << 16)
    else:
        result = torch.randint_like(
            src_fp32,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(src_fp32.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32), non_blocking=True)
    
def agc(p: torch.Tensor, 
        grad: torch.Tensor, 
        agc_clip_val: float, 
        agc_eps: float = 1e-3, 
        eps: float = 1e-16, 
        norm_type: NORM_TYPE = 'layer') -> torch.Tensor:
    r"""Clip gradient values in excess of the norm.
        Clip updates to be at most clipping * parameter_norm.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.
        
    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. Effectively sets a floor for the p_norm, as such, any gradients smaller than this will be clipped
        as though their parameter is at least agc_eps. This helps prevent vanishing gradients and excessive clipping early in training
        for small parameters.
    :param agc_clip_val: float. The desired clipping ratio, e.x. 0.5 would mean any gradient would be clipped to be no greater than half it's
        associated parameter.
    :param eps: float. simple stop from div by zero, as such should be as small as possible to avoid skewing clipping.
    """
    if norm_type in {'global','layer'}:
        # Compute the global norm of the parameters and gradients
        p_norm = torch.norm(p).clamp_(min=agc_eps)
        g_norm = torch.norm(grad)

        # Compute the maximum allowed norm for the gradients
        max_norm = (p_norm * agc_clip_val).clamp(min=eps)

        # Compute the clipping coefficient
        clip_coef = min(1, max_norm / g_norm.clamp(min=eps))

        # Scale the gradients holistically
        grad = grad * clip_coef

        return grad
    elif norm_type == 'unit':
        p_norm = unit_norm(p).clamp_(min=agc_eps)
        g_norm = unit_norm(grad)

        max_norm = (p_norm * agc_clip_val).clamp(min=eps)

        clipped_grad = grad * (max_norm / g_norm.clamp_(min=eps))

        return torch.where(g_norm > max_norm, clipped_grad, grad)
    else:
        raise ValueError(f"'{norm_type}' is not a supported value for norm_type.")


def schedule_alpha(t_alpha: Optional[float], step: int, alpha: float) -> float:
    if t_alpha is None:
        return alpha
    return min(step * alpha / t_alpha, alpha)


def schedule_beta(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        math.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

def schedule_beta_tc(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        torch.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

@torch.no_grad()
def spam_grad_clipping(grad: torch.Tensor, 
                       second_moment: torch.Tensor, 
                       clip_threshold: float, 
                       clip_type: CLIP_TYPE = 'element', 
                       spam_clip_eps: float = 1e-16) -> torch.Tensor:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny
    
    if clip_type in {'unit', 'element'}:
        # Calculate the clipping condition
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        sign_grad = grad.sign()

        # Use torch.where instead of boolean masking
        return torch.where(
            grad.square() > second_momentum_threshold,
            sign_grad * second_momentum_threshold_sqrt,
            grad
        )
    elif clip_type == 'layer':
        # Calculate the global gradient norm
        max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
        grad_norm = torch.norm(grad)

        # Calculate scaling factor for clipping
        scale = torch.where(
            grad_norm > max_norm,
            max_norm / grad_norm,
            torch.ones_like(grad_norm)
        )

        # Apply scaling to gradient
        return grad * scale
    
def spam_grad_clipping_logging(grad: torch.Tensor, 
                               second_moment: torch.Tensor, 
                               clip_threshold: float, 
                               clip_type: str = 'element', 
                               spam_clip_eps: float = 1e-16) -> torch.Tensor:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny

    if clip_type in {'unit', 'element'}:
        # Calculate the clipping condition
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        
        # Check where scaling will occur
        scaling_mask = grad.square() > second_momentum_threshold
        total_elements = grad.numel()
        
        if scaling_mask.any():
            # Calculate scaling ratios for logging
            original_values = grad[scaling_mask].abs()  # Use absolute values
            scaled_values = second_momentum_threshold_sqrt[scaling_mask]
            
            # Add small epsilon to prevent division by zero
            scaling_ratios = scaled_values / (original_values.add(spam_clip_eps))
            
            # Add more detailed logging
            logging.info(
                f"Total elements {total_elements}. "
                f"Unit-wise gradient clipping applied to {scaling_mask.sum().item()} elements. "
                f"\nOriginal values - Mean: {original_values.mean().item():.6f}, Max: {original_values.max().item():.6f}. "
                f"\nScaled values - Mean: {scaled_values.mean().item():.6f}, Max: {scaled_values.max().item():.6f}. "
                f"\nScaling ratios - Mean: {scaling_ratios.mean().item():.6f}, Max: {scaling_ratios.max().item():.6f}"
            )
        
    elif clip_type == 'layer':
        # Calculate the global gradient norm
        max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
        grad_norm = torch.norm(grad)
        
        # Calculate scaling factor
        scale = torch.where(
            grad_norm > max_norm,
            max_norm / grad_norm,
            torch.ones_like(grad_norm)
        )
        
        # Log if scaling is applied
        if grad_norm > max_norm:
            logging.info(
                f"Layer-wise gradient clipping applied. "
                f"Gradient norm: {grad_norm.item():.4f}, "
                f"Max norm: {max_norm.item():.4f}, "
                f"Scaling factor: {scale.item():.4f}"
            )
    

# Modified Adafactor factorisation implementation by Ross Wightman 
# https://github.com/huggingface/pytorch-image-models/pull/2320
@torch.no_grad()
def create_factored_dims(
    shape,
    factored: bool,
    min_dim_size_to_factor: int):
    r"""Whether to use a factored second moment estimator.
    This function returns a tuple with the two largest axes to reduce over.
    If all dimensions have size < min_dim_size_to_factor, return None.
    Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for > 2d vars.
    min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
    Returns:
    None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    if all(dim < min_dim_size_to_factor for dim in shape):
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L389
@torch.no_grad()
def get_denom(second_moment: torch.Tensor, eps: float = 1e-16):
    # Get denom
    if isinstance(second_moment, list):
        row_var, col_var, _, _, reduce_dc = second_moment

        row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(eps)
        row_factor = row_var.div(row_col_mean).sqrt_()
        col_factor = col_var.sqrt()
        denom = row_factor * col_factor
    else:
        denom = second_moment.sqrt()

    return denom
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L411
@torch.no_grad()
def update_second_moment(second_moment: torch.Tensor, grad: torch.Tensor, beta2: float, adopt_first: bool = False) -> torch.Tensor:
    # EMA updates
    if isinstance(second_moment, list):
        row_var, col_var, dr, dc, _ = second_moment
        if adopt_first:
            row_var.copy_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr])
            )
            col_var.copy_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc])
            )
        else:
            row_var.lerp_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]),
                weight=1 - beta2
            )
            col_var.lerp_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]),
                weight=1 - beta2
            )
    else:
        if adopt_first:
            second_moment.addcmul_(grad, grad)
        else:
            second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    return second_moment

# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
@torch.no_grad()
def orthograd_atan(param: torch.Tensor, grad: torch.Tensor):
    grad_shape = grad.shape
    w = param.view(-1)
    grad = grad.view(-1)

    proj = torch.dot(w, grad).atan2_(torch.dot(w, w)).mul_(1.27323954474)
    g_orth = grad.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
    g_orth_scaled = g_orth.mul_(grad.norm(2).div_(g_orth.norm(2).clamp_(min=1e-6)))

    return g_orth_scaled.view(grad_shape)

def clean_dict_params(func, params_dict, wrapped=False):
    """
    Remove dictionary keys that don't match function parameters and warn about removals.
    
    Args:
        func: The function to check parameters against
        params_dict: Dictionary of parameters to clean
        
    Returns:
        dict: New dictionary with only valid parameters
    """
    # Get the function's signature
    sig = inspect.signature(func)
    
    # Create a new dict with only valid parameters
    valid_params = {}
    
    for key, value in params_dict.items():
        if key in sig.parameters:
            valid_params[key] = value
        else:
            print(f"Parameter '{key}' is not a valid parameter for the {'wrapped ' if wrapped else ''}optimizer and will be ignored.")
    
    return valid_params

class CosineDecay:
    """
    Applies cosine decay to a parameter (death_rate), using PyTorch's built-in
    `torch.optim.lr_scheduler.CosineAnnealingLR`.

    Args:
        death_rate (float): Initial value to be decayed.
        T_max (int): Maximum number of iterations for the decay.
        eta_min (float, optional): Minimum value of the parameter after decay.
            Defaults to 0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.sgd = torch.optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        """
        Performs one step of the cosine decay scheduler.

        Args:
            current_step (int): Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        """
        Returns the updated rate (death_rate) at the given step.

        Args:
            current_step (int): Current step index.

        Returns:
            float: The decayed parameter.
        """
        if current_step >= self.T_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]
    
class SSCCosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using PyTorch's built-in `CosineAnnealingLR`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    :param last_epoch: Optional[int]. the index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        r"""One step of the cosine decay scheduler.

        :param current_step: int. Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_death_rate(self, current_step: int) -> float:
        r"""Get the updated rate (death_rate) at the given step.

        :param current_step: int. Current step index.
        """
        if current_step >= self.t_max:
            return self.eta_min

        self.step(current_step)

        return self.sgd.param_groups[0]['lr']
    
def stable_spam_clipping(state: dict, 
                         grad: torch.Tensor, 
                         step: int|torch.Tensor, 
                         scale: float|torch.Tensor = 1.0, 
                         eps: float|torch.Tensor = 1e-8, 
                         gamma1: float|torch.Tensor = 0.85, 
                         gamma2: float|torch.Tensor = 0.99999, 
                         gamma3: float|torch.Tensor = 0.999,
                         torch_compile: bool = False) -> torch.Tensor:    
    
    if torch_compile:
        return _get_compiled_stable_spam_clipping()(state,
                            grad,
                            step,
                            scale,
                            eps,
                            gamma1,
                            gamma2,
                            gamma3)
    else:
        return _stable_spam_clipping_impl(state,
                            grad, 
                            step, 
                            scale, 
                            eps, 
                            gamma1, 
                            gamma2, 
                            gamma3)

def _stable_spam_clipping_compile_wrapper(state: dict,
                         grad: torch.Tensor,
                         step: int|torch.Tensor,
                         scale: float|torch.Tensor = 1.0,
                         eps: float|torch.Tensor = 1e-8,
                         gamma1: float|torch.Tensor = 0.85,
                         gamma2: float|torch.Tensor = 0.99999,
                         gamma3: float|torch.Tensor = 0.999) -> torch.Tensor:
    return _stable_spam_clipping_impl(state,
                         grad,
                         step,
                         scale,
                         eps,
                         gamma1,
                         gamma2,
                         gamma3)

# Module-level cache for lazily compiled functions (avoids looping compile)
_compiled_fns: dict = {}

def _get_compiled_stable_spam_clipping():
    r"""Lazily compile _stable_spam_clipping_compile_wrapper using torch.compile function call."""
    if 'stable_spam_clipping' not in _compiled_fns:
        with torch._dynamo.utils.disable_cache_limit():
            _compiled_fns['stable_spam_clipping'] = torch.compile(
                _stable_spam_clipping_compile_wrapper, fullgraph=True, dynamic=False
            )
    return _compiled_fns['stable_spam_clipping']

@torch.no_grad()
def _stable_spam_clipping_impl(state: dict, 
                         grad: torch.Tensor, 
                         step: int|torch.Tensor, 
                         scale: float|torch.Tensor = 1.0, 
                         eps: float|torch.Tensor = 1e-8, 
                         gamma1: float|torch.Tensor = 0.85, 
                         gamma2: float|torch.Tensor = 0.99999, 
                         gamma3: float|torch.Tensor = 0.999) -> torch.Tensor:   
    if 'ssc_m_norm_t' not in state:
        state['ssc_m_norm_t'] = 0.0
        state['ssc_v_norm_t'] = 0.0
        state['ssc_m_max_t'] = 0.0

    max_grad = torch.max(grad.abs())

    m_max_t = state['ssc_m_max_t']

    m_max_t = gamma3 * m_max_t + (1 - gamma3) * max_grad

    state["ssc_m_max_t"] = m_max_t

    m_max_hat = m_max_t / (1.0 - gamma3 ** step)

    grad = torch.where(grad.abs() > m_max_hat,
                        grad / max_grad * m_max_hat,
                        grad)

    grad_norm = torch.norm(grad)

    m_norm_t, v_norm_t = state['ssc_m_norm_t'], state['ssc_v_norm_t']

    m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
    v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm**2

    m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
    v_norm_hat = v_norm_t / (1.0 - gamma2 ** step)

    state["ssc_m_norm_t"], state["ssc_v_norm_t"] = m_norm_t, v_norm_t

    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

    grad = torch.where(grad_norm > 0,
                        grad / grad_norm * c_norm_t,
                        grad)

    return grad

@torch.no_grad()
def stable_spam_clipping_tensors(
    ssc_m_norm_t: torch.Tensor,
    ssc_v_norm_t: torch.Tensor,
    ssc_m_max_t: torch.Tensor, 
    grad: torch.Tensor, 
    step: int, 
    scale: float = 1.0, 
    eps: float = 1e-8, 
    gamma1: float = 0.85, 
    gamma2: float = 0.99999, 
    gamma3: float = 0.999):    

        m_max_t = ssc_m_norm_t

        max_grad = torch.max(grad.abs())

        m_max_t = gamma3 * m_max_t + (1 - gamma3) * max_grad

        ssc_m_norm_t.copy_(m_max_t)

        m_max_hat = m_max_t / (1.0 - gamma3 ** step)

        grad = torch.where(grad.abs() > m_max_hat,
                           grad / max_grad * m_max_hat,
                           grad)

        grad_norm = torch.norm(grad)

        m_norm_t, v_norm_t = ssc_m_norm_t, ssc_m_max_t

        m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
        v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm**2

        ssc_m_norm_t.copy_(m_norm_t)
        ssc_v_norm_t.copy_(v_norm_t)

        m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
        v_norm_hat = v_norm_t / (1.0 - gamma2 ** step)

        c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

        grad = torch.where(grad_norm > 0,
                           grad / grad_norm * c_norm_t,
                           grad)

        return grad

# From: https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.no_grad()
def newton_schulz_(grad, steps=6, eps=1e-12):
    # Inline reshaping step within the method itself.
    G_shape = grad.shape
    grad = grad.view(grad.size(0), -1)

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    X = grad.to(dtype=torch.bfloat16, copy=True)
    if grad.size(0) > grad.size(1):
        X = X.T.contiguous()

    X /= X.norm().add(eps) # ensure top singular value <= 1
    for a,b,c in abc_list:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T.contiguous()

    # Gradient scaling adaptation from: https://github.com/leloykun/adaptive-muon
    X = torch.einsum('ij,ij->', grad.type_as(X), X).clamp(-1.0, 1.0) * X
    grad.copy_(X)
    del X

    return grad.view(G_shape)

@torch.no_grad()
def adagc_global_clipping_calc(
        self,
        step: int,
        warmup_steps: int = 0,
        lambda_abs: float = 1.0,
        eps: float = 1e-8
) -> torch.Tensor:
        # --- Global Clipping Calculation (outside compiled step) ---
        global_norm_device = 'cpu'
        has_grad = False
        for group in self.param_groups:
             for p in group['params']:
                 if p.grad is not None:
                    has_grad = True
                    global_norm_device = p.grad.device # Use device of first grad found
                    break
             if global_norm_device != 'cpu': break # Found a device

        if step <= warmup_steps and warmup_steps > 0 and has_grad:
            # Calculate total squared global norm of gradients
            global_norm_sq_fp32 = torch.tensor(0.0, dtype=torch.float32, device=global_norm_device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        global_norm_sq_fp32.add_(p.grad.float().pow(2).sum())

            if global_norm_sq_fp32 > 0: # Avoid division by zero
                 global_norm_fp32 = torch.sqrt(global_norm_sq_fp32)
                 eps_fp32 = torch.tensor(eps, dtype=torch.float32, device=global_norm_device)
                 global_clip_factor_fp32 = torch.tensor(lambda_abs, dtype=torch.float32, device=global_norm_device) / (global_norm_fp32 + eps_fp32)
                 global_clip_factor_fp32 = torch.min(global_clip_factor_fp32, torch.tensor(1.0, device=global_norm_device, dtype=torch.float32))
            else:
                 # If global norm is 0, no clipping is needed, factor is 1.0
                 global_clip_factor_fp32 = torch.tensor(1.0, device=global_norm_device, dtype=torch.float32) # Put on device
        else:
             # If not in warm-up or no grads, global clip factor is 1.0 (ensure it's on a device if possible)
             device = global_norm_device # Use the device found earlier, or 'cpu'
             global_clip_factor_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)

        return global_clip_factor_fp32

@torch.no_grad()
def _apply_adagc_clipping_and_update_gamma(
    self,
    grad: torch.Tensor,
    state: Dict[str, Any],
    step: int,
    warmup_steps: int = 0,
    lambda_rel: float = 1.05,
    ema_beta: float = 0.98,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies AdaGC or global clipping to the gradient and updates the gamma state.
    Returns the clipped gradient as an FP32 tensor
    """
    with torch.no_grad():
        grad_fp32 = grad.float()


        device = grad_fp32.device
        # Get gamma state (always FP32 scalar tensor)
        if 'adagc_gamma' not in state:
            state['adagc_gamma'] = torch.tensor(lambda_rel, dtype=torch.float32, device=device)
        gamma_fp32 = state['adagc_gamma']


        # Determine the FINAL clipping factor for this parameter
        final_clip_factor_fp32: torch.Tensor # Define type hint

        if step <= warmup_steps and warmup_steps > 0:
             # Warm-up phase: Use the pre-calculated global clip factor
             # Ensure it's on the same device as the gradient we're modifying
             final_clip_factor_fp32 = self._global_clip_factor_fp32.to(device)
        else:
             # AdaGC phase: Calculate the local AdaGC scaling factor in FP32
             # Norm of the raw gradient (grad_fp32 is FP32)
             param_norm_fp32 = torch.linalg.norm(grad_fp32)

             # Get previous EMA gamma (gamma_fp32 is FP32 scalar tensor)
             prev_gamma_fp32 = gamma_fp32
             # Calculate adaptive threshold (FP32 scalar tensor)
             Arel_t = torch.tensor(lambda_rel, dtype=torch.float32, device=device)
             eps_ema_t = torch.tensor(eps, dtype=torch.float32, device=device)
             adaptive_threshold_fp32 = Arel_t * (prev_gamma_fp32 + eps_ema_t)

             # Calculate the static clipping factor: min(1.0, threshold / norm)
             eps_t = torch.tensor(eps, dtype=torch.float32, device=device)
             ratio_fp32 = adaptive_threshold_fp32 / (param_norm_fp32 + eps_t) # Add eps_t to denominator
             # Ensure ratio is not NaN/Inf in edge cases (though adding eps should help)
             ratio_fp32 = torch.nan_to_num(ratio_fp32, nan=1.0, posinf=1.0, neginf=1.0)

             # Create 1.0 tensor on the correct device for torch.min
             one_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)
             final_clip_factor_fp32 = torch.min(one_fp32, ratio_fp32)


        clipped_grad_fp32 = grad_fp32 # reference for clarity
        clipped_grad_fp32.mul_(final_clip_factor_fp32)
        clipped_param_norm_fp32 = torch.linalg.norm(clipped_grad_fp32)
        ema_beta_t = torch.tensor(ema_beta, dtype=torch.float32, device=device)
        gamma_fp32.mul_(ema_beta_t).add_(clipped_param_norm_fp32, alpha=1.0 - ema_beta_t) # gamma_fp32 is state['gamma']

        # Return the clipped FP32 gradient and the final clipping factor
        return clipped_grad_fp32


@torch.no_grad()
def _paper_orthograd(param, grad, alpha: float = 1.0, eps: float|torch.Tensor = 1e-20):
    """Applies orthogonal projection to a single parameter's gradient."""

    # Skip for scalars
    if param.ndim == 0 or param.numel() <= 1:
        return

    # Flatten parameter and gradient
    w = param.view(-1) # Use p.data to avoid graph tracking if not needed
    g = grad.view(-1)

    w_norm_sq = torch.dot(w, w)

    # Only project if the weight norm is significant
    # If w_norm_sq is near zero, the parameter contributes little,
    # and projection is ill-defined or numerically unstable.
    # Leave the gradient untouched in this case.

    if w_norm_sq > eps:
        # Calculate projection of g onto w: (w·g / w·w) * w
        proj_coeff = torch.dot(w, g) / w_norm_sq # Note: w_norm_sq already > eps
        g_parallel = proj_coeff * w

        # Subtract the parallel component to get the orthogonal one
        g_orth = g - alpha * g_parallel
        # Apply scaled orthogonalization
        g_orth_scaled = g_orth.mul_(grad.norm(2) / (g_orth.norm(2) + eps))

        # Update the gradient in-place with the orthogonal component
        grad.copy_(g_orth_scaled.view_as(grad))
    # Else: w_norm_sq is too small, leave p.grad as is.

@torch.no_grad()
def _paper_orthograd_compile(param, grad, alpha: float = 1.0, eps: float|torch.Tensor = 1e-20):
    """Applies orthogonal projection to a single parameter's gradient."""

    # Skip for scalars
    if param.ndim == 0 or param.numel() <= 1:
        return

    # Flatten parameter and gradient
    w = param.view(-1) # Use p.data to avoid graph tracking if not needed
    g = grad.view(-1)

    w_norm_sq = torch.dot(w, w)

    # Only project if the weight norm is significant
    # If w_norm_sq is near zero, the parameter contributes little,
    # and projection is ill-defined or numerically unstable.
    # Leave the gradient untouched in this case.

    # Calculate projection of g onto w: (w·g / w·w) * w
    proj_coeff = torch.dot(w, g) / w_norm_sq # Note: w_norm_sq already > eps
    g_parallel = proj_coeff * w

    # Subtract the parallel component to get the orthogonal one
    g_orth = g - alpha * g_parallel
    # Apply scaled orthogonalization
    g_orth_scaled = g_orth.mul_(grad.norm(2) / (g_orth.norm(2) + eps))

    # Update the gradient in-place with the orthogonal component
    grad.copy_(torch.where(w_norm_sq > eps, g_orth_scaled.view_as(grad), grad))
    # Else: w_norm_sq is too small, leave p.grad as is.

def _apply_cautious_compile(update: torch.Tensor, grad: torch.Tensor) -> None:
    r"""Compiled variant of apply_cautious — fullgraph-safe with boolean op resolved at trace time."""
    mask = (update * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_(min=1e-3))
    update.mul_(mask)


def _get_compiled_apply_cautious():
    r"""Lazily compile _apply_cautious_compile using torch.compile function call."""
    if 'apply_cautious' not in _compiled_fns:
        with torch._dynamo.utils.disable_cache_limit():
            _compiled_fns['apply_cautious'] = torch.compile(
                _apply_cautious_compile, fullgraph=True, mode="default"
            )
    return _compiled_fns['apply_cautious']


@torch.no_grad()
def apply_cautious(update: torch.Tensor, grad: torch.Tensor, torch_compile: bool = False) -> None:
    r"""Apply the Cautious Optimizer feature.

    :param update: torch.Tensor. update. it'll be masked in in-place manner.
    :param grad: torch.Tensor. gradient.
    :param torch_compile: bool. route through torch.compile'd wrapper.
    """
    if torch_compile:
        _get_compiled_apply_cautious()(update, grad)
    else:
        mask = (update * grad > 0).to(grad.dtype)
        mask.div_(mask.mean().clamp_(min=1e-3))
        update.mul_(mask)

def debias(beta: float, step: int) -> float:
    """Adam-style debias correction. Returns `1 - beta ** step`."""
    return 1 - beta**step


def debias_beta(beta: float, step: int) -> float:
    """Applies the Adam-style debias correction into beta.

    Simplified version of `betahat = beta*(1-beta**(step-1))/(1-beta**step)`
    """
    return (beta**step - beta) / (beta**step - 1)

@torch.no_grad()
def find_closest_orthogonal_matrix(self, A: torch.Tensor, max_iter: int = 8) -> torch.Tensor:
    """
    Find the closest orthogonal matrix to A using an iterative method
    """    
    k, n = A.shape
    if k == 0 or n == 0:
        # Handle empty matrix case: return an empty matrix of the correct shape
        # or raise an error, depending on desired behavior.
        # For now, returning a zero matrix of the same shape.
        # The SVD of an empty or zero-dimension matrix is problematic.
        # Orthogonality is trivial/undefined for k=0 or n=0 in this context.
        # If k > n, the concept of "orthogonal matrix L (k,n)" means L @ L.T = I_k,
        # which is only possible if k <= n. If k < n, it means L.T @ L = I_n.
        # The paper implies symmetric leakage correction, often k=n.
        # If k != n, it's finding a matrix with orthonormal rows (if k < n) or columns (if k > n).
        # Let's assume k <= n for "L L.T = I" interpretation, or simply closest in Frobenius norm to an orthogonal matrix.
        # The algorithm tries to make V (n,k if scA is n,k) orthogonal, then scales.
        # If A is (0,N) or (N,0), let's return zeros.
        return torch.zeros_like(A)


    # Determine the floating point type for calculations involving real numbers (like tolerance parts)
    # If A is complex, its real part's dtype is used. Otherwise, A's dtype.
    # However, torch.finfo requires a float dtype.
    float_dtype = A.real.dtype if A.is_complex() else A.dtype
    
    # Tolerance calculation
    # Original: np.max((1, np.max(A.shape) * np.linalg.svd(A.T, False, False)[0])) * np.finfo(A.dtype).eps
    # svd(A.T) -> singular values of A.T. A.T is (n, k)
    # Need to handle if A.T is empty or too small for SVD
    if A.T.shape[0] == 0 or A.T.shape[1] == 0: # Should be caught by k==0 or n==0 above
        s_A_T_first = torch.tensor(0.0, device=A.device, dtype=float_dtype)
    else:
        try:
            s_A_T = torch.linalg.svdvals(A.T)
            s_A_T_first = s_A_T[0] if s_A_T.numel() > 0 else torch.tensor(0.0, device=A.device, dtype=float_dtype)
        except RuntimeError: # SVD might fail for ill-conditioned or zero-sized dim
            s_A_T_first = torch.tensor(0.0, device=A.device, dtype=float_dtype)


    # Ensure max_A_shape_val is a float tensor for multiplication
    max_A_shape_val = torch.tensor(float(max(A.shape)), device=A.device, dtype=float_dtype)
    
    tolerance_factor = torch.max(
        torch.tensor(1.0, device=A.device, dtype=float_dtype),
        max_A_shape_val * s_A_T_first
    )
    TOLERANCE = tolerance_factor * torch.finfo(float_dtype).eps

    # Helper for relative difference, adding epsilon for numerical stability
    # Ensure reldiff output is float_dtype for comparison with TOLERANCE
    # Note: rhos will be float_dtype
    eps_val = torch.finfo(float_dtype).eps
    def reldiff(a, b): # a and b are scalar tensors
        return 2 * torch.abs(a - b) / (torch.abs(a) + torch.abs(b) + eps_val)

    def convergence(rho, prev_rho):
        return reldiff(rho, prev_rho) <= TOLERANCE

    A_conj = A.conj()
    # d = sqrt(sum(A * A_conj, dim=1)) -> shape (k)
    # (A * A_conj) is element-wise. sum over n (dim=1)
    d_val = torch.sqrt(torch.sum(A * A_conj, dim=1)) # d_val is real, shape (k,)

    rhos = torch.zeros(max_iter, device=A.device, dtype=float_dtype)
    L = torch.zeros_like(A) # Initialize L, will be updated in the loop

    for i in range(max_iter):
        # scA = A.T * d  (NumPy broadcasting)
        # A.T is (n, k), d_val is (k,). We need d_val to be (1, k) for broadcasting.
        scA = A.T * d_val.unsqueeze(0) # d_val.unsqueeze(0) is (1, k)

        # Perform SVD: u is (n, p), s is (p,), vh is (p, k) where p = min(n, k)
        try:
            u, s, vh = torch.linalg.svd(scA, full_matrices=False)
        except RuntimeError as e:
            # SVD can fail (e.g. if scA contains NaNs or Infs, or LAPACK error)
            # print(f"SVD failed at iteration {i}: {e}. Returning current L or A.")
            # Depending on requirements, could return A, L from previous iter, or re-raise.
            # For now, if SVD fails, break and return the last computed L (or initial zeros if i=0).
            if i == 0: L = A.clone() # Or some other fallback
            break


        V = u @ vh # V is (n, k) (if n>=k) or (n,n)@(n,k) (if n < k, u is (n,n), s is (n), vh is (n,k)) -> (n,k)
                # Actually, V is (n, p) @ (p, k) -> (n, k)
                # This V is the "orthogonal part" of scA.
                # If scA is (N, M), U is (N, min(N,M)), S is (min(N,M)), Vh is (min(N,M), M)
                # So V = U @ Vh will be (N, M)

        # d = sum(A_conj * V.T, dim=1)
        # A_conj is (k, n). V.T is (k, n). Element-wise product. Sum over n (dim=1).
        d_val = torch.sum(A_conj * V.T, dim=1) # d_val is potentially complex, shape (k,)

        # L = (V * d).T
        # V is (n, k). d_val is (k,). We need d_val to be (1, k) for broadcasting.
        # Result (V * d_val_row) is (n, k). Transpose to (k, n).
        L = (V * d_val.unsqueeze(0)).T
        
        E = A - L
        # rho is sqrt(sum(E * E_conj)). This is Frobenius norm of E.
        # E*E.conj() is element-wise, sum over all elements. Result is real.
        current_rho = torch.sqrt(torch.sum(E * E.conj())) # scalar, real
        rhos[i] = current_rho

        if torch.isnan(current_rho) or torch.isinf(current_rho):
            # print(f"Warning: rho is NaN or Inf at iteration {i}. Stopping.")
            if i == 0: L = A.clone() # Fallback if first iteration yields NaN/Inf
            break

        if i > 0:
            # Check for convergence against previous rho.
            # Ensure rhos[i-1] is not nan/inf for a valid comparison.
            if not (torch.isnan(rhos[i-1]) or torch.isinf(rhos[i-1])):
                if convergence(rhos[i], rhos[i-1]):
                    break
            elif torch.isnan(rhos[i-1]) or torch.isinf(rhos[i-1]):
                # If previous rho was bad, can't check convergence. Maybe continue or break.
                # print(f"Warning: prev_rho was NaN/Inf at iter {i}, cannot check convergence.")
                pass # Continue, hoping it stabilizes
    
    # The loop might complete max_iter or break early.
    # rhos are kept for potential debugging, could be returned or discarded.
    return L

@torch.no_grad()
def adaptive_eps(grad: torch.Tensor, group:dict, rms_grad: torch.Tensor = None) -> torch.Tensor:
    if 'eps_t' not in group or group['eps_t'].device != group["params"][0].device:
        group['eps_t'] = torch.tensor(group['eps'], device=group["params"][0].device)
    if group['eps_floor'] is not None and group['eps_floor'] < group['eps']:
        if 'eps2_t' not in group or group['eps2_t'].device != group["params"][0].device:
            group['eps2_t'] = torch.tensor(group['eps2'], device=group["params"][0].device)
        if 'eps_floor_t' not in group or group['eps_floor_t'].device != group["params"][0].device:
            group['eps_floor_t'] = torch.tensor(group['eps_floor'], device=group["params"][0].device)

        if rms_grad is None:
            rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
        val_to_bound = group['eps2_t'] * rms_grad
        return torch.clamp(val_to_bound, min=group['eps_floor'], max=group['eps'])
    else:
        return group['eps_t']

def _apply_weight_decay_compile(
    p: torch.Tensor,
    grad: Optional[torch.Tensor],
    lr: torch.Tensor,
    weight_decay: torch.Tensor,
    weight_decouple: bool,
    fixed_decay: bool,
    ratio: Optional[torch.Tensor] = None,
    cautious_weight_decay: bool = False,
) -> None:
    r"""Compiled variant of apply_weight_decay.  Scalar arguments are tensors so that
    torch.compile traces data-dependent branches correctly."""
    if cautious_weight_decay:
        apply_cautious_weight_decay(p, grad, lr, weight_decay)
    elif weight_decouple:
        p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else torch.tensor(1.0, device=p.device, dtype=p.dtype)))
    elif weight_decay > 0.0 and grad is not None:
        grad.add_(p, alpha=weight_decay)


def _get_compiled_apply_weight_decay():
    r"""Lazily compile _apply_weight_decay_compile using torch.compile function call."""
    if 'apply_weight_decay' not in _compiled_fns:
        with torch._dynamo.utils.disable_cache_limit():
            _compiled_fns['apply_weight_decay'] = torch.compile(
                _apply_weight_decay_compile, fullgraph=True, dynamic=False
            )
    return _compiled_fns['apply_weight_decay']


def apply_weight_decay(
    p: torch.Tensor,
    grad: Optional[torch.Tensor],
    lr: float,
    weight_decay: float,
    weight_decouple: bool,
    fixed_decay: bool,
    ratio: Optional[float] = None,
    cautious_weight_decay: bool = False,
    torch_compile: bool = False,
) -> None:
    """Apply weight decay in an in-place manner.

    Args:
        p (torch.Tensor): Parameter tensor to apply weight decay to.
        grad (torch.Tensor): Gradient tensor of parameter p.
        lr (float): Learning rate to scale the update.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        weight_decouple (bool): If True, applies decoupled weight decay as in AdamW.
        fixed_decay (bool): If True, fixes weight decay to not depend on learning rate.
        ratio (Optional[float]): Optional scaling factor for weight decay.
        cautious_weight_decay (bool): If True, applies cautious weight decay.
        torch_compile (bool): If True, route through torch.compile'd wrapper.
    """
    if torch_compile:
        lr_t = torch.tensor(lr, device=p.device, dtype=p.dtype) if not isinstance(lr, torch.Tensor) else lr
        wd_t = torch.tensor(weight_decay, device=p.device, dtype=p.dtype) if not isinstance(weight_decay, torch.Tensor) else weight_decay
        ratio_t = torch.tensor(ratio, device=p.device, dtype=p.dtype) if (ratio is not None and not isinstance(ratio, torch.Tensor)) else ratio
        _get_compiled_apply_weight_decay()(p, grad, lr_t, wd_t, weight_decouple, fixed_decay, ratio_t, cautious_weight_decay)
    elif cautious_weight_decay:
        apply_cautious_weight_decay(p, grad, lr, weight_decay)
    elif weight_decouple:
        p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
    elif weight_decay > 0.0 and grad is not None:
        grad.add_(p, alpha=weight_decay)

def apply_cautious_weight_decay(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply cautious weight decay (CWD) in an in-place manner.

    Args:
        p (torch.Tensor): Parameter tensor to apply weight decay to.
        update (torch.Tensor): update tensor.
        lr (float): Learning rate to scale the update.
        weight_decay (float): Weight decay coefficient (L2 penalty).

    """
    p.copy_(torch.where(update * p >= 0, p * (1.0 - weight_decay * lr), p))