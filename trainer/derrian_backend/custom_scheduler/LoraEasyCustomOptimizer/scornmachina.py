# SCORN from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from enum import IntEnum
import math
from .utils import (copy_stochastic_, _get_compiled_stable_spam_clipping, _stable_spam_clipping_impl, adagc_global_clipping_calc,
                    _apply_adagc_clipping_and_update_gamma, _paper_orthograd, adaptive_eps)
from pytorch_optimizer.base.exception import NoSparseGradientError
from typing import Optional
import logging


# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
def zero_power_via_newton_schulz_6(
    g: torch.Tensor, eps: float = 1e-16
) -> torch.Tensor:
    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param g: torch.Tensor. matrix.
    :param num_steps: int. number of iterations.
    :param eps: float. add this times I to G, to make is positive definite. For scaling, we multiply it by the largest
        eigenvalue of G.
    :param weights: Tuple[int, int, int]. weights.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    x = g.float()
    x.div_(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    #for _ in range(num_steps):
    for weight in abc_list:
        a = x @ x.T
        b = weight[1] * a + weight[2] * a @ a
        x = weight[0] * x + b @ x

    if g.size(0) > g.size(1):
        x = x.T

    x = torch.einsum('ij,ij,ab->ab', g.type_as(x), x, x)

    return x

class LMONorm(IntEnum):
    r"""normalization types."""

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:
    r"""Base class to perform norm onto Scion. This class does no norm."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        r"""Initialize parameter."""
        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        r"""Get LMO."""
        return grad


class Col(Norm):
    r"""col-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=0, keepdim=True)).mul_(math.sqrt(x.size(0)))
        if self.normalized:
            x.div_(x.size(1))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        d_in, d_out = grad.size()

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True)) / math.sqrt(d_in)
        if self.normalized:
            rms_value.mul_(d_out)

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class Row(Norm):
    r"""row-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=-1, keepdim=True))
        if self.normalized:
            x.div_(math.sqrt(x.size(-1)))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=-1, keepdim=True))
        if self.normalized:
            rms_value.mul_(math.sqrt(grad.size(-1)))

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class BiasRMS(Norm):
    r"""bias RMS."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.zeros_(x)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
        grad /= rms_value.add_(eps)
        return grad


class SpectralConv(Norm):
    r"""spectral-convolution normalization.

    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        d_out, d_in, kernel_size, *_ = x_fp64.size()

        for i in range(kernel_size):
            for j in range(kernel_size):
                torch.nn.init.orthogonal_(x_fp64[..., i, j])

        x_fp64.mul_(math.sqrt(d_out / d_in) / (kernel_size**2))

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_6(grad.view(len(grad), -1), eps=eps).view(grad.shape)

        d_out, d_in, kernel_size, *_ = grad.size()

        grad *= math.sqrt(d_out / d_in) / (kernel_size**2)

        return grad


class Spectral(Norm):
    r"""spectral normalization.

    :param max_scale: bool. set upper bound (1.0) of the scale.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        torch.nn.init.orthogonal_(x_fp64)

        d_out, d_in = x_fp64.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        x_fp64.mul_(scale)

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_6(grad.view(len(grad), -1), eps=eps).view(grad.shape)

        d_out, d_in = grad.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        grad *= scale

        return grad


class Sign(Norm):
    r"""sign normalization.

    :param zero_init: bool. initialize with zero.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    """

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            return torch.nn.init.zeros_(x)

        d_in: int = x.size(1)

        x = 2 * torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) - 1
        if self.normalize:
            x.div_(d_in)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        d_in: int = grad.size(1)
        return torch.sign(grad).div_(d_in) if self.normalize else torch.sign(grad)


class Auto(Norm):
    r"""choose Norm type automatically."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ndim: int = x.ndim
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        ndim: int = grad.ndim
        if ndim in (0, 1):
            return BiasRMS().lmo(grad, eps=eps)
        if ndim == 2:
            return Spectral().lmo(grad, eps=eps)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad, eps=eps)
        raise NotImplementedError


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    r"""Build LMONorm by given norm_type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()

class SCORNMachina(Optimizer):
    r"""
    SCORNMachina: Applying the idea of no gradient accumulation, as its been superseded by momentum. Faster training, smoother weights, Papa Johns. 
    
    For optimal use: Utilize a gradient accumulation size of 1, highest batch size you can handle, adjust LR as needed. Standard AdamW LR ought to be stable enough.
    
    If you want extra speed, you can utilize the `reset_interval` and `reset_increment` parameter to reset the optimizer states, speeding up gradient descent and accelerating leaving local minima.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float):
            Coefficient used for computing the running average, and the running square of running average (default: 0.95, 0.9999)
        focus_ratio (float):
            Ratio for FOCUS' valley attraction force - https://arxiv.org/abs/2501.12243. (default: 0.0, recommended if used: 0.1)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.998).
        amp (float):
            Beta-adjusted scaling parameter for adding the running average to the gradient, functionally acts as strength value for a low-pass filter. (default: 5.0).
        reset_interval (int):
            Resets the optimizers running averages after (reset_interval + reset_increment * times_reset) steps (default: 0, recommended if used: >=100).
        reset_increment (int):
            Increments the reset_interval by this amount after every reset (default: 0, recommended if used: >=100).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2 in place of epsilon - https://arxiv.org/abs/2407.05872 (default: False).
        spectral_update_scale (bool):
            Scale the spectral gradient by this value, generally intended for when constrain is used, - https://arxiv.org/pdf/2502.07529 (default: 1.0).
        constrain (bool):
            Scale the parameters by the step size to functionally constrain the norm of the parameters, recommended to divide usual learning rate by spectral_update_scale. (default: False).
        cautious_min (bool):
            Use cautious mask on full step update, clamped to a minimum of cautious_min - https://arxiv.org/abs/2411.16085 (default: 1.0, thus disabling the mask. Use 0 to fully utilize the mask).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: tuple = (0.95, 0.997),
        focus_ratio: float = 0.0,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        amp: float = 5.0,
        reset_interval: int = 0,
        reset_increment: int = 0,
        orthograd: bool = True,
        orthograd_alpha: float = 1.0,
        spectral_update_scale: float = 1.0,
        constrain: bool = False,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
        use_stable_spam_clipping: bool = False,
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: float = 1e-16,
        use_adagc: bool = False,
        adagc_warmup_steps: int = 0,
        amsgrad: bool = False,
        amsgrad_decay_rate: Optional[float] = None,
        torch_compile: bool = False,
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Optimizer argument '{key}' passed into SCORNMachina. It will be ignored."
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

        self._init_lr = lr
        self.use_adagc = use_adagc
        if self.use_adagc:
            self.use_adagc = use_adagc
            self._adagc_global_clip_factor_fp32 = None
            self.adagc_warmup_steps = adagc_warmup_steps
            self._global_step = 0

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        defaults = dict(
            lr = lr,
            betas = betas,
            focus_ratio = focus_ratio,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            amp = amp,
            reset_interval = reset_interval,
            reset_increment = reset_increment,
            orthograd = orthograd,
            orthograd_alpha = orthograd_alpha,
            spectral_update_scale = spectral_update_scale,
            constrain = constrain,
            cautious_min = cautious_min,
            stochastic_fp = stochastic_fp,
            use_stable_spam_clipping = use_stable_spam_clipping,
            eps = eps,
            eps2 = eps2,
            eps_floor = eps_floor,
            amsgrad = amsgrad,
            amsgrad_decay_rate = amsgrad_decay_rate,
            torch_compile = torch_compile,
            sync_chunk_size = sync_chunk_size,
            state_storage_dtype = final_dtype,
            state_storage_device = state_storage_device,
        )

        super(SCORNMachina, self).__init__(params, defaults)
    
    @torch.no_grad()
    def reset_momentums(self, momentum, sq_momentum):
        momentum.copy_(torch.zeros_like(momentum))
        sq_momentum.copy_(torch.zeros_like(sq_momentum))

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(LMONorm.AUTO)
            for p in group['params']:
                state = self.state[p]
                norm.init(p)

                if group["reset_interval"] > 0:
                    state["times_zero"] = 0
                    state["steps_since_reset"] = 1

                # Exponential moving average of gradient values
                state["ema"] = torch.zeros_like(p, 
                                                dtype=self.state_storage_dtype, 
                                                device=self.state_storage_device)
                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p, 
                                                        dtype=self.state_storage_dtype, 
                                                        device=self.state_storage_device)

                if group["focus_ratio"] > 0.0:
                    state["pbar"] = torch.zeros_like(p, 
                                                        dtype=self.state_storage_dtype, 
                                                        device=self.state_storage_device)

                if self.state_storage_device == "cpu":
                    # Exponential moving average of gradient values
                    state["ema"] = state["ema"].pin_memory()
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = state["ema_squared"].pin_memory()

                    if group["focus_ratio"] > 0.0:
                        state["pbar"] = state["pbar"].pin_memory()


    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.use_adagc:
            self._global_step += 1
            self._global_clip_factor_fp32 = adagc_global_clipping_calc(self, self._global_step, self.adagc_warmup_steps)

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            betas = group["betas"]
            focus_ratio = group["focus_ratio"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            amp = group["amp"]
            use_orthograd = group["orthograd"]
            step = group['step']
            spectral_update_scale = group['spectral_update_scale']
            use_stable_spam_clipping = group['use_stable_spam_clipping']
            eps = group['eps']
            if 'eps_t' not in group or group['eps_t'].device != group["params"][0].device:
                group['eps_t'] = torch.tensor(eps, device=group["params"][0].device)

            eps_floor = group['eps_floor']
            if eps_floor is not None and ('eps_floor_t' not in group or group['eps_floor_t'].device != group["params"][0].device):
                group['eps_floor_t'] = torch.tensor(eps_floor, device=group["params"][0].device)

            orthograd_alpha = group['orthograd_alpha']
            apply_ortho_to_group = group.get('is_ortho_group', False) # Default to False if key missing
            amsgrad = group['amsgrad']
            amsgrad_decay_rate = group['amsgrad_decay_rate']

            adopt_clip: float = (step-1)**0.25

            if spectral_update_scale > 0.:
                norm = build_lmo_norm(LMONorm.AUTO)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                device = p.device

                grad = p.grad.data

                # State initialization
                if len(state) == 0:
                    norm.init(p)

                    if group["reset_interval"] > 0:
                        state["times_zero"] = 0
                        state["steps_since_reset"] = 1

                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p, 
                                                    dtype=self.state_storage_dtype, 
                                                    device=self.state_storage_device)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p, 
                                                            dtype=self.state_storage_dtype, 
                                                            device=self.state_storage_device)

                    if group["focus_ratio"] > 0.0:
                        state["pbar"] = torch.zeros_like(p, 
                                                         dtype=self.state_storage_dtype, 
                                                         device=self.state_storage_device)

                    if self.state_storage_device == "cpu":
                        # Exponential moving average of gradient values
                        state["ema"] = state["ema"].pin_memory()
                        # Exponential moving average of squared gradient values
                        state["ema_squared"] = state["ema_squared"].pin_memory()

                        if group["focus_ratio"] > 0.0:
                            state["pbar"] = state["pbar"].pin_memory()                     

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                if focus_ratio > 0.0:
                    pbar = state["pbar"].to(
                        compute_device, 
                        non_blocking=True, 
                        dtype=torch.float32
                    )
                ema = state["ema"].to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                ema_squared = state["ema_squared"].to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                if apply_ortho_to_group and use_orthograd:
                    _paper_orthograd(param=p_fp32, grad=grad, alpha=orthograd_alpha)

                if self.use_adagc:
                    grad = _apply_adagc_clipping_and_update_gamma(self, grad=grad, state=state, step=step, warmup_steps=self.adagc_warmup_steps)

                if use_stable_spam_clipping:
                    if group['torch_compile']:
                        grad = _get_compiled_stable_spam_clipping()(state,
                                            grad,
                                            step=group['step'],
                                            eps=group['eps_floor_t'])
                    else:
                        grad = _stable_spam_clipping_impl(state, 
                                            grad, 
                                            step=group['step'], 
                                            eps=group['eps_floor_t'])

                curr_eps = adaptive_eps(grad, group)

                if group["reset_interval"] > 0:
                    if state["steps_since_reset"] // (group["reset_interval"] + (group["reset_increment"] * state["times_zero"])) > 0:
                        self.reset_momentums(ema, ema_squared)
                        if focus_ratio > 0. and 'pbar' in state:
                            pbar = pbar.copy_(torch.zeros_like(p))
                        state["times_zero"] += 1
                        state["steps_since_reset"] = 1
                    step = state["steps_since_reset"]

                slow_beta = ((betas[1]**step - betas[1]) / (betas[1]**step - 1.0))

                bias_correction = 1 - betas[0] ** step
                bias_correction_sqrt = (1 - slow_beta ** step) ** (1 / 2)
                step_size = lr

                # RMS Norm
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad.div_(rms)

                # SCION spectral norm
                grad = norm.lmo(grad, eps=group['eps_floor_t'])#.mul_(spectral_update_scale)

                # Adaptive ema
                mask = (grad * ema > 0).to(grad.dtype)
                mask.clamp_min_(betas[0])
                mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
                ema.mul_(mask)

                # Update ema
                ema.mul_(betas[0]).add_(grad, alpha=1 - betas[0])

                # Compass amplification
                c_t = grad.add(ema.div(bias_correction), alpha=amp)

                if step == 1 or (group["reset_interval"] > 0 and state["steps_since_reset"] // (group["reset_interval"] + (group["reset_increment"] * (max(0,state["times_zero"] - 1)))) > 0):
                    if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                        ema_squared.copy_(norm.lmo(c_t.pow(2), eps=group['eps_floor_t']))
                    else:
                        state["ema_squared"].copy_(norm.lmo(c_t.pow(2), eps=group['eps_floor_t']))
                else:
                    # AdamW debias
                    denom = ema_squared.sqrt().div_(bias_correction_sqrt).add_(curr_eps)

                    new_ema_squared = ema_squared.mul(slow_beta).addcmul_(c_t, c_t, value=1 - slow_beta)

                    # ADOPT update
                    if amsgrad:
                        torch.maximum(ema_squared.mul(amsgrad_decay_rate if amsgrad_decay_rate is not None else slow_beta), new_ema_squared, out=ema_squared)
                    else:
                        ema_squared.copy_(new_ema_squared)

                    # Atan2-Adamw
                    full_step = c_t.div(denom).mul_(spectral_update_scale)

                    if focus_ratio > 0. and 'pbar' in state:
                        pbar.lerp_(p_fp32, weight=1 - betas[0])
                        pbar_hat = pbar.div(bias_correction)
                        pbar_step = p_fp32 - pbar_hat
                        full_step.add_(pbar_step, alpha=focus_ratio)

                    if weight_decay != 0 and not group["constrain"]:
                        # Perform weight decay
                        grad_weights = p_fp32

                        full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])

                    # Apply caution as per 'Cautious Optimizers' with a modified minimum.
                    if group["cautious_min"] != 1.0:
                        mask = (full_step * grad > 0).to(full_step.dtype)
                        mask.clamp_min_(group["cautious_min"])
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        full_step = full_step.mul(mask)

                    if group["constrain"]:
                        p_fp32.mul_(1.0 - step_size)

                    p_fp32.add_(full_step.clamp_(-adopt_clip, adopt_clip), alpha=-step_size)
                    
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
                    if focus_ratio > 0.0:
                        copy_stochastic_(state["pbar"], pbar)
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                else:
                    if focus_ratio > 0.0:
                        state["pbar"].copy_(pbar, non_blocking=True)
                    state["ema"].copy_(ema, non_blocking=True)
                    state["ema_squared"].copy_(ema_squared, non_blocking=True)

                if group["reset_interval"] > 0:
                    state["steps_since_reset"] += 1

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss