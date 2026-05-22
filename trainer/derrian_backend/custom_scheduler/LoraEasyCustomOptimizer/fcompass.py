import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, NORM_TYPE, agc, adaptive_eps
import math
from torch.nn.functional import softplus

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm

from typing import Optional

# Fisher optimizer (FAdam) from https://github.com/lessw2020/FAdam_PyTorch/blob/main/fadam.py by Less Wright (lessw2020), I may not know them, but I am aware of their expertise. Many thanks for your contributing work!
# Original optimizer (Compass) from https://github.com/lodestone-rock/compass_optimizer/blob/main/compass.py by lodestone-rock, many thanks for their optim, help, and ideas!
# FCompass from https://github.com/Clybius/Personalized-Optimizers/blob/main/FCompass.py by Clybius
# Defaults tuned for lora training based on testing
class FCompass(Optimizer):
    r"""
    Fisher Compass: Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-05)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-30).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr=7e-05, #Original default 1e-3
        betas=(0.98, 0.999), #Original default 0.99, 0.999
        amp_fac=2,
        eps=1e-8,
        eps2=0.01,
        eps_floor=1e-16,
        weight_decay=0.001, #Original default 0.1
        clip=1.0,
        centralization=1.0,
        **kwargs,
    ):
        
        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
        )

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        super(FCompass, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompass'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            clip = group["clip"]
            centralization = group["centralization"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FCompass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p.data)
                    state['max_ema_squared'] = torch.zeros_like(p.data)
                    # Fisher Information Matrix
                    state['fim'] = torch.ones_like(p.data)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32).data
                    momentum, fim, max_ema_squared = state["momentum"].to(torch.float32), state["fim"].to(torch.float32), state['max_ema_squared'].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                else:
                    grad = grad.data
                    momentum, fim, max_ema_squared = state["momentum"], state["fim"], state['max_ema_squared']



                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                curr_beta2 = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction_sqrt = (1 - curr_beta2 ** group["step"]) ** (1 / 2)

                # Update fim
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1 - curr_beta2)

                curr_eps = adaptive_eps(grad, group)

                fim_base = fim**0.5 + curr_eps

                # Compute natural gradient
                grad_nat = grad / fim_base

                if clip != 0:
                    rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_nat.div_(divisor)

                # Momentum + Compass amplification
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)
                grad_nat.add_(momentum, alpha=amplification_factor)

                # Weight decay
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = p_fp32.data / fim_base
                else:
                    grad_weights = p.data / fim_base

                if clip != 0:
                    rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_weights.div_(divisor)
                
                full_step = grad_nat + (weight_decay * grad_weights)

                # Use the max. for normalizing running avg. of gradient (amsgrad)
                torch.max(max_ema_squared, max_ema_squared.mul(beta2).addcmul_(full_step, full_step, value=1 - beta2), out=max_ema_squared)
                denom = (max_ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.addcdiv_(full_step, denom, value=-lr)
                else:
                    p.data.addcdiv_(full_step, denom, value=-lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["max_ema_squared"], max_ema_squared)
                    copy_stochastic_(p, p_fp32)
        return loss
    
class FCompassADOPT(BaseOptimizer):
    r"""ADOPT Style Compass.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        cautious (bool)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: True)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
        compass_second_moment_smoothing (bool):
            Updates the second moment (i.e. ema / fim) with the Compass smoothed gradient. (Default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.97, 0.9999),
        amp_fac: float = 2.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        cautious: bool = True,
        debias_beta1: bool = False,
        debias_beta2: bool = True,
        compass_second_moment_smoothing: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'amp_fac': amp_fac,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'cautious': cautious,
            'debias_beta1': debias_beta1,
            'debias_beta2': debias_beta2,
            'compass_second_moment_smoothing': compass_second_moment_smoothing,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompassADOPT'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['momentum'] = torch.zeros_like(p)
                state['fim'] = torch.ones_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_step_size: float = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]
            amp_fac = group["amp_fac"]
            compass_second_moment_smoothing = group["compass_second_moment_smoothing"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['fim'] = torch.ones_like(p)

                momentum, fim = state['momentum'], state['fim']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    momentum, fim = momentum.to(torch.float32), fim.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(grad, grad.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                    update = grad_nat.add(momentum, alpha=amp_fac)

                    if compass_second_moment_smoothing:
                        fim.mul_(current_beta2).addcmul_(update, update.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)
                    else:
                        fim.mul_(current_beta2).addcmul_(grad, grad.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    p_fp32.add_(update * mask, alpha=-lr_step_size)

                    if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                        fim_sum += fim.sum()

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['momentum'], momentum)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(p, p_fp32)

        return loss
    
class FCompassADOPTMARS(BaseOptimizer):
    r"""Fisher ADOPT Style Compass + MARS correction.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        cautious (bool)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: False)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
        compass_second_moment_smoothing (bool):
            Updates the second moment (i.e. ema / fim) with the Compass smoothed gradient. (Default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.97, 0.9999),
        amp_fac: float = 2.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        gamma: float = 0.025,
        cautious: bool = True,
        debias_beta1: bool = False,
        debias_beta2: bool = True,
        compass_second_moment_smoothing: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'amp_fac': amp_fac,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'gamma': gamma,
            'cautious': cautious,
            'debias_beta1': debias_beta1,
            'debias_beta2': debias_beta2,
            'compass_second_moment_smoothing': compass_second_moment_smoothing,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompassADOPTMARS'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['momentum'] = torch.zeros_like(p)
                state['fim'] = torch.ones_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            step_size: float = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]
            amp_fac = group["amp_fac"]
            gamma = group["gamma"]
            compass_second_moment_smoothing = group["compass_second_moment_smoothing"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['fim'] = torch.ones_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()

                momentum, fim, previous_grad = state['momentum'], state['fim'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    momentum, fim, previous_grad = momentum.to(torch.float32), fim.to(torch.float32), previous_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(c_t, c_t.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)

                    grad_nat = c_t.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                    update = grad_nat.add(momentum, alpha=amp_fac)

                    if compass_second_moment_smoothing:
                        fim.mul_(current_beta2).addcmul_(update, update.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)
                    else:
                        fim.mul_(current_beta2).addcmul_(c_t, c_t.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    p_fp32.add_(update * mask, alpha=-step_size)

                    if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                        fim_sum += fim.sum()

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['momentum'], momentum)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

        return loss

class FCompassPlus(BaseOptimizer):
    r"""
    Fisher Compass: Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
        Components
            * Gradient centralization - https://arxiv.org/abs/2004.01461v2
            * Positive-Negative momentum - https://arxiv.org/abs/2103.17182
            * Norm loss - https://arxiv.org/abs/2103.06583v1
            * Lookahead - https://arxiv.org/abs/1907.08610
            * Softplus transformation - https://arxiv.org/abs/1908.00700
            * Gradient Normalization - https://arxiv.org/pdf/1711.02257 (?)
            * Diff amp - https://github.com/Clybius/Personalized-Optimizers/blob/main/FishMonger.py
            * Amsgrad - https://arxiv.org/pdf/1904.09237

    Arguments:
        :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: float. learning rate.
        :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
        :param use_softplus: bool. use softplus to smooth the updaate denominator.
        :param beta_softplus: float. beta for softplus.
        :param threshold_softplus: float. threshold after which scaling returns to linear. Originally set to 20 by default, instead follows adaptive eps when set to 0.
        :param clip: float. Clipping threshold for gradients.
        :param amp_fac: float. amplification factor for the first moment filter.
        :param centralize_gradients: bool. use GC both convolution & fc layers. Can be selectively applied an int: disabled(0), gradient(1), update(2), both(3)
        :param normalize_gradients: bool. use gradient normalization.  Can be selectively applied using an int: disabled(0), gradient(1), update(2), both(3)
        :param use_lookahead: bool. use lookahead. ADDS 1 State
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param fixed_decay: bool. fix weight decay.
        :param norm_loss_factor: float. norm loss factor.
        :param norm_loss_eps: float. Eps is the term added to the denominator to improve numerical stability.
        :param amsgrad: bool. If true, maintains and uses the max ema squared. ADDS 1 State
        :param use_pnm: bool. use positive negative momentum. ADDS 1 State
        :param pnm_beta: float. Manages the amplitude of the noise introduced by positive negative momentum. Negative values are valid.
        :param diff_amp: float. Accelerate the difference between the current and past gradient by this multiplicative value. 0 is off. ADDS 2 STATES
        :param diff_amp_beta: float. Coefficient used for computing running average of the current and past gradients
        :param eps: float. the maximum eps value for adaptive eps. Eps is the term added to the denominator outside of the root operation to improve numerical stability.
        :param eps2: float. used to multiple the grad rms for determining adaptive eps.
        :param eps_floor: float. term used to determine the floor for adaptive eps.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.97, 0.999),
        amp_fac: float = 2.0,
        centralize_gradients: int = 1,
        normalize_gradients: int = 0,
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float = 1e-16,
        weight_decay: float = 0.001,
        clip: float = 0.01,
        use_lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        norm_loss_factor: float = 0.0005,
        norm_loss_eps: float = 1e-8,
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        threshold_softplus: float = 0.0,
        amsgrad: bool = True,
        diff_amp: float = 0.0,
        diff_amp_beta: float = 0.999,
        use_pnm: bool = False,
        pnm_beta: float = 0.1,
        **kwargs,
    ):
        
        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr':lr,
            'betas':betas,
            'amp_fac':amp_fac,
            'eps':eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
            'weight_decay':weight_decay,
            'clip':clip,
            'use_lookahead' : use_lookahead,
            'lookahead_merge_time' : lookahead_merge_time,
            'lookahead_blending_alpha' : lookahead_blending_alpha,
            'use_softplus' : use_softplus,
            'beta_softplus' : beta_softplus,
            'amsgrad' : amsgrad,
            'diff_amp' : diff_amp,
            'diff_amp_beta' : diff_amp_beta,
            'centralize_gradients' : centralize_gradients,
            'normalize_gradients' : normalize_gradients,
            'threshold_softplus' : threshold_softplus,
            'use_pnm' : use_pnm,
            'pnm_beta' : pnm_beta,
            'norm_loss_factor' : norm_loss_factor,
            'norm_loss_eps' : norm_loss_eps
        }

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        self.use_lookahead = use_lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.threshold_softplus = threshold_softplus
        self.amsgrad = amsgrad
        self.diff_amp = diff_amp
        self.diff_amp_beta = diff_amp_beta
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.use_pnm = use_pnm
        self.pnm_beta = pnm_beta
        self.norm_loss_factor = norm_loss_factor
        self.clip = clip
        self.norm_loss_eps = norm_loss_eps
        self.lookahead_step: int = 0

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompassPlus'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                # Exponential moving average and squared exponential moving average gradient values
                state["momentum"] = torch.zeros_like(p)
                state['ema_squared'] = torch.zeros_like(p)
                # Fisher Information Matrix
                state["fim"] = torch.ones_like(p)

                if self.use_lookahead:
                    state['lookahead_params'] = p.clone()

                if self.use_pnm:
                    state['neg_momentum'] = torch.zeros_like(p)

                # Previous grad
                if self.diff_amp > 0.0:
                    state["ema_diff"] = torch.zeros_like(p.data)
                    state["previous_grad"] = grad.data.clone().mul_(-1.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                state = self.state[p]
                p_fp32 = p
                
                # State initialization
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p)
                    state['ema_squared'] = torch.zeros_like(p)
                    # Fisher Information Matrix
                    state["fim"] = torch.ones_like(p)

                    if self.use_lookahead:
                        state['lookahead_params'] = p.clone()

                    if self.use_pnm:
                        state['neg_momentum'] = torch.zeros_like(p)

                    # Previous grad
                    if self.diff_amp > 0.0:
                        state["ema_diff"] = torch.zeros_like(p)
                        state["previous_grad"] = grad.clone().mul_(-1.0)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                
                # Apply gradient centralization & normalization
                if self.centralize_gradients in {1,3}:
                    centralize_gradient(grad, gc_conv_only=False)

                if self.normalize_gradients in {1,3}:
                    normalize_gradient(grad)

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                fim, ema_squared = state["fim"], state['ema_squared']
                p_fp32 = p

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        momentum, neg_momentum = state['momentum'], state['neg_momentum']
                    else:
                        momentum, neg_momentum = state['neg_momentum'], state['momentum']
                else:
                    momentum = state["momentum"]

                if self.diff_amp > 0.0:
                    grad_diff = state["previous_grad"]
                    ema_diff = state['ema_diff']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff = grad_diff.to(torch.float32)
                        ema_diff = ema_diff.to(torch.float32)

                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff.add_(grad)

                    # Smooth the difference between previous grad and current grad
                    ema_diff.mul_(self.diff_amp_beta).add_(grad_diff, alpha=1 - self.diff_amp_beta)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["previous_grad"], -grad)
                        copy_stochastic_(state["ema_diff"], ema_diff)
                    else:
                        state["previous_grad"].copy_(-grad)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    momentum, fim, ema_squared = momentum.to(torch.float32), fim.to(torch.float32), ema_squared.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                    if self.use_pnm:
                        neg_momentum = neg_momentum.to(torch.float32)

                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]

                # bias correction step size
                # soft warmup
                curr_beta2 = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction_sqrt = (1 - curr_beta2 ** group["step"]) ** (1 / 2)

                # Update fim
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1 - curr_beta2)

                curr_eps = adaptive_eps(grad, group)

                fim_base = fim**0.5 + curr_eps

                # Compute natural gradient
                grad_nat = grad / fim_base

                if self.clip > 0.0:
                    rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / self.clip
                    grad_nat.div_(divisor)

                # Momentum + Compass amplification
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)
                if self.use_pnm:
                    noise_norm: float = math.sqrt((1.0 + self.pnm_beta) ** 2 + self.pnm_beta ** 2)
                    final_momentum = momentum.mul(1.0 + self.pnm_beta).add_(neg_momentum, alpha=-self.pnm_beta).mul_(1.0 / noise_norm)
                else:
                    final_momentum = momentum

                grad_nat.add_(final_momentum, alpha=amplification_factor)

                grad_weights = p_fp32 / fim_base
  
                if self.clip > 0.0:
                    rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / self.clip
                    grad_weights.div_(divisor)

                # Differential amplification
                diff_weights = ema_diff / fim_base if self.diff_amp else 0
                if self.diff_amp > 0.0 and self.clip > 0.0:
                    rms = diff_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / self.clip
                    diff_weights.div_(divisor)
                
                # Weight decay
                full_step = grad_nat + (weight_decay * grad_weights) - (self.diff_amp * diff_weights)

                # Use the max. for normalizing running avg. of gradient (amsgrad)
                if self.amsgrad:
                    torch.max(ema_squared, ema_squared.mul(beta2).addcmul_(full_step, full_step, value=1 - beta2), out=ema_squared)
                    de_nom = (ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)
                else:
                    ema_squared.mul_(beta2).addcmul_(full_step, full_step, value=1 - beta2)
                    de_nom = (ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus, threshold=self.threshold_softplus if self.threshold_softplus != 0 else curr_eps)

                if self.norm_loss_factor > 0.0:
                    # norm loss
                    correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p_fp32).add_(self.norm_loss_eps))
                    p_fp32.mul_(1.0 - lr * correction)

                full_step.div_(de_nom)

                if self.centralize_gradients in {2,3}:
                    centralize_gradient(full_step, gc_conv_only=False)

                if self.normalize_gradients in {2,3}:
                    normalize_gradient(full_step) 

                p_fp32.add_(full_step, alpha=-lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if self.use_pnm:
                        if group['step'] % 2 == 1:
                            copy_stochastic_(state["momentum"], momentum)
                        else:
                            copy_stochastic_(state["neg_momentum"], momentum)
                    else:
                        copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        if self.use_lookahead:
            self.lookahead_process_step()

        return loss
    
    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p_fp32 = p

                    lookahead_params = state['lookahead_params']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32 = p.clone().to(torch.float32)
                        lookahead_params = lookahead_params.to(torch.float32)

                    p_fp32.mul_(self.lookahead_blending_alpha).add_(
                        lookahead_params,
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(p, p_fp32)

                    state['lookahead_params'].copy_(p)