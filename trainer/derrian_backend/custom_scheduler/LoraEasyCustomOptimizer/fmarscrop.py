# FMARSCrop from https://github.com/Clybius/Personalized-Optimizers by Clybius
import torch
from .utils import copy_stochastic_, agc, NORM_TYPE, UPDATE_STRATEGY,_paper_orthograd, adaptive_eps
import math

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from torch.optim import Optimizer

# From pytorch_optimizer: https://github.com/kozistr/pytorch_optimizer
def unit_norm_func(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim = True
    dim = None

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

def agc_global_norm(p: torch.Tensor, grad: torch.Tensor, agc_eps: float, agc_clip_val: float, eps: float = 1e-6, unit_norm: bool = 1) -> torch.Tensor:
    r"""Clip gradient values based on the global norm.
    Scale the entire gradient tensor if its norm exceeds a threshold.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.

    :param p: torch.Tensor. Parameter tensor.
    :param grad: torch.Tensor. Gradient tensor.
    :param agc_eps: float. A small epsilon value to prevent division by zero.
    :param agc_clip_val: float. Clipping threshold multiplier.
    :param eps: float. Small value to prevent division by zero in normalization.
    """
    func = unit_norm_func
    if not unit_norm:
        func = torch.linalg.norm
    # Compute the global norm of the parameters and gradients
    p_norm = func(p).clamp_(min=agc_eps)
    g_norm = func(grad)

    # Compute the maximum allowed norm for the gradients
    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    return torch.where(g_norm > max_norm, clipped_grad, grad)


class FMARSCrop(BaseOptimizer):
    r"""
    FMARSCrop: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with ADOPT's AdamW changes (https://arxiv.org/abs/2411.02853).
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_lambda (float):
            The lambda value for slow momentum / EMA, controlling how much the momentum is amplified while being added to the update. (default: 2.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.99).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
        adaptive_clip (float):
            Adaptive clip value to applied to the MARS corrected gradient. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.0005)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: Betas = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 0.1,
        momentum_beta: float = 0.99,
        clip: float = 1.0,
        cautious: bool = True,
        gamma: float = 0.0005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'global',
        stable_weight_decay: bool = False,
        debias_beta2: bool = True,
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
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
            'weight_decay':weight_decay,
            'centralization':centralization,
            'moment_centralization':moment_centralization,
            'diff_mult':diff_mult,
            'momentum_beta':momentum_beta,
            'momentum_lambda':momentum_lambda,
            'clip':clip,
            'cautious':cautious,
            'gamma': gamma,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'stable_weight_decay': stable_weight_decay,
            'debias_beta2':debias_beta2,
            'weight_decouple':weight_decouple,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCrop'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = None
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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
                group['fim_mean_sqrt'] = None

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            step = group["step"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            stable_weight_decay = group["stable_weight_decay"]
            weight_decouple = group['weight_decouple']

            clip_lambda = (step - 1)**0.25

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if stable_weight_decay:
                    param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.to(dtype=p.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32,copy=True)

                prev_grad = prev_grad.add(grad)
                # Calculate cₜ (gradient with correction term)
                correction = (gamma * (beta1 / (1.0 - beta1))) * prev_grad
                c_t = grad + correction

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = grad_diff_fim.to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = grad_diff_fim.sqrt().add_(curr_eps)

                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1.0 - beta1).clamp_(-clip_lambda, clip_lambda)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                if group['step'] == 1:
                    fim.addcmul_(approx_grad_nat, approx_grad_nat)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)

                    grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_nat.div_(divisor)

                    momentum.mul_(momentum_beta).add_(grad_nat, alpha=1.0 - momentum_beta)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    if group['cautious']:
                        mask = (momentum_cent * grad_nat < 0).to(momentum_cent.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        momentum_cent = momentum_cent * mask

                    # Compass-style amplification
                    full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                    # center the gradient vector
                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                                centralization
                            )
                        )

                    # Perform weight decay
                    if weight_decay != 0 and weight_decouple:
                        if stable_weight_decay and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - weight_decay * lr * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = p_fp32.data.div(fim_base).div_(diff_fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                        if stable_weight_decay and group['fim_mean_sqrt'] is not None:
                            scale = 1.0 / group['fim_mean_sqrt']
                        else:
                            scale = 1.0

                        p_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * scale)

                    if group["cautious"]:
                        mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    # Apply full step
                    p_fp32.data.add_(full_step * mask, alpha=-lr)

                    fim.mul_(current_beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - current_beta2).clamp_(-clip_lambda, clip_lambda)

                if stable_weight_decay:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss

class FMARSCropV2ExMachina(BaseOptimizer):
    r"""
    FMARSCrop: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with ADOPT's AdamW changes (https://arxiv.org/abs/2411.02853).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            coefficients used for computing running averages of momentum,
            approx. natural grad FIM, and gradient difference FIM (default: 0.99, 0.9999, 0.999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_lambda (float):
            The lambda value for slow momentum / EMA, controlling how much the momentum is amplified while being added to the update. (default: 2.0).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool) (deprecated, use update strategy)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
            Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
            and 'grams' (https://arxiv.org/abs/2412.17107) (default: cautious)
        adaptive_clip (float):
            Adaptive clip value to applied to the MARS corrected gradient. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.0005)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: False)
        debias_beta2 (bool):
            Apply bias correction to fim. (Default: True)
        debias_beta3 (bool):
            Apply bias correction to diff fim. (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: Betas = (0.99,0.9999,0.999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 0.1,
        clip: float = 1.0,
        cautious: bool = False,
        gamma: float = 0.005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'global',
        stable_weight_decay: bool = False,
        debias_beta1: bool = False,
        debias_beta2: bool = True,
        debias_beta3: bool = False,
        update_strategy: UPDATE_STRATEGY = 'cautious',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
            'weight_decay':weight_decay,
            'centralization':centralization,
            'moment_centralization':moment_centralization,
            'diff_mult':diff_mult,
            'momentum_lambda':momentum_lambda,
            'clip':clip,
            'cautious':cautious,
            'gamma': gamma,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'stable_weight_decay': stable_weight_decay,
            'debias_beta1':debias_beta1,
            'debias_beta2':debias_beta2,
            'debias_beta3':debias_beta3,
            'weight_decouple':weight_decouple,
            'update_strategy': update_strategy,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCropV2ExMachina'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = None
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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
                group['fim_mean_sqrt'] = None

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            step = group["step"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            stable_weight_decay = group["stable_weight_decay"]
            weight_decouple = group['weight_decouple']

            clip_lambda = (step - 1)**0.25

            bias_correction1: float = self.debias(beta1, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            if group["debias_beta3"]:
                current_beta3: float = self.debias_beta(beta3, group['step'])
            else:
                current_beta3 = beta3

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if stable_weight_decay:
                    param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.to(dtype=p.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                prev_grad = prev_grad.add(grad)
                # Calculate cₜ (gradient with correction term)
                c_t = prev_grad.mul(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p_fp32, c_t, adaptive_clip, adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = grad_diff_fim.to(torch.float32)

                    if group['step'] == 1:
                        grad_diff_fim.addcmul_(grad_diff, grad_diff).clamp_(-clip_lambda, clip_lambda)
                        diff_fim_base = 1.0
                    else:
                        # Get natural gradient (squared ema, obtained sqrt of ema)
                        diff_fim_base = grad_diff_fim.sqrt().add_(curr_eps)

                        grad_diff_fim.mul_(current_beta3).addcmul_(grad_diff, grad_diff, value=1.0 - current_beta3).clamp_(-clip_lambda, clip_lambda)
                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                if group['step'] == 1:
                    fim.addcmul_(approx_grad_nat, approx_grad_nat)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)

                    grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_nat.div_(divisor)

                    momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (momentum_cent * grad_nat > 0).to(grad_nat.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            momentum_cent = momentum_cent * mask
                        elif group['update_strategy'] == 'grams':
                            momentum_cent = torch.sign(grad_nat) * momentum_cent.abs()

                    # Compass-style amplification
                    full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                    # center the gradient vector
                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                                centralization
                            )
                        )

                    if stable_weight_decay and group['fim_mean_sqrt'] is not None:
                        swd_scaling = 1.0 / group['fim_mean_sqrt']
                    else:
                        swd_scaling = 1.0

                    # Perform weight decay
                    if weight_decay != 0 and weight_decouple:
                        p_fp32.mul_(1.0 - weight_decay * lr * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = p_fp32.data.div(fim_base).div_(diff_fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                        p_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * swd_scaling)

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            full_step = full_step * mask
                        elif group['update_strategy'] == 'grams':
                            full_step.copy_(torch.sign(grad_nat) * full_step.abs())

                    # Apply full step
                    p_fp32.data.add_(full_step, alpha=-step_size)

                    fim.mul_(current_beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - current_beta2).clamp_(-clip_lambda, clip_lambda)

                if stable_weight_decay:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss

class FMARSCropV2(BaseOptimizer):
    r"""
    FMARSCropV2: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with customized ADOPT AdamW changes (https://arxiv.org/abs/2411.02853), and cautious stepping.
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Intended to arrive at the minima faster and in a more stable manner than FMARSCrop_ExMachina and V1
    Thanks to Machina for introducing the usage of stochastic rounding, adaptive_eps, and further testing!
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps. If 0, round to 1e-36).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.01).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA - https://arxiv.org/abs/2207.09066 (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification, adds another memory state (slightly increased VRAM usage) (default: 0.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.99) (Alternative recommendation: 0.9999).
        momentum_lambda (float):
            Amplification exponent for slow momentum / EMA (default: 0.1) (Alternative recommendation: 0.25).
        gamma (float):
            Scaling parameter for gradient correction for MARS - https://arxiv.org/abs/2411.10438 (default: 0.001).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0).
        adaptive_clip (float):
            Adaptive clip value to apply to the corrected gradient, before further use by the optimizer. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True).
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.01,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 0.0,
        momentum_beta: float = 0.99,
        momentum_lambda: float = 0.1,
        gamma: float = 0.001,
        clip: float = 1.0,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        cautious: bool = True,
        debias_beta2: bool = True,
    ):

        # Override zero to 1e-36, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-36

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            eps2 = eps2,
            eps_floor = eps_floor,
            weight_decay = weight_decay,
            centralization = centralization,
            moment_centralization = moment_centralization,
            diff_mult = diff_mult,
            momentum_beta = momentum_beta,
            momentum_lambda = momentum_lambda,
            gamma = gamma,
            clip = clip,
            adaptive_clip = adaptive_clip,
            adaptive_clip_eps = adaptive_clip_eps,
            cautious = cautious,
            debias_beta2 = debias_beta2,
        )

        super(FMARSCropV2, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCropV2'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            gamma = group["gamma"]
            clip = group["clip"]
            step = group['step']
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            debias_beta2 = group["debias_beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.to(dtype=p.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                state = self.state[p]

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                prev_grad = prev_grad.add(grad)

                # Calculate cₜ (gradient with correction term)
                correction = (gamma * (beta1 / (1 - beta1))) * prev_grad
                c_t = grad + correction

                # Gradient clipping (if necessary)
                if group["adaptive_clip"] > 0.0:
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=group["adaptive_clip"], agc_eps=group["adaptive_clip_eps"], norm_type='layer')

                clip_lambda = step**0.25

                if debias_beta2:
                    fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** (1/2)
                else:
                    fim_slow_beta = beta2

                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = grad_diff_fim.sqrt().add(curr_eps)

                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(-clip_lambda, clip_lambda)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                fim_base = fim.sqrt().add(curr_eps)

                grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                # Compass-style amplification
                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum) * moment_centralization
                else:
                    momentum_cent = momentum
                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (momentum_cent * grad_nat < 0).to(grad_nat.dtype) # Unsure if disagreement masking is more useful than agreement masking, or not masking at all.
                    mask.div_(mask.mean().clamp_(min=1e-3)) #                       It should theoretically help prevent poor updates?
                    momentum_cent = momentum_cent * mask
                full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                # center the gradient vector
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )
                
                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p_fp32.data.div(fim_base).div_(diff_fim_base)

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_weights.div_(divisor)

                    p_fp32.data.add_(grad_weights, alpha=-lr*weight_decay)

                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step * mask
                p_fp32.data.add_(full_step, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(-clip_lambda, clip_lambda)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)
        return loss
    
class FMARSCropV3(Optimizer):
    r"""
    FMARSCropV3: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with customized ADOPT AdamW changes (https://arxiv.org/abs/2411.02853), and cautious stepping.
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Intended to arrive at the minima faster and in a more stable manner than FMARSCrop_ExMachina and V1
    Thanks to Machina for introducing the usage of stochastic rounding, adaptive_eps, and further testing!
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running average of
            momentum and the FIM running average (default: 0.99, 0.95).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps. If 0, round to 1e-30).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.01).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA - https://arxiv.org/abs/2207.09066 (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification, adds another memory state (slightly increased VRAM usage) (default: 0.0).
        momentum_lambda (float):
            Amplification factor for slow momentum / EMA (default: 2.0).
        gamma (float):
            Scaling parameter for gradient correction for MARS - https://arxiv.org/abs/2411.10438 (default: 0.05).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0).
        adaptive_clip (float):
            Adaptive clip value to apply to the corrected gradient, before further use by the optimizer. (default: 0.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updating due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_norm_type (bool):
            Whether or not to use the unit norm (default: 1) or the norm of the whole grad (0) for adaptive clipping.
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.99, 0.95),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.01,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 0.0,
        momentum_lambda: float = 2.0,
        gamma: float = 0.05,
        clip_lambda: float = 1.0,
        adaptive_clip: float = 0.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_norm_type: bool = 1,
        cautious: bool = True,
    ):

        # Override zero to 1e-30, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-30

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            eps2 = eps2,
            eps_floor = eps_floor,
            weight_decay = weight_decay,
            centralization = centralization,
            moment_centralization = moment_centralization,
            diff_mult = diff_mult,
            momentum_lambda = momentum_lambda,
            gamma = gamma,
            clip_lambda = clip_lambda,
            adaptive_clip = adaptive_clip,
            adaptive_clip_eps = adaptive_clip_eps,
            adaptive_clip_norm_type = adaptive_clip_norm_type,
            cautious = cautious,
        )

        super(FMARSCropV3, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_lambda = group["momentum_lambda"]
            gamma = group["gamma"]
            clip_lambda = group["clip_lambda"]
            step = group['step']
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.clone().to(p.dtype).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                state = self.state[p]

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["prev_grad"].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                prev_grad = prev_grad.add(grad)

                # Calculate cₜ (gradient with correction term)
                correction = gamma * beta2 / (1 - beta2) * prev_grad
                c_t = grad + correction

                # Gradient clipping (if necessary)
                if group["adaptive_clip"] > 0.0:
                    c_t = agc_global_norm(p_fp32, c_t, group["adaptive_clip_eps"], group["adaptive_clip"], unit_norm=group["adaptive_clip_norm_type"])
                grad_norm = torch.linalg.norm(c_t)
                if grad_norm > clip_lambda:
                    c_t = c_t * clip_lambda / grad_norm

                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip_lambda, rms) / clip_lambda
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = torch.clamp(grad_diff_fim.sqrt(), curr_eps)

                    grad_diff_fim.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip_lambda, rms) / clip_lambda
                approx_grad_nat.div_(divisor)

                fim_base = torch.clamp(fim.sqrt(), curr_eps)

                grad_nat = c_t.div(fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip_lambda, rms) / clip_lambda
                grad_nat.div_(divisor)

                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # Compass-style amplification
                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum) * moment_centralization
                else:
                    momentum_cent = momentum
                # Apply full step
                full_step = grad_nat.add(momentum_cent, alpha=momentum_lambda)

                # center the gradient vector
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )
                
                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p_fp32.data.div(fim_base)

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip_lambda, rms) / clip_lambda
                    grad_weights.div_(divisor)

                    full_step.add_(grad_weights, alpha=weight_decay)

                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (full_step * c_t > 0).to(full_step.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step * mask
                p_fp32.data.add_(full_step, alpha=-lr)

                fim.mul_(beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - beta2)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)
        return loss
    
def get_rms(tensor, eps=1e-8):
    return tensor.norm().div(tensor.numel() ** 0.5).clamp_min(eps)

class FMARSCropV3ExMachina(BaseOptimizer):
    r"""
    FMARSCropV3ExMachina: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with ADOPT's AdamW changes (https://arxiv.org/abs/2411.02853).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            coefficients used for computing running averages of momentum,
            approx. natural grad FIM, and gradient difference FIM (default: 0.99, 0.9999, 0.999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_lambda (float):
            The lambda value for slow momentum / EMA, controlling how much the momentum is amplified while being added to the update. (default: 2.0).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool) (deprecated, use update strategy)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
            Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
            and 'grams' (https://arxiv.org/abs/2412.17107) (default: cautious)
        adaptive_clip (float):
            Adaptive clip value to applied to the MARS corrected gradient. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.0005)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: False)
        debias_beta2 (bool):
            Apply bias correction to fim. (Default: True)
        debias_beta3 (bool):
            Apply bias correction to diff fim. (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: Betas = (0.99, 0.95),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 2.0,
        clip: float = 1.0,
        cautious: bool = False,
        gamma: float = 0.005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'global',
        stable_weight_decay: bool = False,
        debias_beta1: bool = False,
        debias_beta2: bool = False,
        update_strategy: UPDATE_STRATEGY = 'cautious',
        stable_update: bool = False,
        atan2_denom: bool = False,
        use_orthograd: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams','both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
            'weight_decay':weight_decay,
            'centralization':centralization,
            'moment_centralization':moment_centralization,
            'diff_mult':diff_mult,
            'momentum_lambda':momentum_lambda,
            'clip':clip,
            'cautious':cautious,
            'gamma': gamma,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'stable_weight_decay': stable_weight_decay,
            'debias_beta1':debias_beta1,
            'debias_beta2':debias_beta2,
            'weight_decouple':weight_decouple,
            'update_strategy': update_strategy,
            'stable_update': stable_update,
            'atan2_denom': atan2_denom,
            'use_orthograd': use_orthograd,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCropV3ExMachina'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = None
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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
                group['fim_mean_sqrt'] = None

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            stable_weight_decay = group["stable_weight_decay"]
            weight_decouple = group['weight_decouple']

            bias_correction1: float = self.debias(beta1, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if stable_weight_decay:
                    param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.to(dtype=p.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                prev_grad = prev_grad.add(grad)
                # Calculate cₜ (gradient with correction term)
                c_t = prev_grad.mul(gamma * (beta2 / (1.0 - beta2))).add_(grad)

                if group["use_orthograd"]:
                    _paper_orthograd(p_fp32, c_t)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p_fp32, c_t, adaptive_clip, adaptive_clip_eps, norm_type=adaptive_clip_type)
                    
                grad_norm = torch.linalg.norm(c_t)
                if grad_norm > 1.0:
                    c_t = c_t * 1.0 / grad_norm

                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = grad_diff_fim.to(torch.float32)

                    if group['step'] == 1:
                        grad_diff_fim.addcmul_(grad_diff, grad_diff)
                        diff_fim_base = torch.tensor(1.0)
                    else:
                        # Get natural gradient (squared ema, obtained sqrt of ema)
                        diff_fim_base = grad_diff_fim.sqrt()

                        grad_diff_fim.mul_(current_beta2).addcmul_(grad_diff, grad_diff, value=1.0 - current_beta2)
                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = torch.tensor(1.0)

                if group["atan2_denom"]:
                    approx_grad_nat = c_t.atan2(diff_fim_base)
                else:
                    diff_fim_base.add_(curr_eps)
                    approx_grad_nat = c_t.div(diff_fim_base)

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                if group['step'] == 1:
                    fim.addcmul_(approx_grad_nat, approx_grad_nat)
                else:
                    fim_base = fim.sqrt()
                    if group["atan2_denom"]:
                        grad_nat = c_t.atan2(fim_base)
                    else:
                        fim_base.add_(curr_eps)
                        grad_nat = c_t.div(fim_base)

                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_nat.div_(divisor)

                    momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    # Compass-style amplification
                    full_step = grad_nat.add(momentum_cent, alpha=momentum_lambda)

                    # center the gradient vector
                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                                centralization
                            )
                        )

                    if stable_weight_decay and group['fim_mean_sqrt'] is not None:
                        swd_scaling = 1.0 / group['fim_mean_sqrt']
                    else:
                        swd_scaling = 1.0

                    # Perform weight decay
                    if weight_decay != 0 and weight_decouple:
                        p_fp32.mul_(1.0 - weight_decay * lr * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = p_fp32.data.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                        p_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * swd_scaling)

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (full_step * c_t > 0).to(c_t.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            full_step = full_step * mask
                        elif group['update_strategy'] == 'grams':
                            full_step.copy_(torch.sign(c_t) * full_step.abs())

                    if group["stable_update"]:
                        clip_threshold = 1
                        rms = get_rms(full_step, 1).div(clip_threshold).clamp_min(1)
                        full_step.mul_(1 / rms)

                    # Apply full step
                    p_fp32.data.add_(full_step, alpha=-step_size)

                    fim.mul_(beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - current_beta2)

                if stable_weight_decay:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss