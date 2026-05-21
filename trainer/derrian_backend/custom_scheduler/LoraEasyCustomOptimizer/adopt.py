# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adopt.py
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal

import torch
import math

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.base.exception import NoSparseGradientError
from .utils import copy_stochastic_, NORM_TYPE, agc, UPDATE_STRATEGY, adaptive_eps

class ADOPT(BaseOptimizer):
    r"""Modified Adam Can Converge with Any β2 with the Optimal Rate.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param clip: float. special form of clip for ADOPT, recommended and default value is 0.25.
    cautious (bool) (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
        and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        clip: float = 0.25,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'clip': clip,
            'cautious': cautious,
            'update_strategy':update_strategy,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPT'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                p_fp32 = p

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                 # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                 # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                if group['weight_decay'] != 0 and not group['weight_decouple']:
                    grad = grad.add(p_fp32, alpha=group['weight_decay'])

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    continue

                if group['weight_decay'] != 0 and group['weight_decouple']:
                    p_fp32.add_(p_fp32, alpha=-group['lr'] * group['weight_decay'])

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                normed_grad = grad.div(denom)
                if group['clip'] is not None:
                    clip = (group['step']-1)**group['clip']
                    normed_grad.clamp_(-clip, clip)

                exp_avg.lerp_(normed_grad, 1 - beta1)

                update = exp_avg.clone()

                if group['update_strategy'] in {'cautious','grams'}:
                    if group['update_strategy'] == 'cautious':
                        mask = (update * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = update * mask
                    elif group['update_strategy'] == 'grams':
                        update.copy_(torch.sign(grad) * update.abs())

                p_fp32.add_(update, alpha=-group['lr'])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss

class ADOPTMARS(BaseOptimizer):
    r"""ADOPT with MARS correction.
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
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
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
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        gamma: float = 0.025,
        cautious: bool = True,
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
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'gamma': gamma,
            'cautious': cautious,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTMARS'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_sq_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
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
                group['exp_avg_sq_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            lr: float = group['lr']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]

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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                exp_avg, exp_avg_sq, grad_diff = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    exp_avg, exp_avg_sq, grad_diff = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32), grad_diff.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                grad_diff.add_(grad)

                # MARS Calculate cₜ (gradient with correction term)
                correction = (gamma * (beta1 / (1.0 - beta1))) * grad_diff
                c_t = grad + correction

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(c_t, c_t.conj())
                else:
                    de_nom = exp_avg_sq.sqrt_().add_(curr_eps)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t.conj(), value=1 - beta2)

                    normed_grad = grad.div(de_nom)
                    normed_grad.clamp_(-adopt_clip, adopt_clip)

                    exp_avg.lerp_(normed_grad, 1.0 - beta1)

                    update = exp_avg.clone()

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (update * normed_grad > 0).to(c_t.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    p_fp32.add_(update * mask, alpha=-lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(state['previous_grad'], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss

class FADOPTMARS(BaseOptimizer):
    r"""Fisher ADOPT with MARS correction.
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
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
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
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
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
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTMARS'
    
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

            lr: float = group['lr']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]
            gamma = group["gamma"]

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
                    fim.mul_(beta2).addcmul_(c_t, c_t.conj(), value=1 - beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = c_t.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    momentum.lerp_(grad_nat, 1.0 - beta1)

                    update = momentum.clone()

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

                    p_fp32.add_(update * mask, alpha=-lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['momentum'], momentum)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
    