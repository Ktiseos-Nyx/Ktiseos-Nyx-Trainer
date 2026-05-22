import math
from typing import Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm
from .utils import copy_stochastic_, UPDATE_STRATEGY


class Adan(BaseOptimizer):
    r"""Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param max_grad_norm: float. max gradient norm to clip.
    :param use_gc: bool. use gradient centralization.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param eps: float. term added to the denominator to improve numerical stability.
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
        betas: Betas = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        max_grad_norm: float = 0.0,
        use_gc: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        eps: float = 1e-8,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        self.max_grad_norm = max_grad_norm
        self.use_gc = use_gc

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'max_grad_norm': max_grad_norm,
            'adanorm': adanorm,
            'eps': eps,
            'cautious': cautious,
            'update_strategy': update_strategy,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adan'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_global_grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2, beta3 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction3_sq: float = math.sqrt(self.debias(beta3, group['step']))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = grad.clone().mul_(-clip_global_grad_norm)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                p_fp32 = p
                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']

                grad_diff = state['previous_grad']

                exp_grad_norm = state.get('exp_grad_norm', None)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)
                    grad_diff = grad_diff.to(torch.float32)
                    exp_avg, exp_avg_sq, exp_avg_diff = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32), exp_avg_diff.to(torch.float32)

                    if exp_grad_norm:
                        exp_grad_norm = exp_grad_norm.to(torch.float32)

                grad.mul_(clip_global_grad_norm)

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                grad_diff.add_(grad)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=exp_grad_norm,
                    r=group.get('r', None),
                )

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)

                grad_diff.mul_(beta2).add_(grad)
                exp_avg_sq.mul_(beta3).addcmul_(grad_diff, grad_diff, value=1.0 - beta3)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction3_sq).add_(group['eps'])

                if group['weight_decouple']:
                    p_fp32.mul_(1.0 - group['lr'] * group['weight_decay'])

                if group['update_strategy'] in {'cautious','grams'}:
                    if group['update_strategy'] == 'cautious':
                        exp_avg_mask = (exp_avg * grad > 0).to(grad.dtype)
                        exp_avg_mask.div_(exp_avg_mask.mean().clamp_(min=1e-3))
                    elif group['update_strategy'] == 'grams':
                        exp_avg_upd = torch.sign(grad) * exp_avg.abs()
                        exp_avg_mask = 1.0
                else:
                    exp_avg_upd = exp_avg
                    exp_avg_mask = 1.0

                p_fp32.addcdiv_(exp_avg_upd * exp_avg_mask, de_nom, value=-group['lr'] / bias_correction1)

                if group['update_strategy'] in {'cautious','grams'}:
                    if group['update_strategy'] == 'cautious':
                        exp_avg_diff_mask = (exp_avg_diff * grad > 0).to(grad.dtype)
                        exp_avg_diff_mask.div_(exp_avg_diff_mask.mean().clamp_(min=1e-3))
                    elif group['update_strategy'] == 'grams':
                        exp_avg_diff_upd = torch.sign(grad) * exp_avg_diff.abs()
                        exp_avg_diff_mask = 1.0
                else:
                    exp_avg_diff_upd = exp_avg_diff
                    exp_avg_diff_mask = 1.0

                p_fp32.addcdiv_(exp_avg_diff_upd * exp_avg_diff_mask, de_nom, value=-group['lr'] * beta2 / bias_correction2)

                if not group['weight_decouple']:
                    p_fp32.div_(1.0 + group['lr'] * group['weight_decay'])

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(state['exp_avg_diff'], exp_avg_diff)
                    copy_stochastic_(state['previous_grad'], -grad)
                    if exp_grad_norm:
                        copy_stochastic_(state['exp_grad_norm'], exp_grad_norm)
                else:
                    state['previous_grad'].copy_(-grad)

        return loss