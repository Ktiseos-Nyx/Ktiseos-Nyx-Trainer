import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_


class LaProp(BaseOptimizer):
    r"""Separating Momentum and Adaptivity in Adam.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param centered: bool.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param cautious: bool. whether to use the Cautious variant.
    :param eps: float. epsilon value.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 4e-4,
        betas: Betas = (0.9, 0.999),
        centered: bool = False,
        steps_before_using_centered: int = 10,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        cautious: bool = False,
        eps: float = 1e-15,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.cautious = cautious
        self.steps_before_using_centered: int = steps_before_using_centered

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'centered': centered,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
            'torch_compile': torch_compile,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'LaProp'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_lr_1'] = 0.0
            group['exp_avg_lr_2'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                if group['centered']:
                    state['exp_mean_avg_beta2'] = torch.zeros_like(p)
                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

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
                group['exp_avg_lr_1'] = 0.0
                group['exp_avg_lr_2'] = 0.0

            beta1, beta2 = group['betas']

            group['exp_avg_lr_1'] = group['exp_avg_lr_1'] * beta1 + (1.0 - beta1) * group['lr']
            group['exp_avg_lr_2'] = group['exp_avg_lr_2'] * beta2 + (1.0 - beta2)

            bias_correction1: float = group['exp_avg_lr_1'] / group['lr'] if group['lr'] != 0.0 else 1.0
            bias_correction2: float = group['exp_avg_lr_2']
            step_size: float = 1.0 / bias_correction1

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

                    if group['centered']:
                        state['exp_mean_avg_beta2'] = torch.zeros_like(p)
                    if group['ams_bound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                
                p_fp32 = p
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq
                if group['centered']:
                    exp_mean_avg_beta2 = state['exp_mean_avg_beta2']

                    # unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        exp_mean_avg_beta2 = exp_mean_avg_beta2.to(torch.float32)

                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                    if group['step'] > self.steps_before_using_centered:
                        de_nom -= exp_mean_avg_beta2.pow(2)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["exp_mean_avg_beta2"], exp_mean_avg_beta2)

                if group['ams_bound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                    # unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        max_exp_avg_sq = max_exp_avg_sq.to(torch.float32)

                        if not (group['centered'] and state['step'] <= self.steps_before_using_centered): 
                            # Maintains the maximum of all (centered) 2nd moment running avg. till now
                            torch.max(max_exp_avg_sq, de_nom, out=max_exp_avg_sq)
                            # Use the max. for normalizing running avg. of gradient
                            de_nom = max_exp_avg_sq

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["max_exp_avg_sq"], max_exp_avg_sq)

                    de_nom = de_nom.div(bias_correction2).sqrt_().add_(group['eps'])

                exp_avg.mul_(beta1).addcdiv_(grad, de_nom, value=(1.0 - beta1) * group['lr'])

                if self.cautious:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                p_fp32.add_(exp_avg * mask, alpha=-step_size)

                apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    torch_compile=group.get('torch_compile', False),
                )

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss