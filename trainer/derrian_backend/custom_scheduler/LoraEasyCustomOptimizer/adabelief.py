# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adabelief.py
import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_


class AdaBelief(BaseOptimizer):
    r"""Adapting Step-sizes by the Belief in Observed Gradients.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param rectify: bool. perform the rectified update similar to RAdam.
    :param n_sma_threshold: number of SMA threshold (recommended is 5).
    :param degenerated_to_sgd: bool. perform SGD update when variance of gradient is high.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param cautious: bool: Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        ams_bound: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-16,
        cautious: bool = False,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'ams_bound': ams_bound,
            'adanorm': adanorm,
            'adam_debias': adam_debias,
            'eps': eps,
            'cautious': cautious,
            'torch_compile': torch_compile,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaBelief'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['ams_bound']:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group['rectify'],
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=group['adam_debias'],
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    if group['ams_bound']:
                        state['max_exp_avg_var'] = torch.zeros_like(p)

                apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    torch_compile=group.get('torch_compile', False),
                )

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                exp_grad_norm = state.get('exp_grad_norm', None)
                max_exp_avg_var = state.get('max_exp_avg_var', None)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg, exp_avg_var = exp_avg.to(torch.float32), exp_avg_var.to(torch.float32)
                    if group['adanorm']:
                        exp_grad_norm = exp_grad_norm.to(torch.float32)
                    if group['ams_bound']:
                        max_exp_avg_var = max_exp_avg_var.to(torch.float32)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=exp_grad_norm,
                    r=group.get('r', None),
                )

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)

                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2).add_(group['eps'])

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_var,
                    max_exp_avg_sq=max_exp_avg_var,
                    eps=group['eps'],
                )

                if group["cautious"]:
                    # compute norm gradient
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                if not group['rectify']:
                    de_nom.div_(bias_correction2_sq)
                    p_fp32.addcdiv_(exp_avg * mask, de_nom, value=-step_size)
                    continue

                if n_sma >= self.n_sma_threshold:
                    p_fp32.addcdiv_(exp_avg * mask, de_nom, value=-step_size)
                elif step_size > 0:
                    p_fp32.add_(exp_avg * mask, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    copy_stochastic_(state['exp_avg_var'], exp_avg_var)
                    if group['adanorm']:
                        copy_stochastic_(state['exp_grad_norm'], exp_grad_norm)
                    if group['ams_bound']:
                        copy_stochastic_(state['max_exp_avg_var'], max_exp_avg_var)
                    copy_stochastic_(p, p_fp32)

        return loss