import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from .utils import copy_stochastic_, debias_beta



class RACS(BaseOptimizer):
    r"""Row and Column Scaled SGD.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. momentum factor.
    :param alpha: float. scaler.
    :param gamma: float. limiter threshold.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 0.01,
        beta: float = 0.9,
        alpha: float = 0.02,
        gamma: float = 1.01,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        adam_lr: float = 5e-4,
        adam_betas: Betas = (0.9, 0.999),
        adam_weight_decay: float = 0.0,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adam_lr)
        self.validate_betas(adam_betas)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(adam_weight_decay, 'adam_weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: Defaults = {
            'lr': lr,
            '_lr_ratio': (adam_lr / lr) if lr > 0 else 0,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'adam_lr': adam_lr,
            'adam_betas': adam_betas,
            'adam_weight_decay': adam_weight_decay,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RACS'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

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

            beta = group['beta']
            adam_betas = group['adam_betas']

            beta1_comp = 1 - debias_beta(adam_betas[0], group['step'])
            beta2_hat = debias_beta(adam_betas[1], group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                # RACS doesn't support scalars or dim > 2
                # Fallback to AdamW
                if p.ndim == 0 or p.ndim > 2:
                    p_fp32 = p

                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
  
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad = grad.to(torch.float32)
                        p_fp32 = p.to(torch.float32)
                        exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                    # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
                    if group['adam_weight_decay']:
                        if group['weight_decouple']:
                            p_fp32.mul_(group['adam_weight_decay'])
                        else:
                            grad.add_(p_fp32, alpha=group['adam_weight_decay'])

                    # update gradient moving averages with debiased betas
                    exp_avg.lerp_(grad, weight=beta1_comp)
                    exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1 - beta2_hat)

                    # Adam step
                    p_fp32.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-group['lr'] * group['_lr_ratio'])

                    copy_stochastic_(state['exp_avg'], exp_avg)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)
                    continue

                if len(p.shape) == 1:
                    p = p.unsqueeze(0)  # noqa: PLW2901
                    grad = grad.unsqueeze(0)

                if len(state) == 0:
                    state['s'] = torch.zeros(p.size(0), dtype=p.dtype, device=p.device)
                    state['q'] = torch.ones(p.size(1), dtype=p.dtype, device=p.device)
                    state['theta'] = torch.zeros((), dtype=grad.dtype, device=grad.device)

                s, q = state['s'], state['q']

                p_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    s, q = s.to(torch.float32), q.to(torch.float32)

                # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
                if group['weight_decay']:
                    if group['weight_decouple']:
                        p_fp32.mul_(group['weight_decay'])
                    else:
                        grad.add_(p_fp32, alpha=group['weight_decay'])

                grad_p2 = grad.pow(2)
                s.mul_(beta).add_(grad_p2.mean(dim=1), alpha=1.0 - beta)
                q.mul_(beta).add_(grad_p2.mean(dim=0), alpha=1.0 - beta)

                s_sq = s.add(group['eps']).sqrt_().unsqueeze(1)
                q_sq = q.add(group['eps']).sqrt_().unsqueeze(0)

                grad_hat = grad / (s_sq * q_sq)

                grad_hat_norm = torch.norm(grad_hat)
                threshold = (
                    group['gamma'] / max(grad_hat_norm / (state['theta'] + group['eps']), group['gamma'])
                    if group['step'] > 1
                    else 1.0
                )
                copy_stochastic_(state['theta'], grad_hat_norm.mul(threshold))

                p_fp32.add_(grad_hat, alpha=-group['lr'] * group['alpha'] * threshold)

                copy_stochastic_(state['s'], s)
                copy_stochastic_(state['q'], q)
                copy_stochastic_(p, p_fp32)

        return loss
