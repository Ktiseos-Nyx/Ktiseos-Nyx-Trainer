# Authored originally by: https://github.com/kozistr
import math
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_, UPDATE_STRATEGY, _paper_orthograd, SSCCosineDecay
from typing import Optional

class StableSPAM(BaseOptimizer):
    r"""How to Train in 4-Bit More Stably than 16-Bit Adam.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param gamma1: float.
    :param gamma2: float.
    :param gamma3: float.
    :param t_max: Optional[int]. total number of steps.
    :param eta_min: float. eta_min of CosineDecay.
    :param weight_decay: float. weight decay (L2 penalty).
    :param update_proj_gap: int. update projection gap.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        gamma1: float = 0.85,
        gamma2: float = 0.99999,
        gamma3: float = 0.999,
        t_max: Optional[int] = None,
        eta_min: float = 0.5,
        weight_decay: float = 0.0,
        update_proj_gap: int = 1000,
        eps: float = 1e-8,
        use_orthograd: bool = False,
        use_adopt: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        # Override zero to tiny
        if eps <= 0:
            eps = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        self.gamma1: float = betas[0] if gamma1 == -1.0 else gamma1
        self.gamma2: float = gamma2
        self.gamma3: float = gamma3
        self.t_max = t_max
        self.update_proj_gap = update_proj_gap
        self.warmup = SSCCosineDecay(1.0, t_max, eta_min=eta_min) if t_max is not None else None

        self.total_step: int = 0

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
            'use_orthograd': use_orthograd,
            'use_adopt':use_adopt,
            'update_strategy': update_strategy,
            'torch_compile': torch_compile,
            **kwargs}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableSPAM'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['m_norm_t'] = 0.0
                state['v_norm_t'] = 0.0
                state['m_max_t'] = 0.0
                state['step'] = 0

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_step += 1
        scale: float = self.warmup.get_death_rate(self.total_step) if self.warmup is not None else 1.0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta1 *= scale

            eps = group['eps']
            use_orthograd = group['use_orthograd']
            use_adopt = group['use_adopt']
            update_strategy  = group['update_strategy']

            for p in group['params']:
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['m_norm_t'] = 0.0
                    state['v_norm_t'] = 0.0
                    state['m_max_t'] = 0.0
                    state['step'] = 0

                state['step'] += 1

                adopt_clip: float = (state['step']-1)**0.25

                exp_avg, exp_avg_sq, m_max_t = state['exp_avg'], state['exp_avg_sq'], state['m_max_t']

                if p.dtype == torch.bfloat16:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                if use_orthograd:
                    _paper_orthograd(p_fp32, grad)

                apply_weight_decay(
                    p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                    torch_compile=group.get('torch_compile', False),
                )

                max_grad = torch.max(grad.abs())

                m_max_t = self.gamma3 * m_max_t + (1 - self.gamma3) * max_grad

                state["m_max_t"] = m_max_t

                m_max_hat = m_max_t / (1.0 - self.gamma3 ** state['step'])

                mask = grad.abs() > m_max_hat
                if mask.sum() > 0:
                    grad[mask] = grad[mask] / max_grad * m_max_hat

                grad_norm = torch.norm(grad)

                m_norm_t, v_norm_t = state['m_norm_t'], state['v_norm_t']

                m_norm_t = self.gamma1 * scale * m_norm_t + (1 - self.gamma1 * scale) * grad_norm
                v_norm_t = self.gamma2 * v_norm_t + (1 - self.gamma2) * grad_norm**2

                state["m_norm_t"], state["v_norm_t"] = m_norm_t, v_norm_t

                m_norm_hat = m_norm_t / (1.0 - (self.gamma1 * scale) ** state['step'])
                v_norm_hat = v_norm_t / (1.0 - self.gamma2 ** state['step'])

                c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

                if grad_norm > 0:
                    grad = grad / grad_norm * c_norm_t

                if self.update_proj_gap > 0 and self.total_step % self.update_proj_gap == 0:
                    exp_avg = torch.zeros_like(grad)
                    exp_avg_sq = torch.zeros_like(grad)
                    state['step'] = 1

                bias_correction1: float = self.debias(beta1, state['step'])
                bias_correction2: float = self.debias(beta2, state['step'])
                bias_correction2_sq: float = math.sqrt(bias_correction2)

                if use_adopt:
                    step_size: float = group['lr']
                else:
                    step_size: float = group['lr'] / bias_correction1

                if use_adopt and state['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    if use_adopt:
                        de_nom = exp_avg_sq.sqrt().add_(eps)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    else:   
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(eps)

                    update = exp_avg.div(de_nom)

                    if use_adopt:
                        update.clamp_(-adopt_clip, adopt_clip)

                    if update_strategy in {'cautious','grams','both'}:
                        if update_strategy in {'cautious','both'}:
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update.mul_(mask)
                        if update_strategy in {'grams','both'}:
                            update.copy_(torch.sign(grad) * update.abs())

                    p_fp32.add_(update, alpha=-step_size)

                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss