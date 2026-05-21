# https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/sgd.py
import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_


class VSGD(BaseOptimizer):
    r"""Variational Stochastic Gradient Descent for Deep Neural Networks. https://arxiv.org/abs/2404.06549

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param ghattg: float. prior variance ratio between ghat and g, Var(ghat_t-g_t)/Var(g_t-g_{t-1}).
    :param ps: float. prior strength.
    :param tau1: float. remember rate for the gamma parameters of g.
    :param tau2: float. remember rate for the gamma parameter of ghat.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-1,
        ghattg: float = 30.0,
        ps: float = 1e-8,
        tau1: float = 0.81,
        tau2: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(ghattg, 'ghattg')
        self.validate_non_negative(ps, 'ps')
        self.validate_non_negative(tau1, 'tau1')
        self.validate_non_negative(tau2, 'tau2')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'tau1': tau1,
            'tau2': tau2,
            'pa2': 2.0 * ps + 1.0 + 1e-4,
            'pbg2': 2.0 * ps,
            'pbhg2': 2.0 * ghattg * ps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'torch_compile': torch_compile,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'VSGD'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @torch.no_grad()
    def reset(self):
        pass

    def init_group(self, group, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['mug'] = torch.zeros_like(p)
                state['bg'] = torch.zeros_like(p)
                state['bhg'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            pa2, pbg2, pbhg2 = group['pa2'], group['pbg2'], group['pbhg2']

            rho1: float = math.pow(group['step'], -group['tau1'])
            rho2: float = math.pow(group['step'], -group['tau2'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                p_fp32 = p
                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    p_fp32 = p.to(torch.float32)
                    grad = grad.to(torch.float32)

                apply_weight_decay(
                    p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                    torch_compile=group.get('torch_compile', False),
                )

                bg, bhg, mug = state['bg'], state['bhg'], state['mug']

                p_fp32 = p
                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    bg = bg.to(torch.float32)
                    bhg = bhg.to(torch.float32)
                    mug = mug.to(torch.float32)

                if group['step'] == 1:
                    sg = pbg2 / (pa2 - 1.0)
                    shg = pbhg2 / (pa2 - 1.0)
                else:
                    sg = bg / pa2
                    shg = bhg / pa2

                mug_prev = mug.clone()

                mug.mul_(shg).add_(grad * sg).div_(sg + shg)

                sigg = (sg * shg) / (sg + shg)
                mug_sq = mug.pow(2).add_(sigg)

                bg2 = pbg2 + mug_sq - 2.0 * mug * mug_prev + mug_prev.pow(2)
                bhg2 = pbhg2 + mug_sq - 2.0 * grad * mug + grad.pow(2)

                bg.mul_(1.0 - rho1).add_(bg2, alpha=rho1)
                bhg.mul_(1.0 - rho2).add_(bhg2, alpha=rho2)

                p_fp32.add_(group['lr'] / mug_sq.sqrt().add_(group['eps']) * mug, alpha=-1.0)

                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["bg"], bg)
                    copy_stochastic_(state["bhg"], bhg)
                    copy_stochastic_(state["mug"], mug)
                    copy_stochastic_(p, p_fp32)

        return loss