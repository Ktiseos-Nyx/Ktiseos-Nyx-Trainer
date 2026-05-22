from typing import Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm
from .utils import apply_weight_decay, copy_stochastic_


class Lamb(BaseOptimizer):
    r"""Large Batch Optimization for Deep Learning.

        This Lamb implementation is based on the paper v3, which does not use de-biasing.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param rectify: bool. perform the rectified update similar to RAdam.
    :param degenerated_to_sgd: bool. degenerated to SGD.
    :param n_sma_threshold: int. (recommended is 5).
    :param grad_averaging: bool. whether apply (1 - beta2) to gradient when calculating running averages of gradient.
    :param max_grad_norm: float. max gradient norm to clip.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param adam: bool. always use trust ratio = 1, which turns this into Adam. Useful for comparison purposes.
    :param pre_norm: bool. perform pre-normalization of all gradients.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        degenerated_to_sgd: bool = False,
        n_sma_threshold: int = 5,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        pre_norm: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-6,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.degenerated_to_sgd = degenerated_to_sgd
        self.n_sma_threshold = n_sma_threshold
        self.pre_norm = pre_norm

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'grad_averaging': grad_averaging,
            'max_grad_norm': max_grad_norm,
            'adam': adam,
            'adanorm': adanorm,
            'adam_debias': adam_debias,
            'eps': eps,
            'torch_compile': torch_compile,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lamb'
    
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

        grad_norm = 1.0
        if self.pre_norm:
            grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            beta3: float = 1.0 - beta1 if group['grad_averaging'] else 1.0
            bias_correction1: float = self.debias(beta1, group['step'])

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
                
                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)

                if self.pre_norm:
                    grad.div_(grad_norm)

                state = self.state[p]
                p_fp32 = p
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_grad_norm = state.get('exp_grad_norm', None)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                    if exp_grad_norm:
                        exp_grad_norm = exp_grad_norm.to(torch.float32)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=exp_grad_norm,
                    r=group.get('r', None),
                )

                exp_avg.mul_(beta1).add_(s_grad, alpha=beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                apply_weight_decay(
                    p=p_fp32,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    torch_compile=group.get('torch_compile', False),
                )

                if group['rectify']:
                    update = p_fp32.clone()
                    if n_sma >= self.n_sma_threshold:
                        de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                        update.addcdiv_(exp_avg, de_nom, value=-step_size)
                    else:
                        update.add_(exp_avg, alpha=-step_size)
                else:
                    update = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])

                weight_norm = torch.linalg.norm(p_fp32).clamp_(min=0, max=self.clamp)
                p_norm = torch.linalg.norm(update)
                trust_ratio: float = 1.0 if weight_norm == 0 or p_norm == 0 else weight_norm / (p_norm + group['eps'])

                # WHY????
                #state['weight_norm'] = weight_norm
                #state['adam_norm'] = p_norm
                #state['trust_ratio'] = trust_ratio

                if group['adam']:
                    trust_ratio = 1.0

                if group['rectify']:
                    if n_sma >= self.n_sma_threshold:
                        p_fp32.addcdiv_(exp_avg, de_nom, value=-step_size * trust_ratio)
                    else:
                        p_fp32.add_(exp_avg, alpha=-step_size * trust_ratio)
                else:
                    p_fp32.add_(update, alpha=-step_size * trust_ratio)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    if exp_grad_norm:
                        copy_stochastic_(state['exp_grad_norm'], exp_grad_norm)

        return loss