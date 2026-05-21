import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_

class SGDSaI(BaseOptimizer):
    r"""No More Adam: Learning Rate Scaling at Initialization is All You Need.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float.  coefficients used for computing running averages of gradient.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        cautious: bool = False,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.has_warmup: bool = False

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'cautious': cautious,
            'eps': eps,
            'torch_compile': torch_compile,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDSaI'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                if group['momentum'] > 0.0:
                    state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def warmup_step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)

                sigma = grad.std().nan_to_num_()
                grad_norm = grad.norm()

                g_snr = grad_norm.div_(sigma.add_(group['eps'])) if sigma != 0.0 else grad_norm

                self.state[p]['gsnr'] = g_snr

        self.has_warmup = True

        return loss

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.has_warmup:
            self.warmup_step(closure)

        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum: float = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                p_fp32 = p

                state = self.state[p]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32,copy=True)

                if momentum > 0.0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = grad.clone()

                    momentum_buffer = state['momentum_buffer']

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        momentum_buffer = momentum_buffer.to(torch.float32)

                    momentum_buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                    # Pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state['momentum_buffer'], momentum_buffer)
                else:
                    momentum_buffer = grad

                apply_weight_decay(
                    p_fp32,
                    grad,
                    group['lr'],
                    group['weight_decay'],
                    group['weight_decouple'],
                    False,
                    torch_compile=group.get('torch_compile', False),
                )

                if group["cautious"] and momentum > 0.0:
                    mask = (momentum_buffer * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                p_fp32.add_(momentum_buffer * mask, alpha=-group['lr'] * state['gsnr'])

                # Pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss