# Authored by: https://github.com/kozistr
import math
from enum import IntEnum

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_, UPDATE_STRATEGY, NORM_TYPE, _paper_orthograd, agc, _get_compiled_stable_spam_clipping, _stable_spam_clipping_impl, SSCCosineDecay, newton_schulz_
from typing import Dict, Optional

class LMONorm(IntEnum):
    r"""normalization types."""

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:
    r"""Base class to perform norm onto Scion. This class does no norm."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        r"""Initialize parameter."""
        return x

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get LMO."""
        return grad


class Col(Norm):
    r"""col-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=0, keepdim=True)).mul_(math.sqrt(x.size(0)))
        if self.normalized:
            x.div_(x.size(1))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        d_in, d_out = grad.size()

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True)) / math.sqrt(d_in)
        if self.normalized:
            rms_value.mul_(d_out)

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class Row(Norm):
    r"""row-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=-1, keepdim=True))
        if self.normalized:
            x.div_(math.sqrt(x.size(-1)))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=-1, keepdim=True))
        if self.normalized:
            rms_value.mul_(math.sqrt(grad.size(-1)))

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class BiasRMS(Norm):
    r"""bias RMS."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.zeros_(x)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
        grad /= rms_value.add_(eps)
        return grad


class SpectralConv(Norm):
    r"""spectral-convolution normalization.

    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        d_out, d_in, kernel_size, *_ = x_fp64.size()

        for i in range(kernel_size):
            for j in range(kernel_size):
                torch.nn.init.orthogonal_(x_fp64[..., i, j])

        x_fp64.mul_(math.sqrt(d_out / d_in) / (kernel_size**2))

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        
        grad = newton_schulz_(grad, self.num_steps)

        d_out, d_in, kernel_size, *_ = grad.size()

        grad *= math.sqrt(d_out / d_in) / (kernel_size**2)

        return grad


class Spectral(Norm):
    r"""spectral normalization.

    :param max_scale: bool. set upper bound (1.0) of the scale.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        torch.nn.init.orthogonal_(x_fp64)

        d_out, d_in = x_fp64.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        x_fp64.mul_(scale)

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = newton_schulz_(grad, self.num_steps)

        d_out, d_in = grad.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        grad *= scale

        return grad


class Sign(Norm):
    r"""sign normalization.

    :param zero_init: bool. initialize with zero.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    """

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            return torch.nn.init.zeros_(x)

        d_in: int = x.size(1)

        x = 2 * torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) - 1
        if self.normalize:
            x.div_(d_in)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        d_in: int = grad.size(1)
        return torch.sign(grad).div_(d_in) if self.normalize else torch.sign(grad)


class Auto(Norm):
    r"""choose Norm type automatically."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ndim: int = x.ndim
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        ndim: int = grad.ndim
        if ndim in (0, 1):
            return BiasRMS().lmo(grad)
        if ndim == 2:
            return Spectral().lmo(grad)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad)
        raise NotImplementedError


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    r"""Build LMONorm by given norm_type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()

class SCION(BaseOptimizer):
    r"""Training Deep Learning Models with Norm-Constrained LMOs.

    Example:
        >>> radius = 50.0
        >>> parameter_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm_type': 'spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm_type': 'sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius * 60.0,
        ... }]
        >>> optimizer = SCION(parameter_groups)

        For more details, checkout here https://github.com/LIONS-EPFL/scion/tree/main?tab=readme-ov-file#examples

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor. 1.0 - usual momentum.
    :param constraint: bool. whether to use a constraint SCG or not.
    :param norm_type: int. supported LMO norm types. 0 stands for no normalization and 1 stands for AUTO. 0 to 7.
        please check LMONorm Enum class for the details.
    :param norm_kwargs: Optional[Dict]. arguments for the Norm.
    :param scale: float. based on the usage of the original intend, 50.0 is used for Transformer block, and 3000.0 is
        used for others (e.g. Embedding, LM head)
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        norm_type: int = LMONorm.AUTO,
        norm_kwargs: Optional[Dict] = None,
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        use_orthograd: bool = False,
        adaptive_clip: Optional[float] = None,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        use_stable_spam_clipping:bool = False,
        ssc_t_max: Optional[int] = None,
        use_focus: bool = False,
        focus_beta: float = 0.999,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')

        self.ssc_t_max = ssc_t_max
        self.warmup = SSCCosineDecay(1.0, ssc_t_max, eta_min=0.5) if ssc_t_max is not None else None

        if norm_kwargs is None:
            norm_kwargs = {}

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'norm_type': norm_type,
            'norm_kwargs': norm_kwargs,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'use_orthograd': use_orthograd,
            'adaptive_clip': adaptive_clip,
            'adaptive_clip_eps': adaptive_clip_eps,
            'adaptive_clip_type': adaptive_clip_type,
            'update_strategy': update_strategy,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'use_focus':use_focus,
            'focus_beta':focus_beta,
            'torch_compile': torch_compile,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCION'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])
            for p in group['params']:
                norm.init(p)
                p.mul_(group['scale'])
                state = self.state[p]
                state['d'] = torch.zeros_like(p)

                if group['use_focus']:
                    state['pbar'] = torch.zeros_like(p)

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

            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])

            use_orthograd = group['use_orthograd']
            adaptive_clip = group['adaptive_clip']
            adaptive_clip_eps = group['adaptive_clip_eps']
            adaptive_clip_type = group['adaptive_clip_type']
            update_strategy  = group['update_strategy']
            focus_beta = group['focus_beta']
            use_focus = group['use_focus']
            bias_correction2: float = self.debias(focus_beta, group['step'])

            use_stable_spam_clipping = group["use_stable_spam_clipping"]

            if use_stable_spam_clipping:
                scale: float = self.warmup.get_death_rate(group['step']) if self.warmup is not None else 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if 'd' not in state:
                    state['d'] = torch.zeros_like(p)
                    if use_focus:
                        state['pbar'] = torch.zeros_like(p)

                d = state['d']

                if p.dtype == torch.bfloat16:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    d = d.to(torch.float32)

                if use_orthograd:
                    _paper_orthograd(p_fp32, grad)

                if adaptive_clip is not None and adaptive_clip > 0:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                if use_stable_spam_clipping:
                    if group['torch_compile']:
                        grad = _get_compiled_stable_spam_clipping()(state,
                                            grad,
                                            step=group['step'],
                                            scale=scale)
                    else:
                        grad = _stable_spam_clipping_impl(state, 
                                            grad, 
                                            step=group['step'], 
                                            scale=scale)

                d.mul_(1.0 - group['momentum']).add_(grad, alpha=group['momentum'])

                if use_focus:
                    pbar = state['pbar']
                    if p.dtype == torch.bfloat16:
                        pbar = pbar.to(torch.float32)

                    pbar.mul_(focus_beta).add_(p_fp32, alpha=1.0 - focus_beta)

                update = norm.lmo(d).mul_(group['scale'])
                
                if not use_focus:
                    if group['constraint']:
                        p_fp32.mul_(1.0 - group['lr'])

                    if not group['constraint'] and group['weight_decay'] > 0.0:
                        apply_weight_decay(
                            p_fp32,
                            grad,
                            lr=group['lr'],
                            weight_decay=group['weight_decay'],
                            weight_decouple=group['weight_decouple'],
                            fixed_decay=False,
                            torch_compile=group.get('torch_compile', False),
                        )

                if update_strategy in {'cautious','grams','both'}:
                    if update_strategy in {'cautious','both'}:
                        mask = (update * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = update * mask
                    if update_strategy in {'grams','both'}:
                        update.copy_(torch.sign(grad) * update.abs())

                if use_focus:
                    pbar_hat = pbar / bias_correction2

                    if group['weight_decay'] > 0.0:
                        p_fp32.add_(pbar_hat, alpha=-group['lr'] * group['weight_decay'])

                update = (p_fp32 - pbar_hat).sign_().mul_(0.1).add_(torch.sign(d))

                p_fp32.add_(update, alpha=-group['lr'])

                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["d"], d)
                    if use_focus:
                        copy_stochastic_(state['pbar'], pbar)
                    copy_stochastic_(p, p_fp32)

        return loss
