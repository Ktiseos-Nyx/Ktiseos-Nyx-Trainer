import torch
from .utils import copy_stochastic_, agc, NORM_TYPE, create_factored_dims, get_denom, update_second_moment, adaptive_eps
import math
from typing import Optional, Literal

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup

CLIP_LOC = Literal['gradient', 'update', 'both']

class RMSProp(BaseOptimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.001)
        betas (float, optional):
            coefficient used for computing running averages of
            gradient's square (default: 0.95).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for adaptive eps, to prevent NaNs, set to >= 0 to turn on adaptive eps (default: None).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        rectify (bool)
            Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        n_sma_threshold: (int)
            Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        degenerated_to_sgd: (bool)
            degenerated to SGD. (Default: false)
        clip_loc: (string)
            Control where clipping is applied. Can be selectively applied: gradient, update, both (Default: gradient)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: float = 0.95, # normal default is 0.999, but was accidently 0.9 for awhile, so adjusting to 0.95 for now
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        centralization: float = 0.0,
        stable_decay: bool = False,
        clip: float = 0.0,
        clip_eps: float = 1e-8,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        clip_loc: CLIP_LOC = 'gradient',
        adaptive_clip_type: NORM_TYPE = 'layer',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(betas, 'betas', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(centralization, 'centralization')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(adaptive_clip_eps, 'adaptive_clip_eps')
        self.validate_non_negative(clip_eps, 'clip_eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr':lr,
            'betas':betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'fixed_decay' : fixed_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'centralization':centralization,
            'stable_decay':stable_decay,
            'clip':clip,
            'clip_eps':clip_eps,
            'adaptive_clipping':adaptive_clipping,
            'adaptive_clip_eps':adaptive_clip_eps,
            'rectify_variance':rectify_variance,
            'n_sma_threshold':n_sma_threshold,
            'degenerated_to_sgd':degenerated_to_sgd,
            'clip_loc':clip_loc,
            'adaptive_clip_type':adaptive_clip_type,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RMSProp'
    
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

                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        exp_avg_sq_sum: float = 0.0

        for group in self.param_groups:
            beta = group['betas']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                param_size += p.numel()

                state = self.state[p]

                p_fp32 = p

                if len(state) == 0:
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg_sq = state['exp_avg_sq']

                original_grad_dtype = grad.dtype

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                bias_correction_sq: float = self.debias(beta, group['step'])

                # center the gradient vector
                if group["centralization"] > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(group["centralization"])
                    )

                if group['clip'] > 0.0 and group['clip_loc'] in {'gradient','both'}:
                    if group['adaptive_clipping']:
                        # Apply Adaptive Gradient Clipping (AGC)
                        grad = agc(p=p_fp32, grad=grad, agc_clip_val=group['clip'], agc_eps=group['adaptive_clip_eps'], eps=group['clip_eps'], norm_type=group['adaptive_clip_type'])
                    else:
                        # Clip the gradient 
                        grad.div_((self.get_rms(grad).clamp_(group['clip_eps']) / group['clip']).clamp_(min=1.0))

                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    
                # Need to pack grad for next phase
                if original_grad_dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p.grad, grad)

                exp_avg_sq_sum += (exp_avg_sq / bias_correction_sq).sum()

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_mean: float = math.sqrt(exp_avg_sq_sum / param_size)

        for group in self.param_groups:
            beta = group["betas"]

            bias_correction_sqrt: float = math.sqrt(self.debias(beta, group['step']))

            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group["rectify_variance"],
                step=group['step'],
                lr=group['lr'],
                beta2=beta,
                n_sma_threshold=group["n_sma_threshold"],
                degenerated_to_sgd=group["degenerated_to_sgd"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p

                exp_avg_sq = state["exp_avg_sq"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                if not group["rectify_variance"] or step_size > 0 or n_sma >= group["n_sma_threshold"]:
                    if group["weight_decouple"]:
                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if group["fixed_decay"] else step_size) * group["weight_decay"] / (exp_avg_sq_mean if group["stable_decay"] else 1.0))
                    elif group["weight_decay"] > 0.0:
                        grad.add_(p_fp32, alpha=group["weight_decay"])

                curr_eps = adaptive_eps(grad, group)

                if not group["rectify_variance"] or n_sma >= group["n_sma_threshold"]:
                    # lr scaler + eps to prevent zero division
                    # de_nom = exp_avg_sq.sqrt() + group['eps']
                    if group["rectify_variance"]:
                        de_nom = exp_avg_sq.sqrt().add_(curr_eps)
                    else:
                        de_nom = (exp_avg_sq.sqrt() / bias_correction_sqrt).add_(curr_eps)

                    # p = p - lr * grad / denom
                    update = grad.div(de_nom)
                elif step_size > 0:
                    update = grad

                if group['clip'] > 0.0 and group['clip_loc'] in {'update','both'} and (step_size > 0 or n_sma >= group["n_sma_threshold"]):
                    if group['adaptive_clipping']:
                        # Apply Adaptive Gradient Clipping (AGC)
                        update = agc(p=p_fp32, grad=update, agc_clip_val=group['clip'], agc_eps=group['adaptive_clip_eps'], eps=group['clip_eps'], norm_type=group['adaptive_clip_type'])
                    else:
                        # Clip the gradient 
                        update.div_((self.get_rms(update).clamp_(group['clip_eps']) / group['clip']).clamp_(min=1.0))

                if step_size > 0 or n_sma >= group["n_sma_threshold"]:
                    p_fp32.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss

class RMSPropADOPT(BaseOptimizer):
    r"""ADOPT Style RMSProp.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for the exponential moving average squared (default: 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 1.0).
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
        factor_second_moment (bool):
            Stores the second moment, i.e. ema_sq / exponential moving average squared, at the row/column level 
            instead of per parameter saving vram at the cost of lower precision (Default: False)
        debias_beta (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True) 
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: float = 0.9999,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        factor_second_moment: bool = False,
        debias_beta: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(betas, 'betas')
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
            'factor_second_moment':factor_second_moment,
            'debias_beta':debias_beta,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RMSPropADOPT'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_sq_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]
                grad = p.grad

                factored_dims = create_factored_dims(
                    grad.shape,
                    factored=group['factor_second_moment'],
                    min_dim_size_to_factor=32
                )

                if factored_dims is not None:
                    dc, dr = factored_dims
                    row_shape = list(p.grad.shape)
                    row_shape[dr] = 1
                    col_shape = list(p.grad.shape)
                    col_shape[dc] = 1
                    reduce_dc = dc - 1 if dc > dr else dc
                    # Store reduction variables so we don't have to recalculate each step.
                    # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
                    # between bf16/fp16 and fp32 is negligible here.
                    state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                            torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                            dr, dc, reduce_dc]
                else:
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
                group['exp_avg_sq_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta = group['betas']

            lr: float = group['lr']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            if group["debias_beta"]:
                bias_correction_sqrt: float = math.sqrt(self.debias(beta, group['step']))
            else:
                bias_correction_sqrt = 1.0

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
                    factored_dims = create_factored_dims(
                        grad.shape,
                        factored=group['factor_second_moment'],
                        min_dim_size_to_factor=32
                    )

                    if factored_dims is not None:
                        dc, dr = factored_dims
                        row_shape = list(p.grad.shape)
                        row_shape[dr] = 1
                        col_shape = list(p.grad.shape)
                        col_shape[dc] = 1
                        reduce_dc = dc - 1 if dc > dr else dc
                        # Store reduction variables so we don't have to recalculate each step.
                        # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
                        # between bf16/fp16 and fp32 is negligible here.
                        state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                                torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                                dr, dc, reduce_dc]
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg_sq = state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    if not group['factor_second_moment']:
                        exp_avg_sq = exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq = update_second_moment(exp_avg_sq, grad, beta, True)
                else:
                    de_nom = get_denom(exp_avg_sq).div_(bias_correction_sqrt).add_(curr_eps)
                    exp_avg_sq = update_second_moment(exp_avg_sq, grad, beta)

                    normed_grad = grad.div(de_nom)
                    normed_grad.clamp_(-adopt_clip, adopt_clip)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        normed_grad.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.add_(normed_grad, alpha=-lr)

                    if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                        exp_avg_sq_sum += exp_avg_sq.sum()

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    group['exp_avg_sq_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if not group['factor_second_moment']:
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss

class RMSPropADOPTMARS(BaseOptimizer):
    r"""ADOPT Style RMSProp.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for the exponential moving average squared (default: 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 1.0).
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
        factor_second_moment (bool):
            Stores the second moment, i.e. ema_sq / exponential moving average squared, at the row/column level 
            instead of per parameter saving vram at the cost of lower precision (Default: False)
        debias_beta (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True) 
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: float = 0.9999,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        factor_second_moment: bool = False,
        debias_beta: bool = True,
        gamma: float = 0.025,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(betas, 'betas')
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
            'factor_second_moment':factor_second_moment,
            'debias_beta':debias_beta,
            'gamma':gamma,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RMSPropADOPTMARS'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_sq_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]
                grad = p.grad
                state['previous_grad'] = torch.zeros_like(p)

                factored_dims = create_factored_dims(
                    grad.shape,
                    factored=group['factor_second_moment'],
                    min_dim_size_to_factor=32
                )

                if factored_dims is not None:
                    dc, dr = factored_dims
                    row_shape = list(p.grad.shape)
                    row_shape[dr] = 1
                    col_shape = list(p.grad.shape)
                    col_shape[dc] = 1
                    reduce_dc = dc - 1 if dc > dr else dc
                    # Store reduction variables so we don't have to recalculate each step.
                    # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
                    # between bf16/fp16 and fp32 is negligible here.
                    state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                            torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                            dr, dc, reduce_dc]
                else:
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
                group['exp_avg_sq_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta = group['betas']

            lr: float = group['lr']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]

            if group["debias_beta"]:
                bias_correction_sqrt: float = math.sqrt(self.debias(beta, group['step']))
            else:
                bias_correction_sqrt = 1.0

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
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                    factored_dims = create_factored_dims(
                        grad.shape,
                        factored=group['factor_second_moment'],
                        min_dim_size_to_factor=32
                    )

                    if factored_dims is not None:
                        dc, dr = factored_dims
                        row_shape = list(p.grad.shape)
                        row_shape[dr] = 1
                        col_shape = list(p.grad.shape)
                        col_shape[dc] = 1
                        reduce_dc = dc - 1 if dc > dr else dc
                        # Store reduction variables so we don't have to recalculate each step.
                        # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
                        # between bf16/fp16 and fp32 is negligible here.
                        state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                                torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                                dr, dc, reduce_dc]
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg_sq, grad_diff = state['exp_avg_sq'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    grad_diff = grad_diff.to(torch.float32)
                    if not group['factor_second_moment']:
                        exp_avg_sq = exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                grad_diff.add_(grad)
                
                # MARS Calculate cₜ (gradient with correction term)
                correction = gamma * grad_diff
                c_t = grad + correction

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq = update_second_moment(exp_avg_sq, c_t, beta, True)
                else:
                    de_nom = get_denom(exp_avg_sq).div_(bias_correction_sqrt).add_(curr_eps)
                    exp_avg_sq = update_second_moment(exp_avg_sq, c_t, beta)

                    normed_grad = c_t.div(de_nom)
                    normed_grad.clamp_(-adopt_clip, adopt_clip)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        normed_grad.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.add_(normed_grad, alpha=-lr)

                    if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                        exp_avg_sq_sum += exp_avg_sq.sum()

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    group['exp_avg_sq_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['previous_grad'], -grad)
                    if not group['factor_second_moment']:
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

        return loss