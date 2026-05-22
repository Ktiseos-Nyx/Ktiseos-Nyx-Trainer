# Copied from Lodestone and slightly modified, still should function the same, just added an extra check
# to turn off the stochastic rounding
# repo: https://github.com/lodestone-rock/compass_optimizer/blob/main/experimental/compass_experimental_sr_bf16.py
# Defaults tuned for lora training based on testing

import torch
from torch.optim import Optimizer
from .utils import (_paper_orthograd, CosineDecay, CLIP_TYPE, copy_stochastic_, agc, 
                    NORM_TYPE, create_factored_dims, get_denom, update_second_moment, STATE_PRECISION, 
                    UPDATE_STRATEGY, spam_grad_clipping_logging, spam_grad_clipping, _get_compiled_stable_spam_clipping, _stable_spam_clipping_impl, SSCCosineDecay, adaptive_eps)
import math
from torch.nn.functional import softplus
from typing import Optional
import logging

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm
from .low_bit_optim.quant_utils import _fp32_to_bf16_sr
from .low_bit_optim.subclass_8bit import OptimState8bit
from .low_bit_optim.subclass_4bit import OptimState4bit
from .low_bit_optim.subclass_fp8 import OptimStateFp8
from torch.distributed._tensor import DTensor



class Compass(BaseOptimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        stable_weight_decay (bool): 
            Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        lr_decouple (bool): 
            Apply fully decoupled weight decay. (default: false)
        max_lr (float): 
            Max LR used for lr_decouple (default: 0.0)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            Gradient centralization  - https://arxiv.org/abs/2004.01461v2 (default: 0.0).
        adaptive_clipping (bool):
            enable adaptive clipping - https://arxiv.org/abs/2102.06171 (default: false).
        adaptive_clipping_eps (float):
            eps for adaptive gradient clipping (default: 1e-3).
        adam_debias: (bool)
            Only correct the denominator to avoid inflating step sizes early in training. (Default: false)
        rectify_variance: (bool)
            Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        n_sma_threshold: (int)
            Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        degenerated_to_sgd: (bool)
            degenerated to SGD. (Default: false)
        cautious (bool) (deprecated, use update strategy)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
            Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
            and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
        use_orthograd (boolean):
            Experimental. Updates weights using the component of the gradient that is orthogonal to the current 
            weight direction, as described in "Grokking at the Edge of Numerical Stability" (https://arxiv.org/pdf/2501.04697).
            Can help prevent overfitting and improve generalisation.
            (default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1.4e-4, #Original default 1e-3
        betas: Betas = (0.975, 0.999), #Original default 0.99, 0.999
        weight_decay: float = 0.001, #Original default 0
        weight_decouple: bool = True,
        stable_weight_decay: bool = False,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        fixed_decay: bool = False,
        clip: float = 0.01,
        amp_fac: float = 2.0,
        eps: float = 1e-8,
        centralization: float = 0.0,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
        adam_debias: bool = False,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        use_orthograd: bool = False,
        **kwargs,
    ):
        
        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'
        
        defaults: Defaults = {
            'lr':lr,
            'betas':betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'lr_decouple':lr_decouple,
            'max_lr':max_lr,
            'fixed_decay' : fixed_decay,
            'clip':clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'amp_fac':amp_fac,
            'eps':eps,
            'centralization':centralization,
            'adaptive_clipping':adaptive_clipping,
            'adam_debias': adam_debias,
            'rectify_variance': rectify_variance,
            'n_sma_threshold': n_sma_threshold,
            'degenerated_to_sgd': degenerated_to_sgd,
            'cautious':cautious,
            'update_strategy': update_strategy,
            'stable_weight_decay': stable_weight_decay,
            'use_orthograd': use_orthograd,
        }

        self.clip = clip
        self.adaptive_clip_eps = adaptive_clip_eps
        self.adaptive_clipping = adaptive_clipping
        self.adam_debias = adam_debias
        self.rectify_variance = rectify_variance
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['exp_avg_sq_mean_sqrt'] = 0.0
            group['step'] = 0

            for p in group['params']:
                state = self.state[p]

                # Exponential moving average of gradient values
                state['ema'] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

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

            beta1, beta2 = group["betas"]
            amp_fac = group["amp_fac"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            lr_decouple = group["lr_decouple"]
            max_lr = group["max_lr"]
            update_strategy = group["update_strategy"]

            # bias correction step size
            # soft warmup
            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sqrt: float = math.sqrt(self.debias(beta2, group['step']))

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=self.rectify_variance,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=self.adam_debias,
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()   

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

                p_fp32 = p

                ema, ema_squared = state["ema"], state["ema_squared"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                if group["use_orthograd"]:
                    _paper_orthograd(p_fp32, grad)

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                if self.clip > 0.0:
                    if self.adaptive_clipping:
                        # Apply Adaptive Gradient Clipping (AGC)
                        grad = agc(p=p_fp32, grad=grad, agc_clip_val=self.clip, agc_eps=self.adaptive_clip_eps, norm_type='layer')
                    else:
                        # Clip the gradient 
                        grad.div_((self.get_rms(grad).add_(eps) / self.clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amp_fac
                update = grad.add(ema, alpha=amp_fac)
                # ema_squared = ema + (1 - beta2) * update ** 2
                ema_squared.mul_(beta2).addcmul_(update, update, value=1 - beta2)

                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if weight_decouple:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if fixed_decay else step_size if not lr_decouple else step_size / max_lr) * weight_decay * swd_scaling)
                    elif weight_decay > 0.0 and update is not None:
                        update.add_(p_fp32, alpha=weight_decay)


                if update_strategy in {'cautious','grams'}:
                    if update_strategy == 'cautious':
                        mask = (update * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    elif update_strategy == 'grams':
                        update.copy_(torch.sign(grad) * update.abs())
                        mask = 1.0
                else:
                    mask = 1.0

                if not self.rectify_variance or n_sma >= self.n_sma_threshold:
                    # lr scaler + eps to prevent zero division
                    # de_nom = exp_avg_sq.sqrt() + group['eps']
                    if self.rectify_variance:
                        de_nom = ema_squared.sqrt().add_(eps)
                    else:
                        de_nom = (ema_squared.sqrt() / bias_correction2_sqrt).add_(eps)

                    # p = p - lr * grad / de_nom
                    p_fp32.addcdiv_(update * mask, de_nom, value=-step_size)
                elif step_size > 0:
                    p_fp32.add_(update * mask, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += ema_squared.sum()

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_sq_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class CompassPlus(BaseOptimizer):
    r"""
    CompassPlus
        Components
            * Adaptive gradient clipping - https://arxiv.org/abs/2102.06171
            * Gradient centralization - https://arxiv.org/abs/2004.01461v2
            * Positive-Negative momentum - https://arxiv.org/abs/2103.17182
            * Norm loss - https://arxiv.org/abs/2103.06583v1
            * Fully decoupled weight decay - https://optimi.benjaminwarner.dev/fully_decoupled_weight_decay/ / https://arxiv.org/abs/1711.05101
            * Stable weight decay - https://arxiv.org/abs/2011.11152v3
            * Lookahead - https://arxiv.org/abs/1907.08610
            * Softplus transformation - https://arxiv.org/abs/1908.00700
            * Gradient Normalization - https://arxiv.org/pdf/1711.02257 (?)
            * Adaptive eps - https://arxiv.org/abs/2405.12807
            * Diff amp - https://github.com/Clybius/Personalized-Optimizers/blob/main/FishMonger.py
            * Slow EMA - https://arxiv.org/abs/2409.03137
            * Amsgrad - https://arxiv.org/pdf/1904.09237
            * Update Clipping - https://arxiv.org/pdf/2304.13013 (AdamWStable) / https://arxiv.org/pdf/1804.04235 (Adafactor)
            * Variance Rectification - https://arxiv.org/abs/1908.03265 (RAdam)

    Arguments:
        :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: float. learning rate.
        :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
        :param use_softplus: bool. use softplus to smooth the updaate denominator.
        :param beta_softplus: float. beta for softplus.
        :param threshold_softplus: float. threshold after which scaling returns to linear. Originally set to 20 by default, instead follows adaptive eps when set to 0.
        :param agc_clipping_value: float. Clipping threshold for adaptive gradient clipping.
        :param agc_eps: float. eps for adaptive gradient clipping.
        :param amp_fac: float. amplification factor for the first moment filter.
        :param centralize_gradients: bool. use GC both convolution & fc layers. Can be selectively applied an int: disabled(0), gradient(1), update(2), both(3)
        :param normalize_gradients: bool. use gradient normalization.  Can be selectively applied using an int: disabled(0), gradient(1), update(2), both(3)
        :param use_lookahead: bool. use lookahead. ADDS 1 State
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param lr_decouple: bool. fully decouple weight decay from learning rate. This makes weight decay much stronger given the same values.
        :param max_lr: float. Max LR used for lr_decouple, should match your defined max LR for training.
        :param fixed_decay: bool. fix weight decay.
        :param norm_loss_factor: float. norm loss factor.
        :param norm_loss_eps: float. Eps is the term added to the denominator to improve numerical stability.
        :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
        :param amsgrad: bool. If true, maintains and uses the max ema squared. ADDS 1 State
        :param use_pnm: bool. use positive negative momentum. ADDS 1 State
        :param pnm_beta: float. Manages the amplitude of the noise introduced by positive negative momentum. Negative values are valid.
        :param use_slow_ema: bool. use slow ema like that from AdEMAMix. ADDS 1 State
        :param slow_ema_alpha: float. usually between 4 and 10 would work well. The multipler for application of the slow ema to the update.
        :param slow_ema_beta: float. coefficient used for computing running slow average of gradient.
        :param slow_ema_t_alpha_beta: Optional[float]. total number of iterations is preferred when needed. The warmup of slow_ema_alpha and slow_ema_beta over iterations. Results in more stablity.
        :param diff_amp: float. Accelerate the difference between the current and past gradient by this multiplicative value. 0 is off. ADDS 2 STATES
        :param diff_amp_beta: float. Coefficient used for computing running average of the current and past gradients
        :param eps: float. the maximum eps value for adaptive eps. Eps is the term added to the denominator outside of the root operation to improve numerical stability.
        :param eps2: float. used to multiple the grad rms for determining adaptive eps.
        :param eps_floor: float. term used to determine the floor for adaptive eps.
        :param update_clipping: bool. Apply update clipping using root mean square of the gradient, similar to Adafactor. Advise disabling gradient clipping (clip=0.0).
        :param rectify_variance: bool. Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        :param n_sma_threshold: int. Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        :param degenerated_to_sgd: bool. degenerated to SGD. (Default: false)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1.4e-4,
        betas: Betas = (0.975, 0.999),
        weight_decay: float = 0.0005,
        weight_decouple: bool = True,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        stable_decay: bool = False,
        fixed_decay: bool = False,
        clip: float = 0.01,
        clip_eps: float = 1e-3,
        amp_fac: float = 2.0,
        centralize_gradients: int = 0,
        normalize_gradients: int = 0,
        norm_loss_factor: float = 0,
        norm_loss_eps: float = 1e-8,
        use_softplus: bool = False,
        beta_softplus: float = 50.0,
        threshold_softplus: float = 0.0,
        use_lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        adam_debias: bool = False,
        use_pnm: bool = False,
        pnm_beta: float = 0.1,
        amsgrad: bool = False,
        use_slow_ema: bool = False,
        slow_ema_beta: float = 0.9998,
        slow_ema_alpha: float = 3.0,
        slow_ema_t_alpha_beta: Optional[float] = None,
        diff_amp: float = 0.0,
        diff_amp_beta: float = 0.999,
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float = 1e-16,
        update_clipping: bool = False,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(pnm_beta, 'pnm_beta', -1.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_lr, 'max_lr')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(clip, 'clip_eps')
        self.validate_non_negative(amp_fac, 'amp_fac')
        self.validate_non_negative(lookahead_blending_alpha, 'lookahead_blending_alpha')
        self.validate_non_negative(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_non_negative(beta_softplus, 'beta_softplus')
        self.validate_non_negative(threshold_softplus, 'threshold_softplus')
        self.validate_non_negative(norm_loss_factor, 'norm_loss_factor')
        self.validate_non_negative(slow_ema_alpha, 'slow_ema_alpha')
        self.validate_non_negative(diff_amp, 'diff_amp')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(eps2, 'eps2')
        self.validate_non_negative(eps_floor, 'eps_floor')
        self.validate_range(diff_amp_beta, 'diff_amp_beta', 0.0, 1.0, range_type='[]')
        self.validate_range(slow_ema_beta, 'slow_ema_beta', 0.0, 1.0, range_type='[]')

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'lr_decouple': lr_decouple,
            'max_lr': max_lr,
            'stable_decay': stable_decay,
            'fixed_decay': fixed_decay,
            'clip': clip,
            'clip_eps': clip_eps,
            'amp_fac': amp_fac,
            'centralize_gradients': centralize_gradients,
            'normalize_gradients': normalize_gradients,
            'norm_loss_factor': norm_loss_factor,
            'norm_loss_eps': norm_loss_eps,
            'use_softplus': use_softplus,
            'beta_softplus': beta_softplus,
            'threshold_softplus': threshold_softplus,
            'use_lookahead': use_lookahead,
            'lookahead_merge_time': lookahead_merge_time,
            'lookahead_blending_alpha': lookahead_blending_alpha,
            'adam_debias': adam_debias,
            'use_pnm': use_pnm,
            'pnm_beta': pnm_beta,
            'amsgrad': amsgrad,
            'use_slow_ema': use_slow_ema,
            'slow_ema_beta': slow_ema_beta,
            'slow_ema_alpha': slow_ema_alpha,
            'slow_ema_t_alpha_beta': slow_ema_t_alpha_beta,
            'diff_amp': diff_amp,
            'diff_amp_beta': diff_amp_beta,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'update_clipping': update_clipping,
            'rectify_variance': rectify_variance,
            'n_sma_threshold': n_sma_threshold,
            'degenerated_to_sgd': degenerated_to_sgd,
        }

        self.use_lookahead = use_lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead_step: int = 0
        self.use_pnm = use_pnm
        self.adam_debias = adam_debias
        self.pnm_beta = pnm_beta
        self.amsgrad = amsgrad
        self.stable_decay = stable_decay
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.threshold_softplus = threshold_softplus
        self.norm_loss_factor = norm_loss_factor
        self.norm_loss_eps = norm_loss_eps
        self.lr_decouple = lr_decouple
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.max_lr = max_lr
        self.fixed_decay = fixed_decay
        self.clip = clip
        self.clip_eps = clip_eps
        self.amp_fac = amp_fac
        self.use_slow_ema = use_slow_ema
        self.slow_ema_beta = slow_ema_beta
        self.slow_ema_alpha = slow_ema_alpha
        self.slow_ema_t_alpha_beta = slow_ema_t_alpha_beta
        self.diff_amp = diff_amp
        self.diff_amp_beta = diff_amp_beta
        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        self.update_clipping = update_clipping
        self.rectify_variance = rectify_variance
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassPlus'
    
    def init_group(self, group, **kwargs) -> None:
        pass
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0

            beta1, beta2 = group["betas"]

            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                # Exponential moving average of gradient values
                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    state['ema'] = torch.zeros_like(p)
                else: 
                    state['ema'] = None

                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

                if self.use_pnm:
                    state['neg_ema'] = torch.zeros_like(p)

                if self.use_lookahead:
                    state['lookahead_params'] = p.clone()

                if self.amsgrad:
                    state["max_ema_squared"] = torch.zeros_like(p)

                if self.use_slow_ema:
                    state['ema_slow'] = torch.zeros_like(p)

                # Previous grad
                if self.diff_amp:
                    state["ema_diff"] = torch.zeros_like(p)
                    state["previous_grad"] = grad.clone().mul_(-1.0)
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())
    
    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + 1e-8), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        ema_squared_sum: float = 1.0

        # Phase 1 - Condition the grads and gather aggregates 
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                state = self.state[p]
                p_fp32 = p

                param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        state['ema'] = torch.zeros_like(p)
                    else: 
                        state['ema'] = None

                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

                    if self.use_pnm:
                        state['neg_ema'] = torch.zeros_like(p)

                    if self.use_lookahead:
                        state['lookahead_params'] = p.clone()

                    if self.amsgrad:
                        state["max_ema_squared"] = torch.zeros_like(p)

                    if self.use_slow_ema:
                        state['ema_slow'] = torch.zeros_like(p)

                    # Previous grad
                    if self.diff_amp:
                        state["ema_diff"] = torch.zeros_like(p)
                        state["previous_grad"] = grad.clone().mul_(-1.0)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)

                # Apply Adaptive Gradient Clipping (AGC)
                if self.clip > 0.0:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=self.clip, agc_eps=self.clip_eps, norm_type='layer')

                # Apply gradient centralization & normalization
                if self.centralize_gradients in {1,3}:
                    centralize_gradient(grad, gc_conv_only=False)

                if self.normalize_gradients in {1,3}:
                    normalize_gradient(grad)

        if param_size == 0:
            raise ZeroParameterSizeError()

        # Phase 2
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            # bias correction step size
            # soft warmup
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            if self.use_slow_ema:
                # Scale with amp fac for consistency
                slow_ema_alpha_t: float = self.schedule_alpha(self.slow_ema_t_alpha_beta, group['step'], self.slow_ema_alpha * self.amp_fac)
                slow_ema_beta3_t: float = self.schedule_beta3(self.slow_ema_t_alpha_beta, group['step'], beta1, self.slow_ema_beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                ema_squared = state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema, neg_ema = state['ema'], state['neg_ema']
                    else:
                        ema, neg_ema = state['neg_ema'], state['ema']
                else:
                    ema = state["ema"]

                if self.use_slow_ema:
                    ema_slow = state['ema_slow']



                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        ema = ema.to(torch.float32)

                    if self.use_pnm:
                        neg_ema = neg_ema.to(torch.float32)

                    if self.use_slow_ema:
                        ema_slow = ema_slow.to(torch.float32)

                # Decay the first and second moment running average coefficient
                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    # ema = ema + (1 - beta1) * grad
                    ema.mul_(beta1).add_(grad, alpha=1.0 - beta1)  # fmt: skip
                else:
                    ema = grad

                # Natural grad
                if self.diff_amp > 0.0 or self.use_pnm or self.use_slow_ema:
                    nat_grad = grad.clone()
                    nat_grad_amp = nat_grad.add(ema, alpha=self.amp_fac)
                else:
                    nat_grad_amp = grad
                    nat_grad = grad

                if self.use_pnm:
                    noise_norm: float = math.sqrt((1.0 + self.pnm_beta) ** 2 + self.pnm_beta ** 2)
                    adjusted_ema = ema.mul(1.0 + self.pnm_beta).add_(neg_ema, alpha=-self.pnm_beta).mul_(1.0 / noise_norm)
                else:
                    adjusted_ema = ema

                # grad = grad + ema * amplification_factor
                grad.add_(adjusted_ema, alpha=self.amp_fac)

                if self.use_slow_ema:
                    ema_slow.mul_(slow_ema_beta3_t).add_(nat_grad, alpha=1.0 - slow_ema_beta3_t)
                    grad.add_(ema_slow, alpha=slow_ema_alpha_t)

                if self.diff_amp > 0.0:
                    grad_diff = state["previous_grad"]
                    ema_diff = state['ema_diff']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff = grad_diff.to(torch.float32)
                        ema_diff = ema_diff.to(torch.float32)

                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff.add_(nat_grad)

                    # Smooth the difference between previous grad and current grad
                    ema_diff.mul_(self.diff_amp_beta).add_(grad_diff, alpha=1 - self.diff_amp_beta)

                    # Scale with amp fac for consistency
                    grad.add_(ema_diff, alpha=self.diff_amp * self.amp_fac)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["previous_grad"], -nat_grad)
                        copy_stochastic_(state["ema_diff"], ema_diff)
                    else:
                        state["previous_grad"].copy_(-nat_grad)

                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(nat_grad_amp, nat_grad_amp, value=1.0 - beta2)
                if self.stable_decay:
                    ema_squared_sum += (ema_squared / bias_correction2_sq).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema_squared"], ema_squared)

                    if self.use_pnm:
                        if group['step'] % 2 == 1:
                            if beta1 > 0.0:
                                copy_stochastic_(state["ema"], ema)
                        else:
                            # neg_ema is previous grad if beta1 is 0.0
                            copy_stochastic_(state["neg_ema"], ema)

                    else:
                        if beta1 > 0.0:
                            copy_stochastic_(state["ema"], ema)

                    if self.use_slow_ema:
                        copy_stochastic_(state["ema_slow"], ema_slow)

        if self.stable_decay:
            ema_squared_normalized = math.sqrt(ema_squared_sum / param_size)
        else:
            ema_squared_normalized = ema_squared_sum

        # Phase 3 - Weight decay and parameter update
        for group in self.param_groups:
            # bias correction step size
            # soft warmup
            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            eps_p2: float = math.pow(group['eps'], 2)

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=self.rectify_variance,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=self.adam_debias,
                step_size=step_size,
                bias_correction1=bias_correction1,
            )
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p
                ema_squared = state["ema_squared"]

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                # Basically should allow smaller eps whenever grad is small, so eps doesn't have outsized influence
                rms_grad = grad.pow(2).mean().sqrt_()
                current_eps = max(min(rms_grad.item() * self.eps2, self.eps), self.eps_floor) # Set a floor for eps to avoid NaN
 
                # lr scaler + eps to prevent zero division
                # de_nom = exp_avg_sq.sqrt() + group['eps']
                if self.amsgrad:
                    max_ema_squared = state['max_ema_squared']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        max_ema_squared = max_ema_squared.to(torch.float32)
                        
                    torch.max(max_ema_squared, ema_squared, out=max_ema_squared)
                    if self.rectify_variance:
                        de_nom = max_ema_squared.sqrt().add_(current_eps)
                    else:
                        de_nom = (max_ema_squared.sqrt() / bias_correction2_sq).add_(current_eps)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state['max_ema_squared'], max_ema_squared)
                else:
                    if self.rectify_variance:
                        de_nom = ema_squared.sqrt().add_(current_eps)
                    else:
                        de_nom = (ema_squared.sqrt() / bias_correction2_sq).add_(current_eps)

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus, threshold=self.threshold_softplus if self.threshold_softplus != 0 else current_eps)

                if self.update_clipping:
                    rms = grad.pow(2).div_(ema_squared.maximum(eps_p2)).mean().sqrt_()
                    step_size = step_size / max(1, rms.item())

                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if self.weight_decouple:
                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if self.fixed_decay else step_size if not self.lr_decouple else step_size / self.max_lr) * self.weight_decay * (1.0 / ema_squared_normalized if self.stable_decay else 1.0))
                    elif self.weight_decay > 0.0 and not self.use_slow_ema:
                        grad.add_(p_fp32, alpha=self.weight_decay)

                    if self.norm_loss_factor > 0.0:
                        # norm loss
                        correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p_fp32).add_(self.norm_loss_eps))
                        p_fp32.mul_(1.0 - step_size * correction)

                if not self.rectify_variance or n_sma >= self.n_sma_threshold:
                    update = grad.div(de_nom)
                else:
                    update = grad

                # Apply weight decay like AdEMAMix
                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if self.weight_decay > 0.0 and self.use_slow_ema and not self.weight_decouple:
                        update.add_(p_fp32, alpha=self.weight_decay)

                if self.centralize_gradients in {2,3}:
                    centralize_gradient(update, gc_conv_only=False)

                if self.normalize_gradients in {2,3}:
                    normalize_gradient(update) 

                # p = p - lr * grad / de_nom
                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    p_fp32.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        if self.use_lookahead:
            self.lookahead_process_step()

        return loss
    
    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p_fp32 = p

                    lookahead_params = state['lookahead_params']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32 = p.clone().to(torch.float32)
                        lookahead_params = lookahead_params.to(torch.float32)

                    p_fp32.mul_(self.lookahead_blending_alpha).add_(
                        lookahead_params,
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(p, p_fp32)

                    state['lookahead_params'].copy_(p)

class Compass8BitBNB(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0.0).
        quantization_group_size (int):
            number of quant group (default: 64).
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        weight_decay=0.001, #Original default 0
        weight_decouple=True,
        fixed_decay=False,
        clip=0.0,
        amp_fac=2,
        eps=1e-8,
        centralization=0.0,
        quantization_group_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay = weight_decay,
            weight_decouple = weight_decouple,
            fixed_decay = fixed_decay,
            clip=clip,
            centralization=centralization,
            group_size=quantization_group_size,
        )
        super(Compass8BitBNB, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass8BitBNB'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            clip = group["clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass8BitBNB does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                weight_decouple = group["weight_decouple"],
                fixed_decay = group["fixed_decay"]
                centralization = group["centralization"]
                eps = group["eps"]
                clip = group["clip"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** group['step']
                bias_correction_sqrt = (1 - beta2 ** group['step']) ** (1 / 2)
                debiased_lr = lr / bias_correction

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                ema = dequantize_blockwise(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = (
                    dequantize_blockwise(*state["ema_squared"]) + (1 - beta2) * grad**2
                )
                state["ema"] = quantize_blockwise(
                    ema,
                    blocksize=group["group_size"],
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize_blockwise(
                    ema_squared,
                    blocksize=group["group_size"],
                )

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else debiased_lr) * weight_decay)
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-debiased_lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss
    
class CompassADOPT(BaseOptimizer):
    r"""ADOPT Style Compass.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.95, 0.9999).
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
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: True)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
        compass_second_moment_smoothing (bool):
            Updates the second moment (i.e. ema / fim) with the Compass smoothed gradient. (Default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.95, 0.9999),
        amp_fac: float = 2.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = False,
        factor_second_moment: bool = False,
        debias_beta1: bool = True,
        debias_beta2: bool = True,
        compass_second_moment_smoothing: bool = True,
        use_orthograd: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        use_stable_spam_clipping: bool = False,
        ssc_t_max: Optional[int] = None,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.ssc_t_max = ssc_t_max
        self.warmup = SSCCosineDecay(1.0, ssc_t_max, eta_min=0.5) if ssc_t_max is not None else None

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams','both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'amp_fac': amp_fac,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'cautious': cautious,
            'factor_second_moment':factor_second_moment,
            'debias_beta1': debias_beta1,
            'debias_beta2': debias_beta2,
            'compass_second_moment_smoothing': compass_second_moment_smoothing,
            'use_orthograd': use_orthograd,
            'update_strategy': update_strategy,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'torch_compile': torch_compile,
            **kwargs
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassADOPT'
    
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

                state['exp_avg'] = torch.zeros_like(p)

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

            use_stable_spam_clipping = group["use_stable_spam_clipping"]

            if use_stable_spam_clipping:
                scale: float = self.warmup.get_death_rate(group['step']) if self.warmup is not None else 1.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            amp_fac = group["amp_fac"]
            compass_second_moment_smoothing = group["compass_second_moment_smoothing"]
            use_orthograd = group["use_orthograd"]
            update_strategy = group["update_strategy"]

            lr: float = group['lr']

            bias_correction1: float = self.debias(beta1, group['step'])
            if group["debias_beta2"]:
                bias_correction2_sqrt: float = math.sqrt(self.debias(beta2, group['step']))
            else:
                bias_correction2_sqrt = 1.0

            step_size = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

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
                    state['exp_avg'] = torch.zeros_like(p)
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

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    exp_avg = exp_avg.to(torch.float32)
                    if not group['factor_second_moment']:
                        exp_avg_sq = exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if use_orthograd:
                    _paper_orthograd(p_fp32, grad)

                if adaptive_clip is not None and adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
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

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq = update_second_moment(exp_avg_sq, grad, beta2, True)
                else:
                    de_nom = get_denom(exp_avg_sq).div_(bias_correction2_sqrt).add_(curr_eps)

                    if compass_second_moment_smoothing:
                        scaled_adopt_clip = adopt_clip * de_nom
                        normed_grad = grad.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                        unnormed_exp_avg = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
                        exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)

                        unnormed_update = grad.add(unnormed_exp_avg, alpha=amp_fac)
                        update = normed_grad.add(exp_avg, alpha=amp_fac)

                        exp_avg_sq = update_second_moment(exp_avg_sq, unnormed_update, beta2)

                        update_grad = grad
                    else:
                        normed_grad = grad.div(de_nom).clamp(-adopt_clip, adopt_clip)
                        exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)
                        update = normed_grad.add(exp_avg, alpha=amp_fac)
                        exp_avg_sq = update_second_moment(exp_avg_sq, grad, beta2)
                        update_grad = normed_grad

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    if update_strategy in {'cautious','grams','both'}:
                        if update_strategy in {'cautious','both'}:
                            mask = (update * update_grad > 0).to(update_grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        if update_strategy in {'grams','both'}:
                            update.copy_(torch.sign(update_grad) * update.abs())

                    if compass_second_moment_smoothing:
                        p_fp32.addcdiv_(update, de_nom, value=-step_size)
                    else:
                        p_fp32.add_(update, alpha=-step_size)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    if not group['factor_second_moment']:
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_sq_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss

class CompassADOPTMARS(BaseOptimizer):
    r"""ADOPT Style Compass + MARS correction.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.95, 0.9999).
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
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: True)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
        compass_second_moment_smoothing (bool):
            Updates the second moment (i.e. ema / fim) with the Compass smoothed gradient. (Default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.95, 0.9999),
        amp_fac: float = 2.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        stable_weight_decay: bool = False,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = True,
        factor_second_moment: bool = False,
        gamma: float = 0.025,
        debias_beta1: bool = True,
        debias_beta2: bool = True,
        compass_second_moment_smoothing: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'amp_fac': amp_fac,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'cautious': cautious,
            'factor_second_moment':factor_second_moment,
            'gamma':gamma,
            'debias_beta1': debias_beta1,
            'debias_beta2': debias_beta2,
            'compass_second_moment_smoothing': compass_second_moment_smoothing,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassADOPTMARS'
    
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

                state['exp_avg'] = torch.zeros_like(p)
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

            beta1, beta2 = group['betas']

            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            amp_fac = group["amp_fac"]
            gamma = group["gamma"]
            compass_second_moment_smoothing = group["compass_second_moment_smoothing"]

            lr: float = group['lr']

            bias_correction1: float = self.debias(beta1, group['step'])
            if group["debias_beta2"]:
                bias_correction2_sqrt: float = math.sqrt(self.debias(beta2, group['step']))
            else:
                bias_correction2_sqrt = 1.0

            step_size = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()
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

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    exp_avg = exp_avg.to(torch.float32)
                    previous_grad = previous_grad.to(torch.float32)
                    if not group['factor_second_moment']:
                        exp_avg_sq = exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)
                
                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq = update_second_moment(exp_avg_sq, c_t, beta2, True)
                else:
                    de_nom = get_denom(exp_avg_sq).div_(bias_correction2_sqrt).add_(curr_eps)

                    if compass_second_moment_smoothing:
                        scaled_adopt_clip = adopt_clip * de_nom
                        normed_grad = c_t.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                        unnormed_exp_avg = exp_avg.mul(beta1).add_(c_t, alpha=1.0 - beta1)
                        exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)

                        unnormed_update = c_t.add(unnormed_exp_avg, alpha=amp_fac)
                        update = normed_grad.add(exp_avg, alpha=amp_fac)

                        exp_avg_sq = update_second_moment(exp_avg_sq, unnormed_update, beta2)

                        cautious_grad = c_t
                    else:
                        normed_grad = grad.div(de_nom).clamp(-adopt_clip, adopt_clip)
                        exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)
                        update = normed_grad.add(exp_avg, alpha=amp_fac)
                        exp_avg_sq = update_second_moment(exp_avg_sq, grad, beta2)
                        cautious_grad = normed_grad

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_sq_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_sq_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (update * cautious_grad > 0).to(cautious_grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    if compass_second_moment_smoothing:
                        p_fp32.addcdiv_(update * mask, de_nom, value=-step_size)
                    else:
                        p_fp32.add_(update * mask, alpha=-step_size)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['exp_avg'], exp_avg)
                    if not group['factor_second_moment']:
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_sq_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss

def get_rms(tensor:torch.tensor):
    return tensor.norm().div(math.sqrt(tensor.numel()))

class _CompassBase(Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        eps2,
        eps_floor,
        weight_decay,
        weight_decouple,
        stable_weight_decay,
        amp_fac,
        cautious,
        adaptive_clip,
        adaptive_clip_eps,
        adaptive_clip_type,
        debias_beta1,
        debias_beta2,
        adopt,
        mars_gamma,
        compass_second_moment_smoothing,
        update_strategy,
        stable_update,
        stable_update_clip_threshold,
        use_orthograd,
        use_spam_clipping,
        spam_clipping_type,
        spam_clipping_threshold,
        spam_clipping_start_step,
        spam_clipping_eps,
        use_spam_momentum_reset,
        spam_momentum_reset_warmup_steps,
        spam_momentum_reset_interval_steps,
        use_focus,
        focus_gamma,
        focus_beta,
        debug,
        use_exadam,
        *,
        block_size,
        min_quant_size,
        state_precision,
        torch_compile,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams','both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        # Override zero to tiny
        if spam_clipping_eps is None or spam_clipping_eps <= 0:
            spam_clipping_eps = torch.finfo(torch.float32).tiny

        if block_size is None:
            if state_precision == 'parameter':
                block_size = 0
            elif state_precision == 'q8bit':
                block_size = 256
            elif state_precision == 'q4bit':
                block_size = 128
            elif state_precision == 'qfp8':
                block_size = 256
            else:
                raise NotImplementedError

        defaults = dict(
            lr=torch.tensor(lr),
            betas=betas,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            stable_weight_decay=stable_weight_decay,
            amp_fac=amp_fac,
            cautious=cautious,
            adaptive_clip=adaptive_clip,
            adaptive_clip_eps=adaptive_clip_eps,
            adaptive_clip_type=adaptive_clip_type,
            debias_beta1=debias_beta1,
            debias_beta2=debias_beta2,
            adopt=adopt,
            mars_gamma=mars_gamma,
            compass_second_moment_smoothing=compass_second_moment_smoothing,
            update_strategy=update_strategy,
            stable_update=stable_update,
            stable_update_clip_threshold=stable_update_clip_threshold,
            use_orthograd=use_orthograd,
            use_spam_clipping=use_spam_clipping,
            spam_clipping_threshold=spam_clipping_threshold,
            spam_clipping_start_step=spam_clipping_start_step,
            spam_clipping_type=spam_clipping_type,
            spam_clipping_eps=spam_clipping_eps,
            use_spam_momentum_reset=use_spam_momentum_reset,
            spam_momentum_reset_warmup_steps=spam_momentum_reset_warmup_steps,
            spam_momentum_reset_interval_steps=spam_momentum_reset_interval_steps,
            use_focus=use_focus,
            focus_gamma=focus_gamma,
            focus_beta=focus_beta,
            debug=debug,
            use_exadam=use_exadam,
        )
        super().__init__(params, defaults)
        self.block_size = block_size
        self.min_quant_size = min_quant_size
        self.state_precision = state_precision
        self.torch_compile = torch_compile

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            device = group["params"][0].device

            group.setdefault("amp_fac", 2.0)
            group.setdefault("cautious", False)
            group.setdefault("adaptive_clip", 1.0)
            group.setdefault("adaptive_clip_eps", 1e-3)
            group.setdefault("adaptive_clip_type", 'layer')
            group.setdefault("debias_beta1", True)
            group.setdefault("debias_beta2", True)
            group.setdefault("adopt", False)
            group.setdefault("mars_gamma", 0.0)
            group.setdefault("eps2", 1e-3)
            group.setdefault("eps_floor", None)
            group.setdefault("weight_decouple", True)
            group.setdefault("stable_weight_decay", False)
            group.setdefault("compass_second_moment_smoothing", True)
            group.setdefault("swd_second_moment_mean_sqrt", torch.tensor(1.0, dtype=torch.float32, device=device))
            group.setdefault("update_strategy", 'unmodified')
            group.setdefault("stable_update", False)
            group.setdefault("stable_update_clip_threshold", 1.0)
            group.setdefault("use_orthograd", False)
            group.setdefault("use_spam_clipping", False)
            group.setdefault("spam_clipping_threshold", 500.0)
            group.setdefault("spam_clipping_start_step", 20)
            group.setdefault("spam_clipping_type", 'element')
            group.setdefault("spam_clipping_eps", None)
            group.setdefault("use_spam_momentum_reset", False)
            group.setdefault("spam_momentum_reset_warmup_steps", 20)
            group.setdefault("spam_momentum_reset_interval_steps", 41)
            group.setdefault("spam_momentum_reset_warmup_scheduler", CosineDecay(0.99, group.get("spam_momentum_reset_warmup_steps")))
            group.setdefault("spam_momentum_reset_warmup_scheduler_current_step", group.get("spam_momentum_reset_warmup_steps"))
            group.setdefault("spam_warmup_scaling_factor", torch.tensor(1.0, dtype=torch.float32, device=device))
            group.setdefault("step", 0)
            group.setdefault("use_focus", False)
            group.setdefault("focus_gamma", 0.1)
            group.setdefault("focus_beta", 0.9)
            group.setdefault("debug", False)
            group.setdefault("use_exadam", False)


    # bring your own function to create zero-filled subclass
    def _subclass_zeros(self, p: torch.Tensor, signed: bool, block_size: int):
        if self.state_precision == 'parameter':
            return torch.zeros_like(p)
        elif self.state_precision == 'q8bit':
            return OptimState8bit.zeros(p.shape, signed, block_size, p.device)
        elif self.state_precision == 'q4bit':
            return OptimState4bit.zeros(p.shape, signed, block_size, p.device)
        elif self.state_precision == 'qfp8':
            return OptimStateFp8.zeros(p.shape, block_size, p.device)
        else:
            raise NotImplementedError

    def _new_buffer(self, p: torch.Tensor, signed: bool):
        local_p = p.to_local() if isinstance(p, DTensor) else p

        # only quantize tensors >= min_quant_size values, 4096 original default here and in bitsandbytes
        if self.block_size != 0 and (local_p.numel() >= self.min_quant_size and local_p.numel() % self.block_size == 0):
            out = self._subclass_zeros(local_p, signed, self.block_size)
        else:
            out = torch.zeros_like(local_p)

        # wrap subclass in DTensor as needed
        # NOTE: local tensor may have different shapes across ranks.
        # this happens when the 1st dim is not divisible by WORLD_SIZE.
        # thus, we must supply shape (and stride) to DTensor.from_local()
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # for a given model, the number of different argument combinations to single_param_adam() is fixed.
        # thus, it is safe to disable cache limit without the risk of always re-compiling.
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                if 'step' in group:
                    group['step'] += 1
                else:
                    group['step'] = 1

                device = group["params"][0].device

                if 'swd_second_moment_mean_sqrt' not in group:
                    group['swd_second_moment_mean_sqrt'] = torch.tensor(1.0, dtype=torch.float32, device=device)

                swd_param_size_sum = 0
                swd_second_moment_group_sum = 0.0
                mars_gamma = group["mars_gamma"]
                beta1 = group["betas"][0]

                if 'spam_warmup_scaling_factor' not in group:
                    group["spam_warmup_scaling_factor"] = torch.tensor(1.0, dtype=torch.float32, device=device)

                if group["use_spam_momentum_reset"]:
                    group["spam_warmup_scaling_factor"].fill_(1 - group["spam_momentum_reset_warmup_scheduler"].get_dr(group["spam_momentum_reset_warmup_scheduler_current_step"]))
                    group["spam_momentum_reset_warmup_scheduler_current_step"] += 1
                else:
                    group["spam_warmup_scaling_factor"].fill_(1.0)

                if group["use_spam_momentum_reset"] and group['step'] % group["spam_momentum_reset_interval_steps"] == 0:
                    reset_momentum = True
                else:
                    reset_momentum = False

                apply_spam_clipping = False
                if group["use_spam_clipping"] and group["step"] >= group["spam_clipping_start_step"]:
                    if group["use_spam_momentum_reset"]:
                        if group["spam_momentum_reset_warmup_scheduler_current_step"] % group["spam_momentum_reset_interval_steps"] >= group["spam_clipping_start_step"]:
                            apply_spam_clipping = True
                    else:
                        apply_spam_clipping = True

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = torch.tensor(0, device=p.device, dtype=torch.int32)
                        if group["weight_decay"] > 0 and group['weight_decouple'] and group['stable_weight_decay']:
                            state["swd_second_moment_parameter_sum"] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        if group["use_focus"]:
                            state["pbar"] = self._new_buffer(p, True)
                        if mars_gamma > 0:
                            state["previous_grad"] = self._new_buffer(p, True)

                            if state["previous_grad"].dtype == torch.bfloat16:
                                state["previous_grad"].copy_(_fp32_to_bf16_sr(p.grad.float()))
                            else:
                                state["previous_grad"].copy_(p.grad.float())


                    state["step"] = state["step"].add_(1)

                    if not isinstance(group["lr"], torch.Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )

                    if group["weight_decay"] > 0 and group['stable_weight_decay']:
                        swd_param_size_sum += p.numel()

                    if group["adopt"] and state["step"] == 1:
                        grad_f32 = grad.float()
                        p_f32 = p.float()

                        if mars_gamma > 0:
                            # MARS Calculate cₜ (gradient with correction term)
                            previous_grad_f32 = torch.zeros_like(grad_f32, dtype=torch.float32).copy_(state["previous_grad"].float())
                            temp_grad_f32 = grad_f32.clone().detach()
                            grad_f32 = (grad_f32 - previous_grad_f32).mul_(mars_gamma * (beta1 / (1.0 - beta1))).add_(grad_f32)

                            if state["previous_grad"].dtype == torch.bfloat16:
                                state["previous_grad"].copy_(_fp32_to_bf16_sr(temp_grad_f32))
                            else:
                                state["previous_grad"].copy_(temp_grad_f32)

                        if group["use_orthograd"]:
                            _paper_orthograd(p_f32, grad_f32)

                        if group["adaptive_clip"] > 0:
                            grad_f32 = agc(p=p_f32, 
                                           grad=grad_f32, 
                                           agc_clip_val=group["adaptive_clip"], 
                                           agc_eps=group["adaptive_clip_eps"], 
                                           norm_type=group["adaptive_clip_type"])
                        
                        #Make a fp32 copy of exp_avg_sq_f32
                        exp_avg_sq_f32 = torch.zeros_like(p.float(), dtype=torch.float32).copy_(state["exp_avg_sq"].float())
                        exp_avg_sq_f32.add_(grad_f32.square())

                        if state["exp_avg_sq"].dtype == torch.bfloat16:
                            state["exp_avg_sq"].copy_(_fp32_to_bf16_sr(exp_avg_sq_f32))
                        else:
                            state["exp_avg_sq"].copy_(exp_avg_sq_f32)
                    else:
                        if group["debug"] and p.numel() >= 2 and apply_spam_clipping:
                            grad_f32 = grad.float()
                            spam_grad_clipping_logging(grad=grad_f32, second_moment=state["exp_avg_sq"].float(), 
                                                       clip_threshold=group["spam_clipping_threshold"], 
                                                       clip_type=group["spam_clipping_type"],
                                                       spam_clip_eps=group["spam_clipping_eps"])

                        # without calling p.detach(), torch.compile() will have issues with FSDP2 in some cases
                        # https://github.com/pytorch/ao/issues/652#issuecomment-2285040894
                        # thus, by calling p.detach(), DTensor won't have .grad anymore, which is ok since we
                        # are passing grad separately anyway.
                        if self.torch_compile:
                            torch.compile(single_param_compass, fullgraph=True, dynamic=False)(
                                p=p.detach(),
                                grad=grad,
                                step=state["step"],
                                exp_avg=state["exp_avg"],
                                exp_avg_sq=state["exp_avg_sq"],
                                previous_grad=state["previous_grad"] if mars_gamma > 0 else None,
                                pbar=state["pbar"] if group["use_focus"] else None,
                                lr=group["lr"],
                                beta1=group["betas"][0],
                                beta2=group["betas"][1],
                                weight_decay=group["weight_decay"],
                                weight_decouple=group["weight_decouple"],
                                stable_weight_decay=group["stable_weight_decay"],
                                eps=group["eps"],
                                eps2=group["eps2"],
                                eps_floor=group["eps_floor"],
                                amp_fac=group["amp_fac"],
                                adaptive_clip=group["adaptive_clip"],
                                adaptive_clip_eps=group["adaptive_clip_eps"],
                                adaptive_clip_type=group["adaptive_clip_type"],
                                debias_beta1=group["debias_beta1"],
                                debias_beta2=group["debias_beta2"],
                                adopt=group["adopt"],
                                mars_gamma=group["mars_gamma"],
                                compass_second_moment_smoothing=group["compass_second_moment_smoothing"],
                                update_strategy=group["update_strategy"],
                                stable_update=group["stable_update"],
                                stable_update_clip_threshold=group["stable_update_clip_threshold"],
                                use_orthograd=group["use_orthograd"],
                                spam_clipping_threshold = group["spam_clipping_threshold"],
                                spam_clipping_type = group["spam_clipping_type"],
                                spam_clipping_eps = group["spam_clipping_eps"],
                                use_focus=group["use_focus"],
                                focus_gamma=group["focus_gamma"],
                                focus_beta=group["focus_beta"],
                                apply_spam_clipping = apply_spam_clipping,
                                reset_momentum = reset_momentum,
                                use_exadam=group["use_exadam"],
                                spam_warmup_scaling_factor = group["spam_warmup_scaling_factor"],
                                swd_second_moment_mean_sqrt=group['swd_second_moment_mean_sqrt'] if group["stable_weight_decay"] else None,
                                swd_second_moment_parameter_sum=state["swd_second_moment_parameter_sum"] if group["stable_weight_decay"] else None,
                            )
                        else:
                            single_param_compass(
                                p=p.detach(),
                                grad=grad,
                                step=state["step"],
                                exp_avg=state["exp_avg"],
                                exp_avg_sq=state["exp_avg_sq"],
                                previous_grad=state["previous_grad"] if mars_gamma > 0 else None,
                                pbar=state["pbar"] if group["use_focus"] else None,
                                lr=group["lr"],
                                beta1=group["betas"][0],
                                beta2=group["betas"][1],
                                weight_decay=group["weight_decay"],
                                weight_decouple=group["weight_decouple"],
                                stable_weight_decay=group["stable_weight_decay"],
                                eps=group["eps"],
                                eps2=group["eps2"],
                                eps_floor=group["eps_floor"],
                                amp_fac=group["amp_fac"],
                                adaptive_clip=group["adaptive_clip"],
                                adaptive_clip_eps=group["adaptive_clip_eps"],
                                adaptive_clip_type=group["adaptive_clip_type"],
                                debias_beta1=group["debias_beta1"],
                                debias_beta2=group["debias_beta2"],
                                adopt=group["adopt"],
                                mars_gamma=group["mars_gamma"],
                                compass_second_moment_smoothing=group["compass_second_moment_smoothing"],
                                update_strategy=group["update_strategy"],
                                stable_update=group["stable_update"],
                                stable_update_clip_threshold=group["stable_update_clip_threshold"],
                                use_orthograd=group["use_orthograd"],
                                spam_clipping_threshold = group["spam_clipping_threshold"],
                                spam_clipping_type = group["spam_clipping_type"],
                                spam_clipping_eps = group["spam_clipping_eps"],
                                use_focus=group["use_focus"],
                                focus_gamma=group["focus_gamma"],
                                focus_beta=group["focus_beta"],
                                apply_spam_clipping = apply_spam_clipping,
                                reset_momentum = reset_momentum,
                                use_exadam=group["use_exadam"],
                                spam_warmup_scaling_factor = group["spam_warmup_scaling_factor"],
                                swd_second_moment_mean_sqrt=group['swd_second_moment_mean_sqrt'] if group["stable_weight_decay"] else None,
                                swd_second_moment_parameter_sum=state["swd_second_moment_parameter_sum"] if group["stable_weight_decay"] else None,
                            )

                        if group["weight_decay"] > 0 and group['stable_weight_decay']:
                            swd_second_moment_group_sum += state["swd_second_moment_parameter_sum"].item()

                if group["use_spam_momentum_reset"] and group['step'] % group["spam_momentum_reset_interval_steps"] == 0:
                    group["spam_momentum_reset_warmup_scheduler_current_step"] = 0
                    group["spam_momentum_reset_warmup_scheduler"] = CosineDecay(0.99, group["spam_momentum_reset_warmup_steps"])

                if group["weight_decay"] > 0 and group['stable_weight_decay']:
                    swd_second_moment_mean_sqrt = math.sqrt(swd_second_moment_group_sum / swd_param_size_sum)
                    if group["debug"]:
                        logging.info(f"swd_second_moment_mean_sqrt={str(swd_second_moment_mean_sqrt)}")

                    if swd_second_moment_mean_sqrt > 0:
                        group['swd_second_moment_mean_sqrt'].fill_(swd_second_moment_mean_sqrt)
                    else:
                        group['swd_second_moment_mean_sqrt'].fill_(1.0)

                    if group["debug"]:
                        logging.info(f"resulting_stable_weight_decay_multiplier= {str(1.0 / group['swd_second_moment_mean_sqrt'])}")

        return loss


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
# and param tensor subclass that implements aten.add_.Tensor, and aten.addcdiv_.default
def single_param_compass(
    p: torch.Tensor,
    grad: torch.Tensor,
    step: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    previous_grad: Optional[torch.Tensor],
    pbar: Optional[torch.Tensor],
    lr: torch.Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    weight_decouple: bool,
    stable_weight_decay: bool,
    eps: float,
    eps2: float,
    eps_floor: Optional[float],
    amp_fac: float,
    adaptive_clip: float,
    adaptive_clip_eps: float,
    adaptive_clip_type: NORM_TYPE,
    debias_beta1: bool,
    debias_beta2: bool,
    adopt: bool,
    mars_gamma: float,
    compass_second_moment_smoothing: bool,
    update_strategy: UPDATE_STRATEGY,
    stable_update: bool,
    stable_update_clip_threshold: float,
    use_orthograd: bool,
    spam_clipping_threshold: float,
    spam_clipping_type: CLIP_TYPE,
    spam_clipping_eps: float,
    use_focus: bool,
    focus_gamma: float,
    focus_beta: float,
    apply_spam_clipping: bool,
    reset_momentum: bool,
    use_exadam: bool,
    spam_warmup_scaling_factor: torch.Tensor,
    swd_second_moment_mean_sqrt: torch.Tensor,
    swd_second_moment_parameter_sum: torch.Tensor,
):
    # compute in FP32 for accurate calculations
    p_f32 = p.float()
    grad_f32 = grad.float()

    beta1_t: float = beta1**step
    bias_correction1: float = 1.0
    if debias_beta1:
        bias_correction1 = 1 - beta1_t

    beta2_t: float = beta2**step
    bias_correction2: float = 1.0
    current_beta2 = beta2
    if debias_beta2:
        bias_correction2 = 1.0 - beta2_t

    #Make a fp32 copies of state
    exp_avg_sq_f32 = torch.zeros_like(p_f32, dtype=torch.float32).copy_(exp_avg_sq.float())
    exp_avg_f32 = torch.zeros_like(p_f32, dtype=torch.float32).copy_(exp_avg.float())

    if use_focus:
        pbar_f32 = torch.zeros_like(pbar, dtype=torch.float32).copy_(pbar.float())

    if reset_momentum:
        exp_avg_f32 = torch.zeros_like(exp_avg_f32)
        exp_avg_sq_f32 = torch.zeros_like(exp_avg_sq_f32)

    if mars_gamma > 0:
        # MARS Calculate cₜ (gradient with correction term)
        previous_grad_f32 = torch.zeros_like(grad_f32, dtype=torch.float32).copy_(previous_grad.float())
        temp_grad_f32 = grad_f32.clone().detach()
        grad_f32 = (grad_f32 - previous_grad_f32).mul_(mars_gamma * (beta1 / (1.0 - beta1))).add_(grad_f32)

        if previous_grad.dtype == torch.bfloat16:
            previous_grad.copy_(_fp32_to_bf16_sr(temp_grad_f32))
        else:
            previous_grad.copy_(temp_grad_f32)

    if use_orthograd:
         _paper_orthograd(p_f32, grad_f32)

    if spam_clipping_threshold != 0 and apply_spam_clipping and p.numel() >= 2 and p.ndim >= 1:
        grad_f32 = spam_grad_clipping(grad=grad_f32, second_moment=exp_avg_sq_f32, clip_threshold=spam_clipping_threshold, clip_type=spam_clipping_type, spam_clip_eps=spam_clipping_eps)

    if adaptive_clip > 0:
        grad_f32 = agc(p=p_f32, grad=grad_f32, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

    if eps_floor is not None and eps_floor < eps:
        rms_grad = grad_f32.pow(2).mean().sqrt_()
        curr_eps = max(min(eps, eps2 * rms_grad), eps_floor) # Set a floor for eps to avoid NaN
    else:
        curr_eps = eps

    if adopt:
        if use_exadam:
            # Compute the new debiasing terms
            d1: torch.Tensor = 1 + (exp_avg_sq_f32.div(exp_avg_sq_f32 + curr_eps)) * beta2_t
            d2: torch.Tensor = 1 + (exp_avg_f32.pow(2).div(exp_avg_f32.pow(2) + curr_eps)) * beta1_t

            v_tilde: torch.Tensor = exp_avg_sq_f32.div(bias_correction2) * d2

        if use_exadam:
            adopt_denom = v_tilde.sqrt().add_(curr_eps)
        else:
            adopt_denom = exp_avg_sq_f32.div(bias_correction2).sqrt().add_(curr_eps)
        adopt_clip: float = (step-1)**0.25
        if compass_second_moment_smoothing:
            scaled_adopt_clip = adopt_clip * adopt_denom
            normed_grad = grad_f32.clamp(-scaled_adopt_clip, scaled_adopt_clip)

            unnormed_exp_avg_f32 = exp_avg_f32.mul(beta1).add_(grad_f32, alpha=1.0 - beta1)
            exp_avg_f32.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)

            if use_exadam:
                # Bias-corrected gradient
                normed_grad = normed_grad.div(bias_correction1) * d1
                m_tilde: torch.Tensor = exp_avg_f32.div(bias_correction1) * d1
                update = normed_grad.add(m_tilde, alpha=amp_fac)
            else:
                update = normed_grad.add(normed_grad, alpha=amp_fac)

            unnormed_update = grad_f32.add(unnormed_exp_avg_f32, alpha=amp_fac)
            if use_exadam:
                # Bias-corrected gradient
                unnormed_update = grad_f32.div(bias_correction1) * d1
                m_tilde: torch.Tensor = unnormed_exp_avg_f32.div(bias_correction1) * d1
                unnormed_update = grad_f32.add(m_tilde, alpha=amp_fac)
            else:
                unnormed_update = grad_f32.add(unnormed_exp_avg_f32, alpha=amp_fac)

            exp_avg_sq_f32.mul_(current_beta2).addcmul_(unnormed_update, unnormed_update, value=1.0 - current_beta2)

            update_grad = grad_f32
            de_nom = adopt_denom
        else:
            normed_grad = grad_f32.div(adopt_denom).clamp(-adopt_clip, adopt_clip)
            exp_avg_f32.mul_(beta1).add_(normed_grad, alpha=1 - beta1)

            if use_exadam:
                # Bias-corrected gradient
                normed_grad = normed_grad.div(bias_correction1) * d1
                m_tilde: torch.Tensor = exp_avg_f32.div(bias_correction1) * d1
                update = normed_grad.add(m_tilde, alpha=amp_fac)
            else:
                update = normed_grad.add(exp_avg_f32, alpha=amp_fac)

            exp_avg_sq_f32.mul_(current_beta2).addcmul_(grad_f32, grad_f32, value=1 - current_beta2)
            
            update_grad = normed_grad
            de_nom = torch.tensor(1.0, device=p.device, dtype=torch.float32)
    else:
        exp_avg_f32.mul_(beta1).add_(grad_f32, alpha=1 - beta1)

        if use_exadam:
            # Compute the new debiasing terms
            d1: torch.Tensor = 1 + (exp_avg_sq_f32.div(exp_avg_sq_f32 + curr_eps)) * beta2_t
            d2: torch.Tensor = 1 + (exp_avg_f32.pow(2).div(exp_avg_f32.pow(2) + curr_eps)) * beta1_t

            v_tilde: torch.Tensor = exp_avg_sq_f32.div(bias_correction2) * d2

            # Bias-corrected gradient
            grad_f32 = grad_f32.div(bias_correction1) * d1
            m_tilde: torch.Tensor = exp_avg_f32.div(bias_correction1) * d1
            update = grad_f32.add(m_tilde, alpha=amp_fac)
        else:
            update = grad_f32.add(exp_avg_f32, alpha=amp_fac)

        if compass_second_moment_smoothing:
            exp_avg_sq_f32.mul_(current_beta2).addcmul_(update, update, value=1 - current_beta2)
        else:
            exp_avg_sq_f32.mul_(current_beta2).addcmul_(grad_f32, grad_f32, value=1 - current_beta2)

        update_grad = grad_f32
        if use_exadam:
            de_nom = v_tilde.sqrt().add_(curr_eps)
        else:
            de_nom = exp_avg_sq_f32.div(bias_correction2).sqrt().add_(curr_eps)

    update = update.mul(spam_warmup_scaling_factor)

    if weight_decay > 0 and stable_weight_decay:
        swd_second_moment_parameter_sum.copy_(exp_avg_sq_f32.div(bias_correction2).sum())

    if stable_update:
        rms = get_rms(update).div(stable_update_clip_threshold).clamp_min(1)
        update.mul_(1 / rms)

    if use_focus:
        # Compute update
        pbar_f32.mul_(focus_beta).add_(p_f32, alpha=1.0 - focus_beta)
        # Compute bias-corrected pbar
        pbar_hat = pbar / (1.0 - focus_beta ** step)
        update = torch.sign(update) + focus_gamma * torch.sign(p_f32 - pbar_hat)

    if exp_avg.dtype == torch.bfloat16:
        exp_avg.copy_(_fp32_to_bf16_sr(exp_avg_f32))
        exp_avg_sq.copy_(_fp32_to_bf16_sr(exp_avg_sq_f32))
        if use_focus:
            pbar.copy_(_fp32_to_bf16_sr(pbar_f32))
    else:
        exp_avg.copy_(exp_avg_f32)
        exp_avg_sq.copy_(exp_avg_sq_f32)
        if use_focus:
            pbar.copy_(pbar_f32)

    if update_strategy in {'cautious','grams','both'}:
        if update_strategy in {'cautious','both'}:
            mask = (update * update_grad > 0).to(update_grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update = update * mask
        if update_strategy in {'grams','both'}:
            update.copy_(torch.sign(update_grad) * update.abs())

    if weight_decay > 0 and stable_weight_decay:
        swd_scaling = 1.0 / swd_second_moment_mean_sqrt
    else:
        swd_scaling = 1.0

    # Weight decay
    if weight_decay > 0 and weight_decouple and not use_focus:
        p_f32.mul_(1.0 - weight_decay * lr * swd_scaling)
    elif weight_decay > 0 and not use_focus:
        update.add_(p_f32, alpha=weight_decay * swd_scaling)

    if use_exadam:
        step_size = lr * torch.log(torch.sqrt(step + 1) * math.sqrt(2))
    else:
        step_size = lr / bias_correction1

    p_f32.addcdiv_(update, de_nom, value=-step_size)

    # Weight decay
    if weight_decay > 0 and use_focus:
        p_f32.add_(pbar_hat, alpha=-lr * weight_decay * swd_scaling)

    if p.dtype == torch.bfloat16:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)

class CompassAO(_CompassBase):
    r"""Compass supporting a number of optional features and quantization via torchao. 
        Requires Triton is fully setup for your environment, i.e. CUDA framework is installed with paths setup on Linux,
        and steps outlined at https://github.com/woct0rdho/triton-windows for Windows.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 1e-4).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.95, 0.999).
        eps (float):
            Term the denominator is minimally clamped to, to improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        stable_weight_decay (bool): 
            Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        amp_fac (float):
            amplification factor for the first moment filter (default: 2)
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
        adopt (bool)
            Updates the second moment / ema after it is used in a given step, as per ADOPT - https://arxiv.org/abs/2411.02853 (default: False)
        cautious (bool) (deprecated, use update strategy)
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
            Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
            and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
        mars_gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. Zero disables. (default: 0.0)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: True)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: True)
        compass_second_moment_smoothing (bool):
            Updates the second moment (i.e. ema / fim) with the Compass smoothed gradient. (Default: True)
        block_size (int):
            Controls the block sized used during quantization, will be automatically determined by state_precision if not set. 
            Advise not setting unless you have a clear reason to. (Default: None)
        min_quant_size (int):
            Controls the minimum size a tensor must be to be subject to quantization. 
            Advise not setting unless you have a clear reason to. (Default: 4096)
        state_precision (string):
            Determines the precision states should be stored at in the optimizer. Vaid values are 'parameter', 'q8bit', 'q4bit', 'qfp8'.
            Parameter sets the state to the same type as the parameter, i.e. no quantization is applied. (Default: parameter) 
    """

    def __init__(
        self,
        params,
        lr = 1e-4,
        betas=(0.95, 0.999),
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        stable_weight_decay: bool = False,
        amp_fac: float = 2.0,
        cautious: bool = False,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        debias_beta1: bool = True,
        debias_beta2: bool = True,
        adopt: bool = False,
        mars_gamma: float = 0.0,
        compass_second_moment_smoothing: bool = True,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        stable_update: bool = False,
        stable_update_clip_threshold: float = 1.0,
        use_orthograd: bool = False,
        use_spam_clipping: bool = False,
        spam_clipping_threshold: float = 500.0,
        spam_clipping_start_step: int = 20,
        spam_clipping_type: CLIP_TYPE = 'element',
        spam_clipping_eps: Optional[float] = None,
        use_spam_momentum_reset: bool = False,
        spam_momentum_reset_warmup_steps: int = 20,
        spam_momentum_reset_interval_steps: int = 41,
        use_focus: bool = False,
        focus_gamma: float = 0.1,
        focus_beta: float = 0.9,
        debug: bool = False,
        use_exadam: bool = False,
        *,
        block_size: Optional[int] = None,
        min_quant_size: int = 4096,
        state_precision: STATE_PRECISION = 'parameter',
        torch_compile: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            stable_weight_decay=stable_weight_decay,
            amp_fac=amp_fac,
            cautious=cautious,
            adaptive_clip=adaptive_clip,
            adaptive_clip_eps=adaptive_clip_eps,
            adaptive_clip_type=adaptive_clip_type,
            debias_beta1=debias_beta1,
            debias_beta2=debias_beta2,
            adopt=adopt,
            mars_gamma=mars_gamma,
            compass_second_moment_smoothing=compass_second_moment_smoothing,
            update_strategy=update_strategy,
            block_size=block_size,
            min_quant_size=min_quant_size,
            state_precision=state_precision,
            stable_update=stable_update,
            stable_update_clip_threshold=stable_update_clip_threshold,
            use_orthograd=use_orthograd,
            use_spam_clipping=use_spam_clipping,
            spam_clipping_type=spam_clipping_type,
            spam_clipping_threshold=spam_clipping_threshold,
            spam_clipping_start_step=spam_clipping_start_step,
            spam_clipping_eps=spam_clipping_eps,
            use_spam_momentum_reset=use_spam_momentum_reset,
            spam_momentum_reset_warmup_steps=spam_momentum_reset_warmup_steps,
            spam_momentum_reset_interval_steps=spam_momentum_reset_interval_steps,
            use_focus=use_focus,
            focus_gamma=focus_gamma,
            focus_beta=focus_beta,
            debug=debug,
            use_exadam=use_exadam,
            torch_compile=torch_compile,
        )

