# Source: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/wrap_schedulefree.py
# Modified to be an actual optimizer, allowing it to wrap any optimizer and work in Kohya's
from typing import Dict, Optional,Literal

import torch
from torch.optim import Optimizer
import math
import logging
from collections import defaultdict

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.base.exception import NoSparseGradientError
from .utils import (copy_stochastic_, NORM_TYPE, agc, 
                    STATE_PRECISION, _paper_orthograd, _paper_orthograd_compile, schedule_beta_tc, 
                    spam_grad_clipping, CLIP_TYPE, clean_dict_params,
                    CosineDecay, spam_grad_clipping_logging, stable_spam_clipping_tensors, SSCCosineDecay, adaptive_eps, OPTIMIZER)
from .low_bit_optim.quant_utils import _fp32_to_bf16_sr
from .low_bit_optim.subclass_8bit import OptimState8bit
from .low_bit_optim.subclass_4bit import OptimState4bit
from .low_bit_optim.subclass_fp8 import OptimStateFp8
from torch.distributed._tensor import DTensor



UPDATE_STRATEGY = Literal['unmodified','cautious','grams', 'both']

class ScheduleFreeWrapper(BaseOptimizer):
    r"""
        Wrap any optimizer to make it Schedule-Free. 
        
        This version uses a memory-efficient swap operation but may be slower than the reference version. In most cases
        the performance difference is negligible.
        For the best possible performance and memory-usage, Schedule-Free needs 
        to be directly integrated with the base optimizer.

        When using this version, you can disable the base optimizer's 
        momentum, as it's no longer necessary when using our wrapper's 
        momentum (although you can use both types of momentum if you want).

        If you set weight decay on the base optimizer, it computes weight decay
        at $z$. We offer the option to compute weight decay at $y$, via the 
        `weight_decay_at_y` parameter, which seems to give better results in 
        our experiments. This approach to decay only works correctly if the base
        optimizer uses group["lr"] as the current learning rate. 

        params (ParamGroup): 
            iterable of parameters to optimize or dicts defining parameter groups.
        base_optimizer (OPTIMIZER): 
            PyTorch optimizer object, in Kohya's pass in an additional optimizer arg called 
            base_optimizer_type and the fully qualified optimizer name. 
            e.x. 
                base_optimizer_type=LoraEasyCustomOptimizer.compass.Compass
                base_optimizer_type=LoraEasyCustomOptimizer.came.CAME
                base_optimizer_type=LoraEasyCustomOptimizer.adopt.ADOPT
                base_optimizer_type=torch.optim.AdamW
        sf_momentum (float): 
            Apply momentum on the outer optimizer (default 0.9)
        sf_weight_decay_at_y (float): 
            Weight decay calculated at the y point. Set weight decay on the 
            inner optimizer to instead calculate at z (default: 0.0).
        sf_r (float): Use polynomial weighting in the average 
            with power r (default 0.0).
        sf_weight_lr_power (float): The weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, 
                 params: ParamGroup,
                 base_optimizer : OPTIMIZER, 
                 sf_weight_decay_at_y : float = 0.0,
                 sf_momentum : float = 0.9,
                 sf_weight_lr_power : float = 2.0,
                 sf_r : float = 0.0,
                 **kwargs):
        
        self.validate_non_negative(sf_weight_decay_at_y, 'sf_weight_decay_at_y')
        self.validate_non_negative(sf_momentum, 'sf_momentum')
        self.validate_non_negative(sf_weight_lr_power, 'sf_weight_lr_power')
        self.validate_non_negative(sf_r, 'sf_r')

        self.sf_weight_decay_at_y = sf_weight_decay_at_y
        self.sf_weight_lr_power = sf_weight_lr_power
        self.sf_r = sf_r
        self.sf_momentum = sf_momentum
        self.train_mode = False

        defaults: Defaults = {'sf_weight_decay_at_y': sf_weight_decay_at_y, 'sf_momentum': sf_momentum, 'sf_weight_lr_power': sf_weight_lr_power, 'sf_r': sf_r}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        clean_kwargs = clean_dict_params(base_optimizer, kwargs, wrapped=True)

        self.base_optimizer = base_optimizer(self.param_groups, **clean_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'ScheduleFreeWrapper'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['sf_step'] = 0

            for p in group['params']:
                state = self.state[p]
                state['z'] = torch.clone(p, memory_format=torch.preserve_format)

        self.base_optimizer.reset()

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / self.sf_momentum)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - self.sf_momentum)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor):
        # Convert to uint8 while preserving dimensions by viewing as bytes
        x_bytes = x.view(-1).view(torch.uint8)
        y_bytes = y.view(-1).view(torch.uint8)
        
        # Perform bitwise XOR operations
        x_bytes.bitwise_xor_(y_bytes)
        y_bytes.bitwise_xor_(x_bytes)
        x_bytes.bitwise_xor_(y_bytes)

        # If this crashes use ScheduleFreeWrapperReference instead
        #x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        #y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        #x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if 'sf_step' in group:
                group['sf_step'] += 1
            else:
                group['sf_step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)

                z = state['z']

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.to(dtype=torch.float32, copy=True)
                    z = z.to(torch.float32)

                # Apply weight_decay_at_y
                if self.sf_weight_decay_at_y != 0.0:
                    z.sub_(p_fp32, alpha=lr*self.sf_weight_decay_at_y)    
                    p_fp32.sub_(p_fp32, alpha=lr*self.sf_weight_decay_at_y*(1-self.sf_momentum))

                # Unextrapolate p converting from y -> x
                p_fp32.lerp_(end=z, weight=1-1/self.sf_momentum)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(p, p_fp32)

                z = state["z"]

                # Swap x into z buffer temporarily
                self.swap(z, p)

                # Now state['z'] is x and p is z.

        #######
        # Apply step to z
        self.base_optimizer.step()

        ######
        for group in self.param_groups:
            weight_lr_power = self.sf_weight_lr_power
            r = self.sf_r
            # tiny bit of starting LR to avoid divide by zero
            lr = max(group['lr'] * 1.0, 1e-8)
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))
            
            weight = (group['sf_step']**r) * (lr_max**weight_lr_power)
            weight_sum = group['sf_weight_sum'] = group.get('sf_weight_sum', 0.0) + weight

            ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                z = state['z']

                # Swap x back out of z buffer, leaving p as x
                self.swap(z, p)

                # Now state['z'] is z and p is x.

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                # Update x
                p_fp32.lerp_(end=z.to(torch.float32), weight=ckp1)

                # Now set p to y
                p_fp32.lerp_(end=state['z'].to(torch.float32), weight=1-self.sf_momentum)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss
    
    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class ADOPTScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

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
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, exp_avg_sq = state['z'], state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq = z.to(torch.float32), exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)

                    update = grad.div(de_nom)
                    update.clamp_(-adopt_clip, adopt_clip)

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class ADOPTEMAMixScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + AdEMAMix slow ema.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, exponential moving average squared, and slow ema/momentum (default: 0.9, 0.9999, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        alpha (float): 
            usually between 2 and 5 would work well. (default: 2)
        t_alpha_beta3 (Optional[float]): 
            Steps to warmup alpha and beta 3. Total number of steps is recommended when needed. (Default: None)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = True,
        alpha: float = 2.0,
        t_alpha_beta3: Optional[float] = None,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'cautious': cautious,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTEMAMixScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]


            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3, 1e-8)


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
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_slow'] = torch.zeros_like(p)

                z, exp_avg_sq, exp_avg_slow = state['z'], state['exp_avg_sq'], state['exp_avg_slow']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq, exp_avg_slow = z.to(torch.float32), exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)
                    slow_ema_update = (alpha_t * exp_avg_slow).div(de_nom)
                    slow_ema_update.clamp_(-adopt_clip, adopt_clip)

                    grad_update = grad.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (slow_ema_update * grad_update > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        slow_ema_update.mul_(mask)

                    full_update = grad_update + slow_ema_update

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        full_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(full_update, alpha=adaptive_y_lr)

                    z.sub_(full_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class ADOPTNesterovScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + Adan style nesterov momentum.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, grad diff ema, and exponential moving average squared (default: 0.9, 0.92, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.92, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = True,
        debias_beta2: bool = False,
        debias_beta3: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'debias_beta2':debias_beta2,
            'debias_beta3':debias_beta3,
            'cautious': cautious,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTNesterovScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            if group["debias_beta3"]:
                bias_correction3: float = self.debias(beta3, group['step'])
            else:
                bias_correction3 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

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
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                z, exp_avg_sq, exp_avg_diff, grad_diff = state['z'], state['exp_avg_sq'], state['exp_avg_diff'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq, exp_avg_diff, grad_diff = z.to(torch.float32), exp_avg_sq.to(torch.float32), exp_avg_diff.to(torch.float32), grad_diff.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                grad_diff.add_(grad)

                if group['step'] == 1:
                    grad_diff.mul_(beta2).add_(grad)
                    exp_avg_sq.addcmul_(grad_diff, grad_diff.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction3).sqrt_().add_(curr_eps)
                    exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)

                    grad_diff.mul_(beta2).add_(grad)
                    exp_avg_sq.mul_(beta3).addcmul_(grad_diff, grad_diff.conj(), value=1 - beta3)
                    
                    ema_diff_update = exp_avg_diff.div(de_nom)
                    ema_diff_update.clamp_(-adopt_clip, adopt_clip)

                    grad_update = grad.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (ema_diff_update * grad_update > 0).to(grad_update.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        ema_diff_update.mul_(mask)

                    full_update = grad_update + ema_diff_update

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        full_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(full_update, alpha=adaptive_y_lr)

                    z.sub_(full_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction3).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state['exp_avg_diff'], exp_avg_diff)
                    copy_stochastic_(state['previous_grad'], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss

class ADOPTMARSScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + MARS Correction.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
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
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        gamma: float = 0.025,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'gamma':gamma,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTMARSScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

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
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]

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
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()

                z, exp_avg_sq = state['z'], state['exp_avg_sq']
                previous_grad = state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq = z.to(torch.float32), exp_avg_sq.to(torch.float32)
                    previous_grad = previous_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)
                if group['step'] == 1:
                    exp_avg_sq.addcmul_(c_t, c_t.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t.conj(), value=1.0 - beta2)

                    grad_update = c_t.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(grad_update, alpha=adaptive_y_lr)

                    z.sub_(grad_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['z'], z)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class FADOPTScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
            
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
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
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]

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
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)

                z, fim = state['z'], state['fim']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim = z.to(torch.float32), fim.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(grad, grad.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(grad, grad.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    update = grad_nat
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss

class FADOPTEMAMixScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT + AdEMAMix slow ema.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, exponential moving average squared, and slow ema/momentum (default: 0.9, 0.9999, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        alpha (float): 
            usually between 2 and 5 would work well. (default: 2)
        t_alpha_beta3 (Optional[float]): 
            Steps to warmup alpha and beta 3. Total number of steps is recommended when needed. (Default: None)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        cautious: bool = True,
        alpha: float = 2.0,
        t_alpha_beta3: Optional[float] = None,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'cautious':cautious,
            'alpha':alpha,
            't_alpha_beta3':t_alpha_beta3,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTEMAMixScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3, 1e-8)

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
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['exp_avg_slow'] = torch.ones_like(p)

                z, fim, exp_avg_slow = state['z'], state['fim'], state['exp_avg_slow']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, exp_avg_slow = z.to(torch.float32), fim.to(torch.float32), exp_avg_slow.to(torch.float32),
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(grad, grad.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(grad, grad.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    exp_avg_slow.mul_(beta3_t).add_(grad_nat, alpha=1.0 - beta3_t)
                    slow_ema_update = (alpha_t * exp_avg_slow)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (slow_ema_update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        slow_ema_update.mul_(mask)

                    update = grad_nat + slow_ema_update
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
    
class FADOPTNesterovScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, grad diff ema, and exponential moving average squared (default: 0.9, 0.92, 0.9999).
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
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.92, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        cautious: bool = True,
        debias_beta2: bool = False,
        debias_beta3: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'cautious':cautious,
            'debias_beta2':debias_beta2,
            'debias_beta3':debias_beta3,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTNesterovScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            if group["debias_beta3"]:
                current_beta3: float = self.debias_beta(beta3, group['step'])
            else:
                current_beta3 = beta3

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]

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
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                z, fim, exp_avg_diff, grad_diff = state['z'], state['fim'], state['exp_avg_diff'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, exp_avg_slow, grad_diff = z.to(torch.float32), fim.to(torch.float32), exp_avg_slow.to(torch.float32), grad_diff.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                grad_diff.add_(grad)

                if group['step'] == 1:
                    grad_diff.mul_(current_beta2).add_(grad)
                    fim.addcmul_(grad_diff, grad_diff.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    exp_avg_diff.mul_(current_beta2).add_(grad_diff, alpha=1.0 - current_beta2)

                    grad_diff.mul_(current_beta2).add_(grad)
                    fim.mul_(current_beta3).addcmul_(grad_diff, grad_diff.conj(), value=1 - current_beta3).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    ema_diff_update = exp_avg_diff

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (ema_diff_update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        ema_diff_update.mul_(mask)

                    update = grad_nat + ema_diff_update
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['exp_avg_diff'], exp_avg_diff)
                    copy_stochastic_(state['previous_grad'], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
    
class FADOPTMARSScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT + MARS Correction..
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
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
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        weight_decay_lr_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        gamma: float = 0.025,
        debias_beta2: bool = False,
        weight_decay_lr_max: Optional[float] = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        if weight_decay_lr_decouple:
            self.validate_non_negative(weight_decay_lr_max, 'weight_decay_lr_max')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'gamma': gamma,
            'debias_beta2':debias_beta2,
            'weight_decay_lr_decouple':weight_decay_lr_decouple,
            'weight_decay_lr_max':weight_decay_lr_max,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTMARSScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            fisher_clip = group["fisher_clip"]
            gamma = group["gamma"]
            weight_decay_lr_decouple = group["weight_decay_lr_decouple"]
            weight_decay_lr_max = group["weight_decay_lr_max"]
            weight_decay = group["weight_decay"]
            weight_decouple = group['weight_decouple']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()

                z, fim, previous_grad = state['z'], state['fim'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, previous_grad = z.to(torch.float32), fim.to(torch.float32), previous_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)
            
                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(c_t, c_t.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(c_t, c_t.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = c_t.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)
                    
                    # Perform weight decay
                    if weight_decay != 0 and weight_decouple:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - weight_decay * (lr if not weight_decay_lr_decouple else (lr / weight_decay_lr_max)) * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        grad_nat.add_(grad_weights, alpha=weight_decay)

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(grad_nat, alpha=adaptive_y_lr)

                    z.sub_(grad_nat, alpha=lr)

                if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['z'], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss

class _ADOPTAOScheduleFreeBase(Optimizer):
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
        adaptive_clip,
        adaptive_clip_eps,
        adaptive_clip_type,
        debias_beta2,
        use_beta2_warmup,
        beta2_warmup_initial,
        beta2_warmup_steps,
        mars_gamma,
        r,
        weight_lr_power,
        fisher,
        update_strategy,
        stable_update,
        stable_update_clip_threshold,
        atan2_denom,
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
        use_stable_spam_clipping,
        ssc_t_max,
        debug,
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
            
        self.train_mode = False

        defaults = dict(
            lr=torch.tensor(lr),
            betas=betas,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            stable_weight_decay=stable_weight_decay,
            adaptive_clip=adaptive_clip,
            adaptive_clip_eps=adaptive_clip_eps,
            adaptive_clip_type=adaptive_clip_type,
            debias_beta2=debias_beta2,
            use_beta2_warmup=use_beta2_warmup,
            beta2_warmup_initial=beta2_warmup_initial,
            beta2_warmup_steps=beta2_warmup_steps,
            mars_gamma=mars_gamma,
            r=r,
            weight_lr_power=weight_lr_power,
            fisher=fisher,
            update_strategy=update_strategy,
            stable_update=stable_update,
            stable_update_clip_threshold=stable_update_clip_threshold,
            atan2_denom=atan2_denom,
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
            use_stable_spam_clipping=use_stable_spam_clipping,
            ssc_t_max=ssc_t_max,
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

            group.setdefault("adaptive_clip", 1.0)
            group.setdefault("adaptive_clip_eps", 1e-3)
            group.setdefault("adaptive_clip_type", 'layer')
            group.setdefault("debias_beta2", False)
            group.setdefault("mars_gamma", 0.0)
            group.setdefault("eps2", 1e-3)
            group.setdefault("eps_floor", None)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("stable_weight_decay", False)
            group.setdefault("weight_decouple", False)
            group.setdefault("r", 0.0)
            group.setdefault("weight_lr_power", 2.0)
            group.setdefault("swd_second_moment_mean_sqrt", torch.tensor(1.0, dtype=torch.float32, device=device))
            group.setdefault("train_mode", False)
            group.setdefault("fisher", False)
            group.setdefault("update_strategy", 'unmodified')
            group.setdefault("stable_update", False)
            group.setdefault("stable_update_clip_threshold", 1.0)
            group.setdefault("atan2_denom", False)
            group.setdefault("use_orthograd", False)
            group.setdefault("use_spam_clipping", False)
            group.setdefault("spam_clipping_threshold", 500.0)
            group.setdefault("spam_clipping_start_step", 20)
            group.setdefault("spam_clipping_type", 'element')
            group.setdefault("spam_clipping_eps", None)
            group.setdefault("use_spam_momentum_reset", False)
            group.setdefault("spam_momentum_reset_warmup_steps", 20)
            group.setdefault("spam_momentum_reset_interval_steps", 41)
            #Mark CosineDecay as safe for deserialization
            torch.serialization.add_safe_globals([CosineDecay, torch.optim.SGD, defaultdict, dict, torch.optim.lr_scheduler.CosineAnnealingLR, SSCCosineDecay])
            group.setdefault("spam_momentum_reset_warmup_scheduler", CosineDecay(0.99, group.get("spam_momentum_reset_warmup_steps")))
            group.setdefault("spam_momentum_reset_warmup_scheduler_current_step", group.get("spam_momentum_reset_warmup_steps"))
            group.setdefault("spam_warmup_scaling_factor", torch.tensor(1.0, dtype=torch.float32, device=device))
            group.setdefault("beta2_warmup_initial", 0.9)
            group.setdefault("beta2_warmup_steps", 1)
            group.setdefault("step", 0)
            group.setdefault("use_focus", False)
            group.setdefault("focus_gamma", 0.1)
            group.setdefault("focus_beta", 0.9)
            group.setdefault("debug", False)
            group.setdefault("use_stable_spam_clipping", False)
            group.setdefault("ssc_t_max", None)
            group.setdefault("ssc_warmup", SSCCosineDecay(1.0, group['ssc_t_max'], eta_min=0.5) if group['use_stable_spam_clipping'] and group['ssc_t_max'] is not None else None)
            

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
        
    # bring your own function to create zero-filled subclass
    def _subclass_ones(self, p: torch.Tensor, signed: bool, block_size: int):
        if self.state_precision == 'parameter':
            return torch.ones_like(p)
        elif self.state_precision == 'q8bit':
            return OptimState8bit.ones(p.shape, signed, block_size, p.device)
        elif self.state_precision == 'q4bit':
            return OptimState4bit.ones(p.shape, signed, block_size, p.device)
        elif self.state_precision == 'qfp8':
            return OptimStateFp8.ones(p.shape, block_size, p.device)
        else:
            raise NotImplementedError

    def _new_buffer(self, p: torch.Tensor, signed: bool, init_value: str = 'zeros'):
        local_p = p.to_local() if isinstance(p, DTensor) else p

        # only quantize tensors >= min_quant_size values, 4096 original default here and in bitsandbytes
        if self.block_size != 0 and (local_p.numel() >= self.min_quant_size and local_p.numel() % self.block_size == 0):
            if init_value == 'zeros':
                out = self._subclass_zeros(local_p, signed, self.block_size)
            elif init_value == 'ones':
                out = self._subclass_ones(local_p, signed, self.block_size)
        else:
            if init_value == 'zeros':
                out = torch.zeros_like(local_p)
            elif init_value == 'ones':
                out = torch.ones_like(local_p)

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
    
    @staticmethod
    @torch.no_grad()
    def _eval(p: torch.Tensor, z: torch.Tensor, beta1: float):
        p_f32 = p.float()

        p_f32.data.lerp_(end=z.float(), weight=1.0 - 1.0 / beta1)

        if p.dtype == torch.bfloat16:
            p.copy_(_fp32_to_bf16_sr(p_f32))
        else:
            p.copy_(p_f32)

    
    @torch.no_grad()
    def eval(self):
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                if self.train_mode:
                    for p in group['params']:
                        state = self.state[p]
                        if 'z' in state:
                            if self.torch_compile:
                                torch.compile(self._eval, fullgraph=True, dynamic=False)(p=p, z=state["z"], beta1=group['betas'][0])
                            else:
                                self._eval(p=p, z=state["z"], beta1=group['betas'][0])
                    self.train_mode = False

    @staticmethod
    @torch.no_grad()
    def _train(p: torch.Tensor, z: torch.Tensor, beta1: float):
        p_f32 = p.float()

        p_f32.data.lerp_(end=z.float(), weight=1.0 - beta1)

        if p.dtype == torch.bfloat16:
            p.copy_(_fp32_to_bf16_sr(p_f32))
        else:
            p.copy_(p_f32)

    @torch.no_grad()
    def train(self):
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                if not self.train_mode:
                    for p in group['params']:
                        state = self.state[p]
                        if 'z' in state:
                            if self.torch_compile:
                                torch.compile(self._train, fullgraph=True, dynamic=False)(p=p, z=state["z"], beta1=group['betas'][0])
                            else:
                                self._train(p=p, z=state["z"], beta1=group['betas'][0])
                    self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

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

                if 'swd_second_moment_mean_sqrt' not in group:
                    device = group["params"][0].device
                    group['swd_second_moment_mean_sqrt'] = torch.tensor(1.0, dtype=torch.float32, device=device)

                swd_param_size_sum = 0
                swd_second_moment_group_sum = 0.0
                mars_gamma = group["mars_gamma"]
                beta1 = group["betas"][0]
                fisher = group["fisher"]
                use_stable_spam_clipping = group["use_stable_spam_clipping"]
                ssc_warmup = group["ssc_warmup"]

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
                        if group["weight_decay"] > 0 and group['stable_weight_decay']:
                            state["swd_second_moment_parameter_sum"] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                        state["z"] = self._new_buffer(p, True)
                        if state["z"].dtype == torch.bfloat16:
                            state["z"].copy_(_fp32_to_bf16_sr(p.float()))
                        else:
                            state["z"].copy_(p.float())

                        state["exp_avg_sq"] = self._new_buffer(p, False, 'ones' if fisher else 'zeros')
                        state["sf_lr_max"] = torch.tensor(-1.0, device=p.device, dtype=torch.float32)
                        state["sf_weight_sum"] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                        if group["use_focus"]:
                            state["pbar"] = self._new_buffer(p, True)
                        if mars_gamma > 0:
                            state["previous_grad"] = self._new_buffer(p, True)

                            if state["previous_grad"].dtype == torch.bfloat16:
                                state["previous_grad"].copy_(_fp32_to_bf16_sr(p.grad.float()))
                            else:
                                state["previous_grad"].copy_(p.grad.float())
                        if use_stable_spam_clipping:
                            state['ssc_scale'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                            state['ssc_m_norm_t'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                            state['ssc_v_norm_t'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                            state['ssc_m_max_t'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)

                    state["step"] = state["step"].add_(1)

                    if use_stable_spam_clipping:
                        state['ssc_scale'].copy_(ssc_warmup.get_death_rate(state['step']) if ssc_warmup is not None else 1.0)

                    if not isinstance(group["lr"], torch.Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )
                    
                    if group["weight_decay"] > 0 and group['stable_weight_decay']:
                        swd_param_size_sum += p.numel()

                    sf_lr_max = state['sf_lr_max'].copy_(torch.max(group["lr"], state['sf_lr_max']))
                    weight = (state['step'] ** group['r']) * (sf_lr_max ** group['weight_lr_power'])
                    sf_weight_sum = state['sf_weight_sum'].copy_(state['sf_weight_sum'] + weight)

                    checkpoint = (weight / sf_weight_sum).to(device=p.device, dtype=torch.float32) if sf_weight_sum != 0.0 else torch.tensor(0.0, device=p.device, dtype=torch.float32)
                    adaptive_y_lr = group["lr"] * (beta1 * (1.0 - checkpoint) - 1).to(device=p.device, dtype=torch.float32)
                    adopt_clip = ((state['step'].sub(1))**0.25).to(device=p.device, dtype=torch.float32)
                    
                    if state["step"] == 1:
                        grad_f32 = grad.float()
                        p_f32 = p.float()

                        if group["use_orthograd"]:
                            _paper_orthograd(p_f32, grad_f32)

                        if group["adaptive_clip"] > 0:
                            grad_f32 = agc(p=p_f32, 
                                           grad=grad_f32, 
                                           agc_clip_val=group["adaptive_clip"], 
                                           agc_eps=group["adaptive_clip_eps"], 
                                           norm_type=group["adaptive_clip_type"])
                            
                        exp_avg_sq_f32 = state["exp_avg_sq"].float()
                        exp_avg_sq_f32.add_(grad_f32.square())

                        if fisher:
                            # ADOPT clip
                            exp_avg_sq_f32.clamp_(-adopt_clip, adopt_clip)

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
                            torch.compile(single_param_ADOPTAOScheduleFree, fullgraph=True, dynamic=False)(
                                p=p.detach(),
                                grad=grad,
                                step=state["step"],
                                z=state["z"],
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
                                adaptive_clip=group["adaptive_clip"],
                                adaptive_clip_eps=group["adaptive_clip_eps"],
                                adaptive_clip_type=group["adaptive_clip_type"],
                                debias_beta2=group["debias_beta2"],
                                use_beta2_warmup=group["use_beta2_warmup"],
                                beta2_warmup_initial=group["beta2_warmup_initial"],
                                beta2_warmup_steps=group["beta2_warmup_steps"],
                                mars_gamma=group["mars_gamma"],
                                fisher=group["fisher"],
                                update_strategy=group["update_strategy"],
                                stable_update=group["stable_update"],
                                stable_update_clip_threshold=group["stable_update_clip_threshold"],
                                atan2_denom=group["atan2_denom"],
                                use_orthograd=group["use_orthograd"],
                                spam_clipping_threshold = group["spam_clipping_threshold"],
                                spam_clipping_type = group["spam_clipping_type"],
                                spam_clipping_eps = group["spam_clipping_eps"],
                                use_focus=group["use_focus"],
                                focus_gamma=group["focus_gamma"],
                                focus_beta=group["focus_beta"],
                                apply_spam_clipping = apply_spam_clipping,
                                reset_momentum = reset_momentum,
                                spam_warmup_scaling_factor = group["spam_warmup_scaling_factor"],
                                adopt_clip=adopt_clip,
                                sf_checkpoint=checkpoint,
                                sf_adaptive_y_lr=adaptive_y_lr,
                                swd_second_moment_mean_sqrt=group['swd_second_moment_mean_sqrt'] if group["stable_weight_decay"] and group["weight_decay"] > 0 else None,
                                swd_second_moment_parameter_sum=state["swd_second_moment_parameter_sum"] if group["stable_weight_decay"] and group["weight_decay"] > 0 else None,
                                use_stable_spam_clipping=group["use_stable_spam_clipping"],
                                ssc_scale=state['ssc_scale'] if group["use_stable_spam_clipping"] else None,
                                ssc_m_norm_t=state['ssc_m_norm_t'] if group["use_stable_spam_clipping"] else None,
                                ssc_v_norm_t=state['ssc_v_norm_t'] if group["use_stable_spam_clipping"] else None,
                                ssc_m_max_t=state['ssc_m_max_t'] if group["use_stable_spam_clipping"] else None,
                            )
                        else:
                            single_param_ADOPTAOScheduleFree(
                                p=p.detach(),
                                grad=grad,
                                step=state["step"],
                                z=state["z"],
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
                                adaptive_clip=group["adaptive_clip"],
                                adaptive_clip_eps=group["adaptive_clip_eps"],
                                adaptive_clip_type=group["adaptive_clip_type"],
                                debias_beta2=group["debias_beta2"],
                                use_beta2_warmup=group["use_beta2_warmup"],
                                beta2_warmup_initial=group["beta2_warmup_initial"],
                                beta2_warmup_steps=group["beta2_warmup_steps"],
                                mars_gamma=group["mars_gamma"],
                                fisher=group["fisher"],
                                update_strategy=group["update_strategy"],
                                stable_update=group["stable_update"],
                                stable_update_clip_threshold=group["stable_update_clip_threshold"],
                                atan2_denom=group["atan2_denom"],
                                use_orthograd=group["use_orthograd"],
                                spam_clipping_threshold = group["spam_clipping_threshold"],
                                spam_clipping_type = group["spam_clipping_type"],
                                spam_clipping_eps = group["spam_clipping_eps"],
                                use_focus=group["use_focus"],
                                focus_gamma=group["focus_gamma"],
                                focus_beta=group["focus_beta"],
                                apply_spam_clipping = apply_spam_clipping,
                                reset_momentum = reset_momentum,
                                spam_warmup_scaling_factor = group["spam_warmup_scaling_factor"],
                                adopt_clip=adopt_clip,
                                sf_checkpoint=checkpoint,
                                sf_adaptive_y_lr=adaptive_y_lr,
                                swd_second_moment_mean_sqrt=group['swd_second_moment_mean_sqrt'] if group["stable_weight_decay"] and group["weight_decay"] > 0 else None,
                                swd_second_moment_parameter_sum=state["swd_second_moment_parameter_sum"] if group["stable_weight_decay"] and group["weight_decay"] > 0 else None,
                                use_stable_spam_clipping=group["use_stable_spam_clipping"],
                                ssc_scale=state['ssc_scale'] if group["use_stable_spam_clipping"] else None,
                                ssc_m_norm_t=state['ssc_m_norm_t'] if group["use_stable_spam_clipping"] else None,
                                ssc_v_norm_t=state['ssc_v_norm_t'] if group["use_stable_spam_clipping"] else None,
                                ssc_m_max_t=state['ssc_m_max_t'] if group["use_stable_spam_clipping"] else None,
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


def get_rms(tensor:torch.tensor):
    return tensor.norm().div(math.sqrt(tensor.numel()))

# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
# and param tensor subclass that implements aten.add_.Tensor, and aten.addcdiv_.default
def single_param_ADOPTAOScheduleFree(
    p: torch.Tensor,
    grad: torch.Tensor,
    step: torch.Tensor,
    z: torch.Tensor,
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
    adaptive_clip: float,
    adaptive_clip_eps: float,
    adaptive_clip_type: NORM_TYPE,
    debias_beta2: bool,
    use_beta2_warmup: bool,
    beta2_warmup_initial: float,
    beta2_warmup_steps: int,
    mars_gamma: float,
    fisher: bool,
    update_strategy: UPDATE_STRATEGY,
    stable_update: bool,
    stable_update_clip_threshold: float,
    atan2_denom: bool,
    use_orthograd: bool,
    spam_clipping_threshold: float,
    spam_clipping_type: CLIP_TYPE,
    spam_clipping_eps: float,
    use_focus: bool,
    focus_gamma: float,
    focus_beta: float,
    apply_spam_clipping: bool,
    reset_momentum: bool,
    spam_warmup_scaling_factor: torch.Tensor,
    adopt_clip: torch.Tensor,
    sf_checkpoint: torch.Tensor,
    sf_adaptive_y_lr: torch.Tensor,
    swd_second_moment_mean_sqrt: torch.Tensor,
    swd_second_moment_parameter_sum: torch.Tensor,
    use_stable_spam_clipping: bool,
    ssc_scale=torch.Tensor,
    ssc_m_norm_t=torch.Tensor,
    ssc_v_norm_t=torch.Tensor,
    ssc_m_max_t= torch.Tensor,
):
    # compute in FP32 for accurate calculations
    p_f32 = p.float()
    grad_f32 = grad.float()

    y_f32 = p_f32  # Notation to match theory

    bias_correction2: float = 1.0
    current_beta2 = beta2
    if debias_beta2:
        if fisher:
            current_beta2 = ((beta2**step - beta2) / (beta2**step - 1.0)) ** (1/2)
        else:
            bias_correction2 = 1.0 - beta2**step

    if use_beta2_warmup:
        current_beta2 = schedule_beta_tc(beta2_warmup_steps, step, beta2_warmup_initial, beta2)

    #Make fp32 copies of state
    exp_avg_sq_f32 = torch.zeros_like(y_f32, dtype=torch.float32).copy_(exp_avg_sq.float())
    z_f32 = torch.zeros_like(y_f32, dtype=torch.float32).copy_(z.float())

    if use_focus:
        pbar_f32 = torch.zeros_like(pbar, dtype=torch.float32).copy_(pbar.float())

    if reset_momentum:
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
        _paper_orthograd_compile(y_f32, grad_f32)

    if spam_clipping_threshold != 0 and apply_spam_clipping and p.numel() >= 2 and p.ndim >= 1:
        grad_f32 = spam_grad_clipping(grad=grad_f32, second_moment=exp_avg_sq_f32, clip_threshold=spam_clipping_threshold, clip_type=spam_clipping_type, spam_clip_eps=spam_clipping_eps)

    if adaptive_clip > 0:
        grad_f32 = agc(p=y_f32, grad=grad_f32, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

    if use_stable_spam_clipping:
        grad_f32 = stable_spam_clipping_tensors(ssc_m_norm_t=ssc_m_norm_t,ssc_v_norm_t=ssc_v_norm_t,ssc_m_max_t=ssc_m_max_t, grad=grad, step=step, scale=ssc_scale)

    if eps_floor is not None and eps_floor < eps:
        rms_grad = grad_f32.pow(2).mean().sqrt_()
        curr_eps = max(min(eps, eps2 * rms_grad), eps_floor) # Set a floor for eps to avoid NaN
    else:
        curr_eps = eps

    if fisher:
        fim_base = exp_avg_sq_f32.sqrt()
        exp_avg_sq_f32.mul_(current_beta2).addcmul_(grad_f32, grad_f32, value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

        if atan2_denom:
            grad_nat = grad_f32.atan2(fim_base)
        else:
            fim_base.add_(curr_eps)
            grad_nat = grad_f32.div(fim_base)
        rms = grad_nat.pow(2).mean().sqrt_()
        divisor = max(1.0, rms) / 1.0
        update = grad_nat.div_(divisor)
    else:
        de_nom = exp_avg_sq_f32.div(bias_correction2).sqrt()
        
        if atan2_denom:
            # Approximate scaling for a regular Adam-style update.
            # Adam-atan2. Use atan2 rather than epsilon and division 
            # for parameter updates (https://arxiv.org/abs/2407.05872).
            # Has the nice property of "clipping" the gradient as well.
            update = grad_f32.atan2(de_nom).mul_(torch.tensor(1) / torch.tensor(math.atan(1))).clamp_(-adopt_clip, adopt_clip)   
        else:
            de_nom.add_(curr_eps)
            update = grad_f32.div(de_nom).clamp_(-adopt_clip, adopt_clip)   
        exp_avg_sq_f32.mul_(current_beta2).addcmul_(grad_f32, grad_f32, value=1 - current_beta2)
      
    if weight_decay > 0 and stable_weight_decay:
        swd_scaling = 1.0 / swd_second_moment_mean_sqrt
    else:
        swd_scaling = 1.0

    # Weight decay
    if weight_decay > 0 and weight_decouple and not use_focus:
        z_f32.add_(y_f32, alpha=-lr * weight_decay * swd_scaling)
        y_f32.add_(y_f32, alpha=-lr * weight_decay * (1.0 - beta1) * swd_scaling)

    elif weight_decay > 0 and not use_focus:
        if fisher:
            grad_weights = y_f32.div(fim_base)

            rms = grad_weights.pow(2).mean().sqrt_()
            divisor = max(1.0, rms) / 1.0 #fisher_clip
            grad_weights.div_(divisor)

            update.add_(grad_weights, alpha=weight_decay * swd_scaling)
        else:
            update.add_(y_f32, alpha=weight_decay * swd_scaling)

    update = update.mul(spam_warmup_scaling_factor)

    if stable_update:
        rms = get_rms(update).div(stable_update_clip_threshold).clamp_min(1)
        update.mul_(1 / rms)

    if use_focus:
        # Compute update
        pbar_f32.mul_(focus_beta).add_(y_f32, alpha=1.0 - focus_beta)
        # Compute bias-corrected pbar
        pbar_hat = pbar / (1.0 - focus_beta ** step)
        update = torch.sign(update) + focus_gamma * torch.sign(y_f32 - pbar_hat)

    if update_strategy in {'cautious','grams','both'}:
        y_update = (y_f32 - z_f32).mul_(sf_checkpoint).add_(update, alpha=-sf_adaptive_y_lr)

        if update_strategy in {'cautious','both'}:
            mask = (y_update * update > 0).to(update.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            y_update.mul_(mask)
        if update_strategy in {'grams','both'}:
            y_update.abs_().mul_(update.sign())
        y_f32.add_(y_update, alpha=-1)
    else:
        # These operations update y in-place,
        # without computing x explicitly.
        y_f32.lerp_(end=z_f32, weight=sf_checkpoint)
        y_f32.add_(update, alpha=sf_adaptive_y_lr)

    z_f32.add_(update, alpha=-lr)

    # Weight decay
    if weight_decay > 0 and use_focus:
        z_f32.add_(pbar_hat, alpha=-lr * weight_decay * swd_scaling)
        y_f32.add_(pbar_hat, alpha=-lr * weight_decay * (1.0 - beta1) * swd_scaling)

    if weight_decay > 0 and stable_weight_decay:
        swd_second_moment_parameter_sum.copy_(exp_avg_sq_f32.div(bias_correction2).sum())

    if exp_avg_sq.dtype == torch.bfloat16:
        exp_avg_sq.copy_(_fp32_to_bf16_sr(exp_avg_sq_f32))
        z.copy_(_fp32_to_bf16_sr(z_f32))
        if use_focus:
            pbar.copy_(_fp32_to_bf16_sr(pbar_f32))
    else:
        exp_avg_sq.copy_(exp_avg_sq_f32)
        z.copy_(z_f32)
        if use_focus:
            pbar.copy_(pbar_f32)

    if p.dtype == torch.bfloat16:
        p.copy_(_fp32_to_bf16_sr(y_f32))
    else:
        p.copy_(y_f32) 

class ADOPTAOScheduleFree(_ADOPTAOScheduleFreeBase):
    r"""Compass supporting a number of optional features and quantization via torchao. 
        Requires Triton is fully setup for your environment, i.e. CUDA framework is installed with paths setup on Linux,
        and steps outlined at https://github.com/woct0rdho/triton-windows for Windows.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 5e-4).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.999).
        eps (float):
            Term the denominator is minimally clamped to, to improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        stable_weight_decay (bool): 
            Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
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
        mars_gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. Zero disables. (default: 0.0)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: False)
        block_size (int):
            Controls the block sized used during quantization, will be automatically determined by state_precision if not set. 
            Advise not setting unless you have a clear reason to. (Default: None)
        min_quant_size (int):
            Controls the minimum size a tensor must be to be subject to quantization. 
            Advise not setting unless you have a clear reason to. (Default: 4096)
        state_precision (string):
            Determines the precision states should be stored at in the optimizer. Vaid values are 'parameter', 'q8bit', 'q4bit', 'qfp8'.
            Parameter sets the state to the same type as the parameter, i.e. no quantization is applied. (Default: parameter) 
        r (float): 
            ScheduleFree: use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            ScheduleFree: during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2.0)
        update_strategy (str)
            Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
            and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
        stable_update (boolean):
            Scales parameter updates by the root-mean-square of the normalised gradient, in essence identical to 
            Adafactor's gradient scaling. Set to False if the adaptive learning rate never improves.
            (default: False)
        atan2_denom (boolean). Use atan2 rather than epsilon and division for parameter updates (https://arxiv.org/abs/2407.05872).
            Has the nice property of "clipping" the gradient as well.
            (default: False)
    """

    def __init__(
        self,
        params,
        lr = 5e-4,
        betas=(0.85, 0.9998),
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        debias_beta2: bool = False,
        use_beta2_warmup: bool = False,
        beta2_warmup_initial: float = 0.9,
        beta2_warmup_steps: int = 0,
        mars_gamma: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        fisher: float = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        stable_update: bool = False,
        stable_update_clip_threshold: float = 1.0,
        atan2_denom: bool = False,
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
        use_stable_spam_clipping: bool = False,
        ssc_t_max: Optional[int] = None,
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
            adaptive_clip=adaptive_clip,
            adaptive_clip_eps=adaptive_clip_eps,
            adaptive_clip_type=adaptive_clip_type,
            debias_beta2=debias_beta2,
            use_beta2_warmup=use_beta2_warmup,
            beta2_warmup_initial=beta2_warmup_initial,
            beta2_warmup_steps=beta2_warmup_steps,
            mars_gamma=mars_gamma,
            r=r,
            weight_lr_power=weight_lr_power,
            fisher=fisher,
            block_size=block_size,
            min_quant_size=min_quant_size,
            state_precision=state_precision,
            update_strategy=update_strategy,
            stable_update=stable_update,
            stable_update_clip_threshold=stable_update_clip_threshold,
            atan2_denom=atan2_denom,
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
            use_stable_spam_clipping=use_stable_spam_clipping,
            ssc_t_max=ssc_t_max,
            debug=debug,
            torch_compile=torch_compile,
        )