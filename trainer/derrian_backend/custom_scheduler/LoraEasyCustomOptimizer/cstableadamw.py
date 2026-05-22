#

import math
import torch
from torch import Tensor # For type hinting

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from .utils import apply_weight_decay, _get_compiled_stable_spam_clipping, _stable_spam_clipping_impl


class CStableAdamW(BaseOptimizer):
    r"""CStableAdamW Optimizer with optional Stable SPAM Clipping.
        Stable and low-precision training for large-scale vision-language models,
    with optional Kahan summation, optional RMS-based scaling, optional Adam-atan2 updates,
    and optional ADOPT (Adam with Offload of the Previous grad for the second moment).

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param kahan_sum: bool. Enables Kahan summation for more accurate parameter updates when training in low precision.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay (à la AdamW).
    :param eps: float. term added to the denominator for numerical stability (when not using atan2).
    :param use_rms: bool. if True, uses RMS-based scaling for the final “effective” learning rate.
    :param use_atan2: bool. if True, replaces the standard division-based approach with an Adam-atan2 update step.
                      This step eliminates the typical eps-based denominator but changes the update rule.
    :param atan2_a: float. scaling factor for the Adam-atan2 update. Default=1.0 or 1.27 are typical usage.
    :param atan2_b: float. scaling factor inside the atan2 denominator. Default=1.0 for typical usage.
    :param cautious_factor: float ∈ [0,1]. If < 1.0, the update is “cautious” w.r.t. gradient alignment:
                           directions that do not align with the gradient are scaled down.
    :param use_adopt: bool. If True, apply the ADOPT modification, using the *previous* grad for exp_avg_sq
                      to break correlation between current grad and second moment.
    :param use_stable_spam_clipping: bool. If True, enables Stable SPAM clipping on gradients.
    :param ssc_scale: float. Scale factor used within Stable SPAM clipping's m_norm_t update.
    :param ssc_gamma1: float. Decay rate for ssc_m_norm_t (running average of grad norms).
    :param ssc_gamma2: float. Decay rate for ssc_v_norm_t (running average of squared grad norms).
    :param ssc_gamma3: float. Decay rate for ssc_m_max_t (running average of max absolute grad).
    :param ssc_eps_floor: float. Floor value for the epsilon used within Stable SPAM clipping.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,  # Main epsilon for Adam updates
        use_rms: bool = False,
        use_atan2: bool = False,
        atan2_a: float = 1.27,
        atan2_b: float = 1.0,
        cautious_factor: float = 1.0,
        use_adopt: bool = False,
        # New parameters for stable_spam_clipping
        use_stable_spam_clipping: bool = False,
        ssc_scale: float = 1.0,
        ssc_gamma1: float = 0.85,
        ssc_gamma2: float = 0.99999,
        ssc_gamma3: float = 0.999,
        ssc_eps_floor: float = 1e-16,
        torch_compile: bool = False,
        **kwargs,
    ):
        # Validate original arguments
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        if not (0.0 <= cautious_factor <= 1.0):
            raise ValueError(f"cautious_factor ({cautious_factor}) must lie in [0.0, 1.0].")

        # Validate new SSC params
        self.validate_non_negative(ssc_scale, 'ssc_scale')
        if not (0.0 <= ssc_gamma1 <= 1.0):
            raise ValueError(f"ssc_gamma1 ({ssc_gamma1}) must be between 0.0 and 1.0.")
        if not (0.0 <= ssc_gamma2 <= 1.0):
            raise ValueError(f"ssc_gamma2 ({ssc_gamma2}) must be between 0.0 and 1.0.")
        if not (0.0 <= ssc_gamma3 <= 1.0):
            raise ValueError(f"ssc_gamma3 ({ssc_gamma3}) must be between 0.0 and 1.0.")
        self.validate_non_negative(ssc_eps_floor, 'ssc_eps_floor')

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'kahan_sum': kahan_sum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'use_rms': use_rms,
            'use_atan2': use_atan2,
            'atan2_a': atan2_a,
            'atan2_b': atan2_b,
            'cautious_factor': cautious_factor,
            'use_adopt': use_adopt,
            'use_stable_spam_clipping': use_stable_spam_clipping,
            'ssc_scale': ssc_scale,
            'ssc_gamma1': ssc_gamma1,
            'ssc_gamma2': ssc_gamma2,
            'ssc_gamma3': ssc_gamma3,
            'ssc_eps_floor': ssc_eps_floor,
            'torch_compile': torch_compile,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        # Check if any group uses SSC, not just the default.
        # However, defaults.get is probably fine if groups don't override this flag.
        is_ssc_used = any(group.get('use_stable_spam_clipping', False) for group in self.param_groups)
        return 'CStableAdamW_with_SSC' if is_ssc_used else 'CStableAdamW'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        """Reset state for all parameter groups."""
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['kahan_comp'] = (
                    torch.zeros_like(p) if group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16} else None
                )
                if group.get('use_adopt', False):
                    state['prev_grad'] = torch.zeros_like(p)
                
                state['steps'] = 0

                if group.get('use_stable_spam_clipping', False):
                    # Initialize SSC states as 0-dim tensors on the parameter's device and dtype
                    param_dtype = p.dtype if p.is_floating_point() else torch.float32 # Fallback dtype
                    state['ssc_m_norm_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)
                    state['ssc_v_norm_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)
                    state['ssc_m_max_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)

    def get_stable_adamw_rms(self, grad: Tensor, exp_avg_sq: Tensor, eps: float) -> Tensor:
        return exp_avg_sq.sqrt().add(eps)

    @torch.no_grad()
    def _stable_spam_clipping(
        self,
        state: dict,
        grad: Tensor,
        group_step: int,
        ssc_scale: float,
        ssc_eps_clip: float, # This is a Python float, will be converted to tensor if needed
        ssc_gamma1: float,
        ssc_gamma2: float,
        ssc_gamma3: float,
        torch_compile: bool = False,
    ) -> Tensor:
        if grad.numel() == 0:
            return grad

        if torch_compile:
            return _get_compiled_stable_spam_clipping()(state,
                                grad,
                                step=group_step,
                                scale=ssc_scale,
                                eps=ssc_eps_clip,
                                gamma1=ssc_gamma1,
                                gamma2=ssc_gamma2,
                                gamma3=ssc_gamma3)
        else:
            return _stable_spam_clipping_impl(state, 
                                grad, 
                                step=group_step,
                                scale=ssc_scale,
                                eps=ssc_eps_clip,
                                gamma1=ssc_gamma1,
                                gamma2=ssc_gamma2,
                                gamma3=ssc_gamma3)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['step'] = group.get('step', 0) + 1

            beta1, beta2 = group['betas']
            # These are raw betas (e.g., 0.9, 0.999) from pytorch_optimizer's BaseOptimizer.debias_beta default
            # So beta1_comp becomes 1.0 - beta1, and beta2_for_ema becomes beta2.
            beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step']) 
            beta2_for_ema: float = self.debias_beta(beta2, group['step']) 

            adam_main_eps: float = group['eps']
            eps_p2: float = adam_main_eps ** 2

            use_rms = group['use_rms']
            use_atan2 = group['use_atan2']
            cautious_factor = group['cautious_factor']
            a = group['atan2_a']
            b = group['atan2_b']
            use_adopt = group['use_adopt']

            use_ssc = group['use_stable_spam_clipping']
            if use_ssc:
                ssc_s = group['ssc_scale']
                ssc_g1 = group['ssc_gamma1']
                ssc_g2 = group['ssc_gamma2']
                ssc_g3 = group['ssc_gamma3']
                ssc_ef = group['ssc_eps_floor']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                grad_for_processing = grad.clone() if use_ssc else grad # Clone if SSC will modify

                state = self.state[p]
                param_dtype = p.dtype if p.is_floating_point() else torch.float32 # Fallback dtype

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['kahan_comp'] = (
                        torch.zeros_like(p)
                        if (group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16})
                        else None
                    )
                    if use_adopt:
                        state['prev_grad'] = torch.zeros_like(p)
                    state['steps'] = 0
                    
                    if use_ssc:
                        state['ssc_m_norm_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)
                        state['ssc_v_norm_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)
                        state['ssc_m_max_t'] = torch.tensor(0.0, device=p.device, dtype=param_dtype)
                
                state['steps'] += 1
                local_param_step = state['steps']

                if use_ssc:
                    eps_clip_for_ssc = ssc_ef
                    if grad_for_processing.numel() > 0:
                        rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
                        val_to_bound = 1e-2 * rms_grad
                        eps_clip_for_ssc = torch.clamp(val_to_bound, min=ssc_ef, max=adam_main_eps)

                    grad_for_processing = self._stable_spam_clipping(
                        state=state,
                        grad=grad_for_processing,
                        group_step=group['step'],
                        ssc_scale=ssc_s,
                        ssc_eps_clip=eps_clip_for_ssc,
                        ssc_gamma1=ssc_g1,
                        ssc_gamma2=ssc_g2,
                        ssc_gamma3=ssc_g3
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if use_adopt:
                    prev_grad = state['prev_grad']
                    exp_avg.lerp_(grad_for_processing, weight=beta1_comp) # beta1_comp is 1-beta1
                    exp_avg_sq.mul_(beta2_for_ema).addcmul_(prev_grad, prev_grad, value=1.0 - beta2_for_ema)
                    prev_grad.copy_(grad_for_processing)
                else:
                    exp_avg.lerp_(grad_for_processing, weight=beta1_comp) # beta1_comp is 1-beta1
                    exp_avg_sq.mul_(beta2_for_ema).addcmul_(grad_for_processing, grad_for_processing, value=1.0 - beta2_for_ema)

                apply_weight_decay(
                    p=p,
                    grad=grad_for_processing,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                    torch_compile=group.get('torch_compile', False),
                )

                if use_rms:
                    rms_factor = self.get_stable_adamw_rms(grad_for_processing, exp_avg_sq, eps=eps_p2)
                    lr_val = group['lr'] / rms_factor
                else:
                    lr_val = group['lr']

                if not use_atan2:
                    denom = exp_avg_sq.sqrt().add_(adam_main_eps)
                    update = exp_avg.div(denom)
                    update.mul_(-lr_val)
                else:
                    bc1 = 1.0 - (beta1 ** local_param_step)
                    bc2 = 1.0 - (beta2 ** local_param_step)
                    div_bc1 = bc1 if bc1 > 1e-12 else 1e-12
                    div_bc2 = bc2 if bc2 > 1e-12 else 1e-12

                    bias_corrected_avg = exp_avg.div(div_bc1)
                    bias_corrected_avg_sq_root = exp_avg_sq.div(div_bc2).sqrt_()
                    bias_corrected_avg_sq_scaled = bias_corrected_avg_sq_root.mul_(b)

                    update = torch.atan2(bias_corrected_avg, bias_corrected_avg_sq_scaled).mul_(a)

                    if cautious_factor < 1.0:
                        align_mask = (update * grad_for_processing) > 0
                        scale_cautious = torch.where(align_mask, torch.ones_like(grad_for_processing), grad_for_processing.new_full((), cautious_factor, device=p.device, dtype=p.dtype))
                        scale_mean_cautious = scale_cautious.mean().clamp(min=1e-12)
                        update.mul_(scale_cautious.div(scale_mean_cautious))
                    
                    update.mul_(-lr_val)

                if group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16}:
                    kahan_comp = state['kahan_comp']
                    kahan_comp.add_(update)
                    temp_p = p.detach().clone()
                    p.add_(kahan_comp)
                    kahan_comp.add_(temp_p.sub_(p))
                else:
                    p.add_(update)
        return loss