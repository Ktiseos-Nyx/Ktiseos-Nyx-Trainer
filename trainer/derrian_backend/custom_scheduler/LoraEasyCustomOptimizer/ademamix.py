# Authored by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/ademamix.py

import math
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_, UPDATE_STRATEGY, NORM_TYPE, agc, _paper_orthograd, adaptive_eps, _get_compiled_stable_spam_clipping, _stable_spam_clipping_impl
import logging

logger = logging.getLogger(__name__)


# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
@torch.no_grad()
def zero_power_via_newton_schulz_6(grad: torch.Tensor) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param grad: torch.Tensor. matrix.
    """
    # Inline reshaping step within the method itself.
    G_shape = grad.shape
    grad = grad.view(grad.size(0), -1)

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    X = grad.float()
    if grad.size(0) > grad.size(1):
        X = X.T

    X = X.div(X.norm().add(1e-16))# ensure top singular value <= 1
    #for _ in range(num_steps):
    for a,b,c in abc_list:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T

    # Gradient scaling adaptation from: https://github.com/leloykun/adaptive-muon
    X = torch.einsum('ij,ij->', grad.type_as(X), X).clamp(-1.0, 1.0) * X

    return X.view(G_shape)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def zero_power_via_newton_schulz_6_compile(grad: torch.Tensor) -> torch.Tensor:
    return zero_power_via_newton_schulz_6(grad)

@torch.no_grad()

def bias_rms(grad: torch.Tensor) -> torch.Tensor:
    rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
    grad = grad.div(rms_value.add_(1e-16))
    return grad

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def bias_rms_compile(grad: torch.Tensor) -> torch.Tensor:
    return bias_rms(grad)


class AdEMAMix(BaseOptimizer):
    r"""Better, Faster, Older.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip: float. threshold of root-mean-square of gradient update.
    :param alpha: float. usually between 4 and 10 would work well.
    :param t_alpha_beta3: Optional[float]. total number of iterations is preferred when needed.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param centralization: float. center model grad 
    cautious (bool) (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
        and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        clip: float = 0.0,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        eps: float = 1e-8,
        centralization: float = 0.0,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        adopt: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(centralization, 'centralization')

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'clip': clip,
            'fixed_decay': fixed_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps,
            'centralization': centralization,
            'cautious': cautious,
            'update_strategy': update_strategy,
            'adopt': adopt
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdEMAMix'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                state = self.state[p]

                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    state['exp_avg'] = torch.zeros_like(p)
                else: 
                    state['exp_avg'] = None
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
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

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

            beta1, beta2, beta3 = group['betas']

            step = group['step']

            bias_correction1: float = self.debias(beta1, step)
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

            eps = group['eps']
            clip = group['clip']
            centralization = group['centralization']
            adopt = group['adopt']

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], step, group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], step, beta1, beta3, eps)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                p_fp32 = p
                grad = p.grad

                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                
                state = self.state[p]

                if len(state) == 0:
                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        state['exp_avg'] = torch.zeros_like(p)
                    else: 
                        state['exp_avg'] = None
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_slow'] = torch.zeros_like(p)

                # center the gradient vector
                if centralization > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_(((self.get_rms(grad) + eps) / clip).clamp_(min=1.0))

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq, exp_avg_slow = exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)

                if adopt and step == 0:
                    exp_avg_sq.add_(grad)
                else:
                    og_grad = grad
                    if not adopt:
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        de_nom = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)
                    else:
                        de_nom = (exp_avg_sq.sqrt()).add_(eps)
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        adopt_clip: float = (step-1)**0.25
                        scaled_adopt_clip = adopt_clip * de_nom
                        grad = grad.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                    if beta1 > 0.0:
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    else:
                        exp_avg = grad

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)

                    update = (exp_avg.div(bias_correction1) + alpha_t * exp_avg_slow)

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        elif group['update_strategy'] == 'grams':
                            update.copy_(torch.sign(grad) * update.abs())

                    update = update / de_nom

                    apply_weight_decay(
                        p=p_fp32,
                        grad=update,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                        torch_compile=group.get('torch_compile', False),
                    )

                    p_fp32.add_(-group['lr'] * update)
                    
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

        return loss

class SimplifiedAdEMAMix(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param alpha: float. coefficient for mixing the current gradient and EMA.
    :param beta1_warmup: Optional[int]. number of warmup steps used to increase beta1.
    :param min_beta1: float. minimum value of beta1 to start from.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param bias_correction1: bool. whether to use bias_correction in numerator
    :param bias_correction2: bool. whether to use bias_correction in denominator
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float|torch.Tensor = 1e-4,
        betas: Betas = (0.99, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 1.0,
        beta1_warmup: Optional[int] = None,
        min_beta1: float = 0.9,
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        use_orthograd: bool = False,
        adaptive_clip: Optional[float] = None,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        bias_correction1: bool = False,
        bias_correction2: bool = True,
        use_stable_spam_clipping:bool = False,
        use_adopt: bool = False,
        torch_compile: bool = False,          # Compile helper functions (apply_weight_decay, stable_spam_clipping)
        cautious_weight_decay: bool = False,
        compile_step: bool = False,            # Compile the core _core_step_fp32 function via torch.compile
        foreach: bool = False,
        kahan_sum: bool = False,
        sync_chunk_size: int = 256,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(min_beta1, 'min_beta1')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )
        
        if isinstance(state_storage_dtype, str):
            normalized_str_dtype = state_storage_dtype.strip().lower()
            if normalized_str_dtype == "float32":
                final_dtype = torch.float32
            elif normalized_str_dtype == "float16":
                final_dtype = torch.float16
            elif normalized_str_dtype == "bfloat16":
                final_dtype = torch.bfloat16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        # Cache for placeholder tensors (device -> empty(0) tensor)
        self._empty_tensor_cache: dict = {}
        self._compiled_step = None            # lazy-initialized compiled callable

        # Single shared int32 scratch buffer for stochastic rounding.
        # Grows to the largest parameter size encountered; reused across all params each step.
        self._srng_buf: torch.Tensor | None = None

        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'beta1_warmup': beta1_warmup,
            'min_beta1': min_beta1,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'use_orthograd': use_orthograd,
            'adaptive_clip': adaptive_clip,
            'adaptive_clip_eps': adaptive_clip_eps,
            'adaptive_clip_type': adaptive_clip_type,
            'update_strategy': update_strategy,
            'bias_correction1': bias_correction1,
            'bias_correction2': bias_correction2,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'use_adopt':use_adopt,
            'torch_compile': torch_compile,
            'cautious_weight_decay': cautious_weight_decay,
            'compile_step': compile_step,
            'foreach': foreach,
            'kahan_sum': kahan_sum,
            'sync_chunk_size': sync_chunk_size,
            'state_storage_dtype': final_dtype,
            'state_storage_device': state_storage_device,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SimplifiedAdEMAMix'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:

        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))

        return beta_end

    def _get_srng_buf(self, like_tensor: torch.Tensor) -> torch.Tensor:
        r"""Get a reusable int32 scratch buffer for stochastic rounding noise.

        Returns a view of a single shared buffer sized to the largest parameter
        encountered. The buffer is reused across all parameters within a step,
        eliminating per-parameter int32 allocations. Content is NOT preserved
        across calls — callers must refill before each use.
        """
        n = like_tensor.numel()
        if self._srng_buf is None or self._srng_buf.device != like_tensor.device or self._srng_buf.numel() < n:
            self._srng_buf = torch.empty(n, dtype=torch.int32, device=like_tensor.device)
        return self._srng_buf[:n].view(like_tensor.shape)

    def _get_empty_tensor(self, device: torch.device) -> torch.Tensor:
        r"""Get or create cached empty tensor for optional state placeholder."""
        if device not in self._empty_tensor_cache:
            self._empty_tensor_cache[device] = torch.empty(0, device=device)
        return self._empty_tensor_cache[device]

    @staticmethod
    @torch.no_grad()
    def _core_step_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        p_fp32: torch.Tensor,
        num_sum: torch.Tensor,
        den_sum: torch.Tensor,
        beta1_t: float,
        beta2_t: float,
        alpha_t: float,
        lr_t: float,
        wd_t: float,
        curr_eps_t: float,
        adopt_clip_t: float,
        use_adopt: bool,
        is_step_one_and_adopt: bool,
        bias_correction1: bool,
        bias_correction2: bool,
        weight_decouple: bool,
        fixed_decay: bool,
        cautious_weight_decay: bool,
        use_cautious: bool,
        use_grams: bool,
    ) -> None:
        r"""Core per-parameter step INCLUDING weight decay, LR scale, update strategy, and param update.

        All inputs are FP32 tensors on the compute device.
        Modifies all tensor arguments in-place.
        Branch booleans are compile-time constants resolved during tracing.
        """
        if is_step_one_and_adopt:
            # Step 1 adopt: just initialize exp_avg_sq with grad^2
            exp_avg_sq.addcmul_(grad, grad)
            return

        # 0. Apply L2 weight decay to grad BEFORE EMA updates (correct L2 behavior).
        #    This ensures the weight decay term is incorporated into momentum and
        #    second moment estimates, matching standard Adam L2 regularization.
        #    Cautious and decoupled WD modify p_fp32 directly and are handled in step 9.
        if not cautious_weight_decay and not weight_decouple and wd_t > 0.0:
            grad.add_(p_fp32, alpha=wd_t)

        # 1. Update exp_avg (momentum)
        exp_avg.mul_(beta1_t).add_(grad, alpha=1.0 - beta1_t)

        # 2. Update bias correction accumulators
        num_sum.mul_(beta1_t).add_(1.0)
        den_sum.mul_(beta2_t).add_(1.0 - beta2_t)

        # 3. Denominator computation
        if use_adopt:
            de_nom = exp_avg_sq.sqrt().add_(den_sum.sqrt() * curr_eps_t)
            exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=1.0 - beta2_t)
        else:
            exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=1.0 - beta2_t)
            de_nom = exp_avg_sq.sqrt().add_(den_sum.sqrt() * curr_eps_t)

        # 4. Update = alpha * grad + exp_avg
        update = grad.mul(alpha_t).add_(exp_avg)

        # 5. Update strategies
        if use_cautious:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)
        if use_grams:
            update.copy_(torch.sign(grad) * update.abs())

        # 6. Divide by denominator
        update.div_(de_nom)

        # 7. Bias correction
        if bias_correction1:
            update.div_(num_sum)
        if bias_correction2:
            update.mul_(den_sum.sqrt())

        # 8. ADOPT clamping
        if use_adopt:
            update.clamp_(-adopt_clip_t, adopt_clip_t)

        # 9. Weight decay (cautious and decoupled only; L2 already applied in step 0)
        if cautious_weight_decay:
            # Apply weight decay only where gradient and parameter agree in sign
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - lr_t * wd_t * cwd_mask)
        elif weight_decouple:
            wd_factor = 1.0 if fixed_decay else lr_t
            p_fp32.mul_(1.0 - wd_t * wd_factor)

        # 10. Parameter update
        p_fp32.add_(update, alpha=-lr_t)

    def _compile_core_fns(self) -> None:
        r"""Lazily compile the core step function with torch.compile."""
        if self.defaults.get('compile_step', False):
            try:
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 64
                )
                with torch._dynamo.utils.disable_cache_limit():
                    self._compiled_step = torch.compile(
                        self._core_step_fp32, fullgraph=True, dynamic=False
                    )
                logger.info("SimplifiedAdEMAMix core function compiled with torch.compile(fullgraph=True, dynamic=False).")
            except Exception as e:
                logger.warning(f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. Falling back to uncompiled step.")
                self._compiled_step = self._core_step_fp32
        else:
            self._compiled_step = self._core_step_fp32

    @torch.no_grad()
    def _foreach_step(
        self,
        group,
        active_params: list,
        beta1: float,
        beta2: float,
        compute_device: torch.device,
    ) -> None:
        r"""Foreach step for 1D parameters (ndim == 1, numel >= 16).

        Batches operations using ``torch._foreach_*`` for better GPU utilization.
        Handles the full SimplifiedAdEMAMix pipeline: momentum, denominator,
        update strategy, bias correction, weight decay, parameter update with
        optional Kahan summation and stochastic rounding.
        """
        use_adopt = group['use_adopt']
        update_strategy = group['update_strategy']
        lr = group['lr']
        wd = group['weight_decay']
        wd_decouple = group['weight_decouple']
        fixed_decay = group['fixed_decay']
        cwd = group.get('cautious_weight_decay', False)
        bc1 = group['bias_correction1']
        bc2 = group['bias_correction2']
        alpha = group['alpha']
        use_kahan = group.get('kahan_sum', False)
        step = group['step']
        adopt_clip = (step - 1) ** 0.25

        n = len(active_params)

        # ========= Collect phase: build FP32 tensor lists on compute device =========
        p_fp32_list = [None] * n
        grad_list = [None] * n
        exp_avg_list = [None] * n
        exp_avg_sq_list = [None] * n
        num_sum_list = [None] * n
        den_sum_list = [None] * n
        eps_list = [None] * n
        kahan_comp_list = [None] * n if use_kahan else None
        kahan_sim_list = [None] * n if use_kahan else None
        state_list = [None] * n
        param_kahan_flags = [False] * n
        param_list = [None] * n

        for idx, p in enumerate(active_params):
            state = self.state[p]
            state_list[idx] = state
            param_list[idx] = p

            exp_avg_list[idx] = state["exp_avg"].to(
                compute_device, non_blocking=True, dtype=torch.float32
            )
            exp_avg_sq_list[idx] = state["exp_avg_sq"].to(
                compute_device, non_blocking=True, dtype=torch.float32
            )
            grad_list[idx] = p.grad.data.to(
                compute_device, dtype=torch.float32, non_blocking=True
            )
            p_fp32_list[idx] = p.to(
                compute_device, dtype=torch.float32, non_blocking=True
            )

            # Scalar state as tensors
            num_sum_list[idx] = torch.tensor(
                float(state.get('num_sum', 0.0)), device=compute_device, dtype=torch.float32
            )
            den_sum_list[idx] = torch.tensor(
                float(state.get('den_sum', 0.0)), device=compute_device, dtype=torch.float32
            )

            # Compute adaptive eps per-parameter (BUG-2 fix: foreach now uses adaptive eps)
            eps_list[idx] = adaptive_eps(grad_list[idx], group)

            param_kahan = use_kahan and p.dtype in {torch.float16, torch.bfloat16}
            param_kahan_flags[idx] = param_kahan
            if param_kahan:
                kahan_comp_list[idx] = state['kahan_comp'].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )
                kahan_sim_list[idx] = torch.empty_like(p_fp32_list[idx])

        # ========= Kahan pre-compensation =========
        if use_kahan:
            for idx in range(n):
                if param_kahan_flags[idx]:
                    p_fp32_list[idx].add_(kahan_comp_list[idx])

        # ========= L2 weight decay BEFORE EMA updates (BUG-1 fix) =========
        # Ensures the weight decay term is incorporated into momentum and
        # second moment estimates, matching standard Adam L2 regularization.
        if wd != 0 and not cwd and not wd_decouple:
            torch._foreach_add_(grad_list, p_fp32_list, alpha=wd)

        # ========= BATCH: momentum update =========
        # exp_avg = beta1 * exp_avg + (1-beta1) * grad
        torch._foreach_mul_(exp_avg_list, beta1)
        torch._foreach_add_(exp_avg_list, grad_list, alpha=1.0 - beta1)

        # ========= BATCH: bias correction accumulators =========
        for idx in range(n):
            num_sum_list[idx].mul_(beta1).add_(1.0)
            den_sum_list[idx].mul_(beta2).add_(1.0 - beta2)

        # ========= BATCH: denominator =========
        # Save current exp_avg_sq for adopt path before update
        if use_adopt:
            old_exp_avg_sq_list = [sq.clone() for sq in exp_avg_sq_list]

        torch._foreach_mul_(exp_avg_sq_list, beta2)
        torch._foreach_addcmul_(exp_avg_sq_list, grad_list, grad_list, value=1.0 - beta2)

        if use_adopt:
            de_nom_list = []
            for idx in range(n):
                dn = old_exp_avg_sq_list[idx].sqrt_().add_(
                    den_sum_list[idx].sqrt() * eps_list[idx]
                )
                de_nom_list.append(dn)
        else:
            de_nom_list = []
            for idx in range(n):
                dn = exp_avg_sq_list[idx].sqrt().add_(
                    den_sum_list[idx].sqrt() * eps_list[idx]
                )
                de_nom_list.append(dn)

        # ========= PER-TENSOR: update = alpha * grad + exp_avg =========
        update_list = []
        for idx in range(n):
            upd = grad_list[idx].mul(alpha).add_(exp_avg_list[idx])
            update_list.append(upd)

        # ========= BATCH: update strategies =========
        if update_strategy in ('cautious', 'both'):
            mask_list = []
            for idx in range(n):
                mask = (update_list[idx] * grad_list[idx] > 0).to(grad_list[idx].dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                mask_list.append(mask)
            torch._foreach_mul_(update_list, mask_list)
        if update_strategy in ('grams', 'both'):
            # In-place: sign(grad) * |update|
            sign_list = torch._foreach_sign(grad_list)
            torch._foreach_abs_(update_list)
            torch._foreach_mul_(update_list, sign_list)

        # ========= BATCH: divide by denominator =========
        torch._foreach_div_(update_list, de_nom_list)

        # ========= PER-TENSOR: bias correction =========
        if bc1:
            for idx in range(n):
                update_list[idx].div_(num_sum_list[idx])
        if bc2:
            for idx in range(n):
                update_list[idx].mul_(den_sum_list[idx].sqrt())

        # ========= BATCH: ADOPT clamping =========
        if use_adopt:
            torch._foreach_clamp_(update_list, -adopt_clip, adopt_clip)

        # ========= BATCH: weight decay (cautious and decoupled only; L2 already applied above) =========
        if wd != 0:
            if cwd:
                for idx in range(n):
                    cwd_mask = (grad_list[idx] * p_fp32_list[idx] >= 0).to(p_fp32_list[idx].dtype)
                    p_fp32_list[idx].mul_(1.0 - lr * wd * cwd_mask)
            elif wd_decouple:
                wd_factor = 1.0 if fixed_decay else lr
                torch._foreach_mul_(p_fp32_list, 1.0 - wd * wd_factor)

        # ========= BATCH: LR scale and parameter update =========
        torch._foreach_mul_(update_list, -lr)
        torch._foreach_add_(p_fp32_list, update_list)

        # ========= Write-back phase =========
        for idx in range(n):
            p = param_list[idx]
            state = state_list[idx]
            p_fp32 = p_fp32_list[idx]
            device = p.device
            srng = self._get_srng_buf(exp_avg_list[idx])

            # Kahan write-back
            if param_kahan_flags[idx]:
                kahan_sim = kahan_sim_list[idx]
                kahan_comp = kahan_comp_list[idx]

                kahan_sim.copy_(p_fp32)
                if p.dtype == torch.bfloat16:
                    sim_int = kahan_sim.view(dtype=torch.int32)
                    if srng is not None:
                        srng.random_(0, 1 << 16)
                        sim_int.add_(srng)
                    else:
                        sim_int.add_(torch.randint_like(sim_int, 0, 1 << 16))
                    sim_int.bitwise_and_(-65536)
                else:
                    kahan_sim.copy_(p_fp32.to(p.dtype).to(torch.float32))

                if device.type == "cpu":
                    p.data.copy_(kahan_sim)
                else:
                    p.data.copy_(kahan_sim, non_blocking=True)

                kahan_sim.sub_(p_fp32)

                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state['kahan_comp'], kahan_sim, scratch=srng)
                else:
                    state['kahan_comp'].copy_(kahan_sim, non_blocking=True)
            else:
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32, scratch=srng)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32, scratch=srng)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

            # State write-back
            exp_avg = exp_avg_list[idx]
            exp_avg_sq = exp_avg_sq_list[idx]
            if self.state_storage_dtype == torch.bfloat16:
                copy_stochastic_(state["exp_avg"], exp_avg, scratch=srng)
                copy_stochastic_(state["exp_avg_sq"], exp_avg_sq, scratch=srng)
            else:
                state["exp_avg"].copy_(exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

            # Store scalar state back as Python floats (CONSISTENCY-1 fix)
            state['num_sum'] = num_sum_list[idx].item()
            state['den_sum'] = den_sum_list[idx].item()

            # Sync chunking (CONSISTENCY-2 fix: use self.sync_chunk_size)
            if (idx + 1) % self.sync_chunk_size == 0:
                torch.cuda.synchronize()

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

            step = group['step']
            adopt_clip: float = (step - 1) ** 0.25

            beta1, beta2 = group['betas']

            use_orthograd = group['use_orthograd']
            adaptive_clip = group['adaptive_clip']
            adaptive_clip_eps = group['adaptive_clip_eps']
            adaptive_clip_type = group['adaptive_clip_type']
            update_strategy = group['update_strategy']
            use_adopt = group['use_adopt']
            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get('is_ortho_group', False)
            compile_step = group.get('compile_step', False)
            cwd = group.get('cautious_weight_decay', False)
            use_kahan = group.get('kahan_sum', False)

            if group['beta1_warmup']:
                beta1 = self.linear_hl_warmup_scheduler(
                    step, beta_end=beta1, beta_start=group['min_beta1'], warmup=group['beta1_warmup']
                )

            # Store compiled beta1 and adopt_clip in group for scalar tensor caching
            group['_compiled_beta1'] = beta1
            group['_compiled_adopt_clip'] = adopt_clip

            # ========= Foreach bucketing: collect 1D params for batched processing =========
            use_foreach = group.get('foreach', False)
            foreach_params = []
            if use_foreach:
                # Skip foreach on step 1 + adopt (simple init path, handled per-param)
                skip_foreach = use_adopt and step == 1
                if not skip_foreach:
                    # Params needing per-tensor preprocessing cannot be batched in foreach
                    needs_preprocessing = (
                        (apply_ortho_to_group and use_orthograd) or
                        (adaptive_clip is not None and adaptive_clip > 0) or
                        use_stable_spam_clipping
                    )
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if p.grad.ndim == 1 and p.numel() >= 16:
                            if needs_preprocessing:
                                continue  # Skip foreach — falls through to per-param path
                            # Lazy-init state if needed
                            state = self.state[p]
                            if len(state) == 0:
                                self._init_param_state(p, state, group, use_kahan)
                            foreach_params.append(p)

                if foreach_params:
                    first_device = foreach_params[0].device
                    foreach_compute_device = (
                        torch.cuda.current_device() if first_device.type == "cpu" else first_device
                    )
                    self._foreach_step(
                        group, foreach_params, beta1, beta2, foreach_compute_device
                    )
                    torch.cuda.synchronize()

            # ========= Lazily compile core function on first step =========
            if self._compiled_step is None:
                self._compile_core_fns()

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # Skip 1D params already handled by foreach
                if use_foreach and p.grad.ndim == 1 and p.numel() >= 16:
                    if not (use_adopt and step == 1):
                        continue

                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                device = p.device

                if len(state) == 0:
                    self._init_param_state(p, state, group, use_kahan)

                # Determine target GPU device for computation
                if device.type == "cpu":
                    compute_device = torch.cuda.current_device()
                else:
                    compute_device = device

                # Transfer state to compute device
                exp_avg = state["exp_avg"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )
                exp_avg_sq = state["exp_avg_sq"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)

                param_kahan = use_kahan and p.dtype in {torch.float16, torch.bfloat16}
                srng = self._get_srng_buf(exp_avg)

                if param_kahan:
                    kahan_comp = state['kahan_comp'].to(
                        compute_device, non_blocking=True, dtype=torch.float32
                    )
                    p_fp32.add_(kahan_comp)

                # ========= Preprocessing (outside compiled step) =========
                if apply_ortho_to_group and use_orthograd:
                    _paper_orthograd(param=p_fp32, grad=grad)

                if adaptive_clip is not None and adaptive_clip > 0:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip,
                               agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                if use_stable_spam_clipping:
                    if group['torch_compile']:
                        grad = _get_compiled_stable_spam_clipping()(state, grad, step=step)
                    else:
                        grad = _stable_spam_clipping_impl(state, grad, step=step)

                curr_eps = adaptive_eps(grad, group)
                group['_compiled_eps'] = curr_eps

                # ========= Core computation =========
                if compile_step:
                    # Compiled path — pre-allocate and reuse scalar tensors (OPT-1 fix)
                    if '_num_sum_t' not in state or state['_num_sum_t'].device != compute_device:
                        state['_num_sum_t'] = torch.tensor(
                            float(state.get('num_sum', 0.0)), device=compute_device, dtype=torch.float32
                        )
                        state['_den_sum_t'] = torch.tensor(
                            float(state.get('den_sum', 0.0)), device=compute_device, dtype=torch.float32
                        )
                    else:
                        state['_num_sum_t'].fill_(float(state.get('num_sum', 0.0)))
                        state['_den_sum_t'].fill_(float(state.get('den_sum', 0.0)))
                    num_sum_t = state['_num_sum_t']
                    den_sum_t = state['_den_sum_t']

                    is_step_one_and_adopt = use_adopt and step == 1

                    self._compiled_step(
                        grad, exp_avg, exp_avg_sq, p_fp32,
                        num_sum_t, den_sum_t,
                        beta1, beta2, group['alpha'],
                        group['lr'], group['weight_decay'], curr_eps,
                        adopt_clip,
                        use_adopt, is_step_one_and_adopt,
                        group['bias_correction1'], group['bias_correction2'],
                        group['weight_decouple'], group['fixed_decay'],
                        cwd,
                        update_strategy in {'cautious', 'both'},
                        update_strategy in {'grams', 'both'},
                    )

                    # Store scalar state back as Python float (CONSISTENCY-1 fix)
                    state['num_sum'] = num_sum_t.item()
                    state['den_sum'] = den_sum_t.item()
                else:
                    # Uncompiled path
                    if use_adopt and step == 1:
                        exp_avg_sq.addcmul_(grad, grad)
                    else:
                        # BUG-1 fix: Apply L2 weight decay to grad BEFORE EMA updates
                        if not group['weight_decouple'] and not cwd and group['weight_decay'] > 0:
                            grad.add_(p_fp32, alpha=group['weight_decay'])

                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                        state['num_sum'] = beta1 * state['num_sum'] + 1.0
                        state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)

                        if use_adopt:
                            de_nom = exp_avg_sq.sqrt().add_(math.sqrt(state['den_sum']) * curr_eps)
                            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        else:
                            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                            de_nom = exp_avg_sq.sqrt().add_(math.sqrt(state['den_sum']) * curr_eps)

                        update = (group['alpha'] * grad + exp_avg)

                        if update_strategy in {'cautious', 'grams', 'both'}:
                            if update_strategy in {'cautious', 'both'}:
                                mask = (update * grad > 0).to(grad.dtype)
                                mask.div_(mask.mean().clamp_(min=1e-3))
                                update = update * mask
                            if update_strategy in {'grams', 'both'}:
                                update.copy_(torch.sign(grad) * update.abs())

                        update.div_(de_nom)

                        if group['bias_correction1']:
                            update.div_(state['num_sum'])
                        if group['bias_correction2']:
                            update.mul_(math.sqrt(state['den_sum']))

                        if use_adopt:
                            update.clamp_(-adopt_clip, adopt_clip)

                        # Apply cautious/decoupled weight decay (L2 already applied above)
                        if cwd or group['weight_decouple']:
                            apply_weight_decay(
                                p=p_fp32,
                                grad=grad,
                                lr=group['lr'],
                                weight_decay=group['weight_decay'],
                                weight_decouple=group['weight_decouple'],
                                fixed_decay=group['fixed_decay'],
                                cautious_weight_decay=cwd,
                                torch_compile=group.get('torch_compile', False),
                            )

                        p_fp32.add_(update, alpha=-group['lr'])

                # ========= Write-back =========
                if param_kahan:
                    kahan_sim = torch.empty_like(p_fp32)
                    kahan_sim.copy_(p_fp32)
                    if p.dtype == torch.bfloat16:
                        sim_int = kahan_sim.view(dtype=torch.int32)
                        if srng is not None:
                            srng.random_(0, 1 << 16)
                            sim_int.add_(srng)
                        else:
                            sim_int.add_(torch.randint_like(sim_int, 0, 1 << 16))
                        sim_int.bitwise_and_(-65536)
                    else:
                        kahan_sim.copy_(p_fp32.to(p.dtype).to(torch.float32))

                    if device.type == "cpu":
                        p.data.copy_(kahan_sim)
                    else:
                        p.data.copy_(kahan_sim, non_blocking=True)

                    kahan_sim.sub_(p_fp32)

                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state['kahan_comp'], kahan_sim, scratch=srng)
                    else:
                        state['kahan_comp'].copy_(kahan_sim, non_blocking=True)
                else:
                    if device.type == "cpu":
                        if p.dtype == torch.bfloat16:
                            copy_stochastic_(p.data, p_fp32, scratch=srng)
                        else:
                            p.data.copy_(p_fp32)
                    else:
                        if p.dtype == torch.bfloat16:
                            copy_stochastic_(p, p_fp32, scratch=srng)
                        else:
                            p.data.copy_(p_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg, scratch=srng)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq, scratch=srng)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                # Sync chunking
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization
            torch.cuda.synchronize()

        return loss

    def _init_param_state(self, p, state, group, use_kahan: bool) -> None:
        r"""Initialize optimizer state for a parameter (shared by foreach and per-param paths)."""
        state["exp_avg"] = torch.zeros_like(
            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
        )
        state["exp_avg_sq"] = torch.zeros_like(
            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
        )

        if self.state_storage_device == "cpu":
            state["exp_avg"] = state["exp_avg"].pin_memory()
            state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()

        state['num_sum'] = 0.0
        state['den_sum'] = 0.0

        if use_kahan and p.dtype in {torch.float16, torch.bfloat16}:
            state["kahan_comp"] = torch.zeros(
                p.shape, dtype=torch.float32, device=self.state_storage_device
            )
            if self.state_storage_device == "cpu":
                state["kahan_comp"] = state["kahan_comp"].pin_memory()
    
class SimplifiedAdEMAMixExM(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param alpha: float. coefficient for mixing the current gradient and EMA.
    :param beta1_warmup: Optional[int]. number of warmup steps used to increase beta1. Recommend setting to iteration/step count.
    :param min_beta1: float. minimum value of beta1 to start from.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float|torch.Tensor = 2e-4,
        betas: Betas = (0.95, 0.997),
        min_beta1: float = 0.95,
        beta1_warmup: Optional[int] = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        alpha: float = 1.0,
        eps: float = 1e-8,
        eps_floor: Optional[float] = 1e-12,
        use_orthograd: bool = True,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        update_strategy_scale: float = 1.0,
        use_stable_spam_clipping:bool = True,
        use_compass: bool = False,
        use_adabelief: bool = True,
        use_newton_schulz: bool = True,
        amsgrad_min_decay_rate: float = 0.98,
        amsgrad_max_decay_rate: float = 0.98,
        torch_compile: bool = True,
        sync_chunk_size: int = 256,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(min_beta1, 'min_beta1')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        if isinstance(state_storage_dtype, str):
            normalized_str_dtype = state_storage_dtype.strip().lower()
            if normalized_str_dtype == "float32":
                final_dtype = torch.float32
            elif normalized_str_dtype == "float16":
                final_dtype = torch.float16
            elif normalized_str_dtype == "bfloat16":
                final_dtype = torch.bfloat16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        if not (0.0 <= update_strategy_scale <= 1.0):
            raise ValueError(f"update_strategy_scale ({update_strategy_scale}) must lie in [0.0, 1.0].")
        
        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'beta1_warmup': beta1_warmup,
            'min_beta1': min_beta1,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'eps2': 1e-2,
            'eps_floor': eps_floor,
            'use_orthograd': use_orthograd,
            'update_strategy': update_strategy,
            'update_strategy_scale': update_strategy_scale,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'use_compass': use_compass,
            'use_adabelief': use_adabelief,
            'torch_compile': torch_compile,
            'amsgrad_max_decay_rate': amsgrad_max_decay_rate,
            'amsgrad_min_decay_rate': amsgrad_min_decay_rate,
            'use_newton_schulz':use_newton_schulz,
            'sync_chunk_size': sync_chunk_size,
            'state_storage_dtype': final_dtype,
            'state_storage_device':state_storage_device,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SimplifiedAdEMAMixExM'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:

        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))

        return beta_end

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

            step = group['step']

            adopt_clip: float = (step-1)**0.25

            beta1, beta2 = group['betas']

            use_orthograd = group['use_orthograd']
            use_compass = group['use_compass']
            use_adabelief = group['use_adabelief']
            use_newton_schulz = group['use_newton_schulz']
            update_strategy  = group['update_strategy']
            update_strategy_scale  = group['update_strategy_scale']
            amsgrad_min_decay_rate  = group['amsgrad_min_decay_rate']
            amsgrad_max_decay_rate  = group['amsgrad_max_decay_rate']
            torch_compile = group['torch_compile']

            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get('is_ortho_group', False) # Default to False if key missing

            eps_floor = group['eps_floor']

            if group['beta1_warmup']:
                beta1 = self.linear_hl_warmup_scheduler(
                    step, beta_end=beta1, beta_start=group['min_beta1'], warmup=group['beta1_warmup']
                )

            beta2 = ((beta2 ** step - beta2) / (beta2 ** step - 1.0))

            bias_correction1 = 1 - beta1 ** step
            bias_correction2_sqrt = (1 - beta2 ** step) ** (1/2)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                device = p.device
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    if self.state_storage_device == "cpu":
                        state["exp_avg"] = torch.zeros_like(
                            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
                        ).pin_memory()
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
                        ).pin_memory()
                    else:
                        state["exp_avg"] = torch.zeros_like(
                            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=self.state_storage_dtype, device=self.state_storage_device
                        )

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                exp_avg = state["exp_avg"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )
                exp_avg_sq = state["exp_avg_sq"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )

                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                if apply_ortho_to_group and use_orthograd:
                    _paper_orthograd(param=p_fp32, grad=grad)

                if use_stable_spam_clipping:
                    if torch_compile:
                        grad = _get_compiled_stable_spam_clipping()(state,
                                            grad,
                                            step=step,
                                            eps=eps_floor)
                    else:
                        grad = _stable_spam_clipping_impl(state, 
                                            grad, 
                                            step=step,
                                            eps=eps_floor)

                # Calculate RMS of grad once
                rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
                curr_eps = adaptive_eps(grad, group, rms_grad=rms_grad)

                # RMS Norm
                grad_normed = grad.div(rms_grad.clamp_min_(1))

                if use_newton_schulz:
                    if grad_normed.ndim > 0:
                        if torch_compile:
                            grad_normed = zero_power_via_newton_schulz_6_compile(grad_normed)
                        else:
                            grad_normed = zero_power_via_newton_schulz_6(grad_normed)
                    elif grad_normed.numel() > 1:
                        if torch_compile:
                            grad_normed = bias_rms_compile(grad_normed)
                        else:
                            grad_normed = bias_rms(grad_normed)
                            
                # Adaptive ema
                mask = (grad_normed * exp_avg > 0).to(grad_normed.dtype)
                mask.clamp_min_(beta1)
                mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
                exp_avg.mul_(mask)

                exp_avg.mul_(beta1).add_(grad_normed, alpha=1.0 - beta1)

                # Compass amplification + beta1 Bias correction
                if use_compass:
                    bias_corrected_axp_avg = exp_avg.div(bias_correction1)
                    c_t = grad_normed.add(bias_corrected_axp_avg, alpha=group['alpha'])
                else:
                    c_t = grad_normed

                if step == 1:
                    if use_compass:
                        # Try adding residual to c_t
                        grad_residual = c_t.add(grad_normed.add(bias_corrected_axp_avg, alpha=-1))
                    else:
                        grad_residual = grad_normed - exp_avg
                    exp_avg_sq.addcmul_(grad_residual, grad_residual)
                else:
                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(curr_eps)

                    if use_adabelief:
                        if use_compass:
                            # Try adding residual to c_t
                            grad_residual = c_t.add(grad_normed.add(bias_corrected_axp_avg, alpha=-1))
                        else:
                            grad_residual = grad_normed - exp_avg
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)
                    else:
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Decaying amsgrad
                    torch.maximum(exp_avg_sq.mul(max(min(beta2, amsgrad_max_decay_rate), amsgrad_min_decay_rate)), new_exp_avg_sq, out=exp_avg_sq)

                    if use_compass:
                        update = c_t
                    else:
                        update = (group['alpha'] * grad_normed + exp_avg)

                    update = apply_update_strategies(update, grad_normed, update_strategy, update_strategy_scale)

                    update.div_(de_nom)

                    if not use_compass:
                        update.div_(bias_correction1)

                    update.clamp_(-adopt_clip, adopt_clip)

                    apply_weight_decay(
                        p=p_fp32,
                        grad=grad_normed,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=False,
                        torch_compile=group.get('torch_compile', False),
                    )

                    p_fp32.add_(update, alpha=-group['lr'])

                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    # Original GPU path
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)
                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss

@torch.no_grad()
def apply_update_strategies(update, grad, update_strategy, scale=1.0):
    """
    Applies update strategies with scaling factors.

    Args:
        update (torch.Tensor): The current update tensor to be modified.
        grad_normed (torch.Tensor): The normalized gradient.
        update_strategy (str): One of 'cautious', 'grams', 'both'.
        scale (float): Scaling factor for the Grams strategies.

    Returns:
        torch.Tensor: The modified update tensor.
    """
    if scale > 0 and update_strategy in {'cautious', 'grams', 'both'}:
        if update_strategy in {'cautious', 'both'}:
            if scale >= 1.0:
                update_before_cautious = update

                # 1. Calculate the "fully cautious" update
                mask = (update_before_cautious * grad > 0).to(grad.dtype)
                mask_mean = mask.mean().clamp_(min=1e-3) # Avoid division by zero or tiny numbers
                mask.div_(mask_mean)
                update = update.mul(mask)
            else:
                update_before_cautious = update

                # 1. Calculate the "fully cautious" update
                mask = (update_before_cautious * grad > 0).to(grad.dtype)
                mask_mean = mask.mean().clamp_(min=1e-3) # Avoid division by zero or tiny numbers
                mask.div_(mask_mean)

                update_if_fully_cautious = update_before_cautious * mask

                update = (1 - scale) * update_before_cautious + scale * update_if_fully_cautious

        if update_strategy in {'grams', 'both'}:
            if scale >= 1.0:
                update = torch.sign(grad).mul_(update.abs())
            else:
                update_before_grams = update

                update_if_fully_grams = torch.sign(grad).mul_(update_before_grams.abs())

                update = (1 - scale) * update_before_grams + scale * update_if_fully_grams

    return update