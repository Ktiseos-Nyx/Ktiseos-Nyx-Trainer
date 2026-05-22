# Authored originally by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/came.py
# With stochastic rounding added per https://github.com/neggles/neurosis/blob/main/src/neurosis/optimizers/came.py

import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import copy_stochastic_, UPDATE_STRATEGY
import logging

logger = logging.getLogger(__name__)


class CAME(BaseOptimizer):
    r"""Confidence-guided Adaptive Memory Efficient Optimization.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip_threshold: float. threshold of root-mean-square of final gradient update.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps1: float. term added to the denominator to improve numerical stability.
    :param eps2: float. term added to the denominator to improve numerical stability.
    :param cautious: bool: (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    :param update_strategy: str: (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085),
        'grams' (https://arxiv.org/abs/2412.17107), and 'both' (cautious then grams sequentially) (default: unmodified)
    :param sync_chunk_size: int: Size of chunks to sync between devices (default: 256)
    :param state_storage_dtype: str|torch.dtype: Data type for storing optimizer state (default: bfloat16)
    :param state_storage_device: str|torch.device: Device for storing optimizer state (default: cpu)
    :param cautious_weight_decay: bool: Applies weight decay only to parameter coordinates whose signs align with the optimizer update. (default: False)
    :param compile_step: bool: Use torch.compile on the core per-parameter step (default: False)
    :param foreach: bool: Use torch._foreach_* operations for unfactored (1D/0D) parameters (default: False)
    :param non_factored_confidence: bool: Apply confidence/residual mechanism to non-factored (1D/0D) parameters (default: False)
    :param kahan_sum: bool: Enable Kahan summation for parameter updates in low-precision (bf16/fp16) training.
        Tracks rounding error from each low-precision write-back and compensates in the next step,
        preventing small gradient updates from being lost to rounding. Only applies to bf16/fp16 parameters.
        (default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-5,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        sync_chunk_size: int = 256,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        cautious_weight_decay: bool = False,
        compile_step: bool = False,
        foreach: bool = False,
        non_factored_confidence: bool = False,
        kahan_sum: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

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

        # Caches to avoid per-parameter tensor allocations in compiled step
        self._scalar_cache: dict = {}       # (device, group_idx) -> dict of scalar tensors
        self._empty_tensor_cache: dict = {}  # device -> empty(0) tensor for AMSBound placeholder

        # Pre-allocated GPU staging buffers to avoid per-step allocation through the CUDA caching allocator.
        # Only used when state transfer involves device change (CPU→GPU) or dtype conversion (bfloat16→float32).
        # Maps id(p) -> dict of GPU FP32 tensors keyed by buffer name ('p_fp32', 'exp_avg', 'grad', etc.)
        self._staging_bufs: dict = {}

        # Single shared int32 scratch buffer for stochastic rounding.
        # Grows to the largest parameter size encountered; reused across all params each step.
        self._srng_buf: torch.Tensor | None = None

        # Staging is beneficial when state transfer requires work (device or dtype change).
        # When storage is already on GPU in float32, .to() is a no-op and staging would waste memory.
        self._use_staging: bool = not (
            str(state_storage_device).startswith("cuda") and final_dtype == torch.float32
        )

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams','both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2

        # Compiled step callables (lazily compiled on first step() call)
        self._compiled_factored = None
        self._compiled_unfactored = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps1': eps1,
            'eps2': eps2,
            'cautious':cautious,
            'update_strategy':update_strategy,
            'sync_chunk_size': sync_chunk_size,
            'state_storage_dtype': final_dtype,
            'state_storage_device': state_storage_device,
            'clip_threshold': clip_threshold,
            'cautious_weight_decay': cautious_weight_decay,
            'compile_step': compile_step,
            'foreach': foreach,
            'non_factored_confidence': non_factored_confidence,
            'kahan_sum': kahan_sum,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CAME'

    def init_group(self, group, **kwargs) -> None:
        pass

    def _init_param_state(self, p: torch.Tensor, group: dict, grad_shape: Tuple[int, ...], factored: bool) -> None:
        r"""Initialize optimizer state tensors for a parameter on first encounter.

        :param p: torch.Tensor. the parameter.
        :param group: dict. the parameter group.
        :param grad_shape: Tuple[int, ...]. shape of the gradient.
        :param factored: bool. whether to use factored second-moment estimator.
        """
        state = self.state[p]
        nfc = group.get('non_factored_confidence', False)

        state["exp_avg"] = torch.zeros_like(p,
                                        dtype=self.state_storage_dtype,
                                        device=self.state_storage_device)
        if factored:
            state['exp_avg_sq_row'] = torch.zeros(
                grad_shape[:-1],
                dtype=torch.float32,
                device=self.state_storage_device
            )
            state['exp_avg_sq_col'] = torch.zeros(
                grad_shape[:-2] + grad_shape[-1:],
                dtype=torch.float32,
                device=self.state_storage_device
            )
            state['exp_avg_res_row'] = torch.zeros(
                grad_shape[:-1],
                dtype=torch.float32,
                device=self.state_storage_device
            )
            state['exp_avg_res_col'] = torch.zeros(
                grad_shape[:-2] + grad_shape[-1:],
                dtype=torch.float32,
                device=self.state_storage_device
            )
        else:
            state['exp_avg_sq'] = torch.zeros(
                grad_shape,
                dtype=self.state_storage_dtype,
                device=self.state_storage_device
            )

        if group['ams_bound']:
            state['exp_avg_sq_hat'] = torch.zeros(
                grad_shape,
                dtype=self.state_storage_dtype,
                device=self.state_storage_device
            )

        # Non-factored confidence residual state
        if not factored and nfc:
            state['exp_avg_res'] = torch.zeros(
                grad_shape,
                dtype=torch.float32,
                device=self.state_storage_device
            )

        # Kahan compensation state for parameter (bf16/fp16 only)
        if group.get('kahan_sum', False) and p.dtype in {torch.float16, torch.bfloat16}:
            state['kahan_comp'] = torch.zeros(
                p.shape,
                dtype=torch.float32,
                device=self.state_storage_device
            )

        # Pin memory for CPU storage (enables async GPU transfers)
        if self.state_storage_device == "cpu":
            state["exp_avg"] = state["exp_avg"].pin_memory()

            if factored:
                state['exp_avg_sq_row'] = state["exp_avg_sq_row"].pin_memory()
                state['exp_avg_sq_col'] = state["exp_avg_sq_col"].pin_memory()
                state['exp_avg_res_row'] = state["exp_avg_res_row"].pin_memory()
                state['exp_avg_res_col'] = state["exp_avg_res_col"].pin_memory()
            else:
                state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()

            if group['ams_bound']:
                state['exp_avg_sq_hat'] = state['exp_avg_sq_hat'].pin_memory()

            if not factored and nfc:
                state['exp_avg_res'] = state['exp_avg_res'].pin_memory()

            if 'kahan_comp' in state:
                state['kahan_comp'] = state['kahan_comp'].pin_memory()

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                grad_shape: Tuple[int, ...] = p.grad.shape
                factored: bool = self.get_options(grad_shape)
                self._init_param_state(p, group, grad_shape, factored)

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        r"""Get `factored`."""
        return len(shape) >= 2

    # --- Compiled Core Functions ---

    @staticmethod
    @torch.no_grad()
    def _core_factored_full_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        exp_avg_res_row: torch.Tensor,
        exp_avg_res_col: torch.Tensor,
        exp_avg_sq_hat: torch.Tensor,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
        beta3: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        clip_threshold: torch.Tensor,
        use_amsbound: bool,
        lr: torch.Tensor,
        weight_decay: torch.Tensor,
        weight_decouple: bool,
        fixed_decay: bool,
        cautious_weight_decay: bool,
        use_cautious: bool,
        use_grams: bool,
        p_fp32: torch.Tensor,
    ) -> None:
        r"""Core factored per-parameter step INCLUDING weight decay, LR scale, update strategy, and param update.

        All inputs are FP32 tensors on the compute device.
        Modifies all state tensors and p_fp32 in-place.
        Branch booleans (use_amsbound, weight_decouple, fixed_decay, cautious_weight_decay,
        use_cautious, use_grams) are compile-time constants resolved during tracing.
        """
        # update = grad^2 + eps1
        update = torch.mul(grad, grad).add_(eps1)

        # Factored second moment EMA
        exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
        exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

        # Approximate sq grad as denominator
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=update)

        # AMSBound
        if use_amsbound:
            torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
            torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

        # Precondition gradient
        update.mul_(grad)

        # RMS clip — numel is static with dynamic=False, so math.sqrt is trace-time constant
        rms = update.norm(2) / math.sqrt(update.numel())
        clip_factor = (rms / clip_threshold).clamp_(min=1.0)
        update.div_(clip_factor)

        # Momentum
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        # Confidence (residual)
        res = update - exp_avg
        res.pow_(2).add_(eps2)

        exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
        exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

        # Approximate sq grad for confidence modulation
        r_factor_res = (exp_avg_res_row / exp_avg_res_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor_res = exp_avg_res_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor_res, c_factor_res, out=update)
        update.mul_(exp_avg)

        # === Weight decay (inlined for compilation; branch resolved at trace time) ===
        if cautious_weight_decay:
            # Cautious weight decay: apply WD only where gradient and param agree in sign
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - weight_decay * lr * cwd_mask)
        elif weight_decouple:
            wd_factor = 1.0 if fixed_decay else lr
            p_fp32.mul_(1.0 - weight_decay * wd_factor)
        else:
            # Standard (non-decoupled) weight decay: add scaled parameter to gradient
            grad.add_(p_fp32, alpha=weight_decay)

        # === LR scale ===
        update.mul_(lr)

        # === Update strategy (resolved at compile time) ===
        if use_cautious:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)
        if use_grams:
            update.copy_(torch.sign(grad) * update.abs())

        # === Parameter update ===
        p_fp32.add_(-update)

    @staticmethod
    @torch.no_grad()
    def _core_unfactored_full_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        exp_avg_res: torch.Tensor,
        exp_avg_sq_hat: torch.Tensor,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
        beta3: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        clip_threshold: torch.Tensor,
        use_amsbound: bool,
        use_nfc: bool,
        lr: torch.Tensor,
        weight_decay: torch.Tensor,
        weight_decouple: bool,
        fixed_decay: bool,
        cautious_weight_decay: bool,
        use_cautious: bool,
        use_grams: bool,
        p_fp32: torch.Tensor,
    ) -> None:
        r"""Core unfactored per-parameter step INCLUDING confidence, weight decay, LR scale, update strategy, and param update.

        All inputs are FP32 tensors on the compute device.
        Modifies all state tensors and p_fp32 in-place.
        Branch booleans are compile-time constants resolved during tracing.
        """
        # update = grad^2 + eps1
        update = torch.mul(grad, grad).add_(eps1)

        # EMA of squared gradient
        exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
        torch.rsqrt(exp_avg_sq, out=update)

        # AMSBound
        if use_amsbound:
            torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
            torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

        # Precondition gradient
        update.mul_(grad)

        # RMS clip -- numel is static with dynamic=False, so math.sqrt is trace-time constant
        rms = update.norm(2) / math.sqrt(update.numel())
        clip_factor = (rms / clip_threshold).clamp_(min=1.0)
        update.div_(clip_factor)

        # Momentum exp_avg = beta1*exp_avg + (1-beta1)*update
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        # Non-factored confidence: apply residual modulation
        if use_nfc:
            # update still holds pre-momentum value; res = (pre - post)^2 + eps2
            res = update.sub(exp_avg).pow_(2).add_(eps2)
            exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
            # update = exp_avg / (sqrt(exp_avg_res) + eps2)
            update.copy_(exp_avg).div_(exp_avg_res.sqrt().add_(eps2))
        else:
            # copy exp_avg into update so subsequent in-place ops don't corrupt exp_avg
            update.copy_(exp_avg)

        # === Weight decay (inlined for compilation; branch resolved at trace time) ===
        if cautious_weight_decay:
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - weight_decay * lr * cwd_mask)
        elif weight_decouple:
            wd_factor = 1.0 if fixed_decay else lr
            p_fp32.mul_(1.0 - weight_decay * wd_factor)
        else:
            grad.add_(p_fp32, alpha=weight_decay)

        # === LR scale ===
        update.mul_(lr)

        # === Update strategy (resolved at compile time) ===
        if use_cautious:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)
        if use_grams:
            update.copy_(torch.sign(grad) * update.abs())

        # === Parameter update ===
        p_fp32.add_(-update)

    def _compile_core_fns(self) -> None:
        r"""Lazily compile the core step functions with torch.compile."""
        if self.defaults.get('compile_step', False):
            try:
                # Raise recompile limit to accommodate diverse parameter shapes
                # (e.g. LoRA layers with [rank, 768], [rank, 320], [rank, 4096], etc.)
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 64
                )
                with torch._dynamo.utils.disable_cache_limit():
                    # Always compile the factored step (used for 2D+ params regardless of foreach)
                    self._compiled_factored = torch.compile(
                        self._core_factored_full_fp32, fullgraph=True, dynamic=False
                    )
                    self._compiled_unfactored = torch.compile(
                        self._core_unfactored_full_fp32, fullgraph=True, dynamic=False
                    )
                logger.info("CAME core functions compiled with torch.compile(fullgraph=True, dynamic=False).")
            except Exception as e:
                logger.warning(f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. Falling back to uncompiled step.")
                self._compiled_unfactored = self._core_unfactored_full_fp32
                self._compiled_factored = self._core_factored_full_fp32
        else:
            self._compiled_unfactored = self._core_unfactored_full_fp32
            self._compiled_factored = self._core_factored_full_fp32

    # --- Scalar Tensor Caching (avoids per-parameter allocation) ---

    def _get_scalar_tensors(
        self, device: torch.device, group_idx: int, group: dict
    ):
        r"""Get or create cached scalar tensors for a given (device, group)."""
        key = (device, group_idx)
        if key not in self._scalar_cache:
            self._scalar_cache[key] = {
                'beta1_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'beta2_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'beta3_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'eps1_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'eps2_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'clip_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'lr_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'wd_t': torch.tensor(0.0, device=device, dtype=torch.float32),
            }
        scalars = self._scalar_cache[key]
        betas = group['betas']
        scalars['beta1_t'].fill_(betas[0])
        scalars['beta2_t'].fill_(betas[1])
        scalars['beta3_t'].fill_(betas[2])
        scalars['eps1_t'].fill_(group['eps1'])
        scalars['eps2_t'].fill_(group['eps2'])
        scalars['clip_t'].fill_(group['clip_threshold'])
        scalars['lr_t'].fill_(group['lr'])
        scalars['wd_t'].fill_(group['weight_decay'])
        return scalars

    def _get_empty_tensor(self, device: torch.device) -> torch.Tensor:
        r"""Get or create cached empty tensor for AMSBound placeholder."""
        if device not in self._empty_tensor_cache:
            self._empty_tensor_cache[device] = torch.empty(0, device=device)
        return self._empty_tensor_cache[device]

    # --- GPU Staging Buffer Management (avoids per-step .to() allocations) ---

    def _create_staging(
        self,
        p: torch.Tensor,
        compute_device: torch.device,
        factored: bool,
        ams_bound: bool,
        nfc: bool,
        kahan: bool = False,
    ) -> dict:
        r"""Create pre-allocated GPU FP32 staging buffers for a parameter and cache them.

        Returns a dict with keys ``'p_fp32'``, ``'grad'``, ``'exp_avg'``, and shape-dependent
        state buffers (``'exp_avg_sq'`` for unfactored, ``'exp_avg_sq_row'``/``'exp_avg_sq_col'``
        for factored, plus optional ``'exp_avg_sq_hat'``, ``'exp_avg_res'``, and Kahan buffers).
        """
        grad_shape = tuple(p.shape)
        buf = {
            '_device': compute_device,
            '_shape': grad_shape,
            'p_fp32': torch.empty(grad_shape, dtype=torch.float32, device=compute_device),
            'grad': torch.empty(grad_shape, dtype=torch.float32, device=compute_device),
            'exp_avg': torch.empty(grad_shape, dtype=torch.float32, device=compute_device),
        }

        if factored:
            row_shape = grad_shape[:-1]
            col_shape = grad_shape[:-2] + grad_shape[-1:]
            buf['exp_avg_sq_row'] = torch.empty(row_shape, dtype=torch.float32, device=compute_device)
            buf['exp_avg_sq_col'] = torch.empty(col_shape, dtype=torch.float32, device=compute_device)
            buf['exp_avg_res_row'] = torch.empty(row_shape, dtype=torch.float32, device=compute_device)
            buf['exp_avg_res_col'] = torch.empty(col_shape, dtype=torch.float32, device=compute_device)
        else:
            buf['exp_avg_sq'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)

        if ams_bound:
            buf['exp_avg_sq_hat'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)

        if not factored and nfc:
            buf['exp_avg_res'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)
            # Dedicated scratch buffer for pre-momentum values (separate from exp_avg_res to avoid aliasing)
            buf['pre_mom_scratch'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)

        if kahan:
            buf['kahan_comp'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)
            buf['kahan_sim'] = torch.empty(grad_shape, dtype=torch.float32, device=compute_device)

        self._staging_bufs[id(p)] = buf
        return buf

    def _get_staging(
        self,
        p: torch.Tensor,
        compute_device: torch.device,
        factored: bool,
        ams_bound: bool,
        nfc: bool,
        kahan: bool = False,
    ) -> dict:
        r"""Get or create staging buffers for a parameter, validating device and shape."""
        pid = id(p)
        buf = self._staging_bufs.get(pid)
        p_shape = tuple(p.shape)
        if buf is not None and buf['_device'] == compute_device and buf['_shape'] == p_shape:
            # Check if kahan buffers are present when needed (handles stale cache from non-kahan creation)
            if kahan and 'kahan_comp' not in buf:
                return self._create_staging(p, compute_device, factored, ams_bound, nfc, kahan=True)
            # Check if pre_mom_scratch is present when needed (handles stale cache from non-nfc creation)
            if nfc and not factored and 'pre_mom_scratch' not in buf:
                return self._create_staging(p, compute_device, factored, ams_bound, nfc, kahan=kahan)
            return buf
        # Stale or missing — (re-)create
        return self._create_staging(p, compute_device, factored, ams_bound, nfc, kahan=kahan)

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

    # --- Foreach Support (Unfactored params only) ---

    @torch.no_grad()
    def _foreach_unfactored_step(
        self,
        group,
        active_params: list,
        beta1: float,
        beta2: float,
        beta3: float,
        compute_device: torch.device,
    ) -> None:
        r"""Foreach step for unfactored (1D/0D) parameters.

        Batches operations using ``torch._foreach_*`` for better GPU utilization.
        Handles AMSBound, update strategy, weight decay, and non-factored confidence.
        Uses pre-allocated staging buffers when ``_use_staging`` is True to avoid
        per-step tensor allocations through the CUDA caching allocator.
        """
        use_amsbound = group['ams_bound']
        update_strategy = group['update_strategy']
        lr = group['lr']
        wd = group['weight_decay']
        wd_decouple = group['weight_decouple']
        fixed_decay = group['fixed_decay']
        cwd = group['cautious_weight_decay']
        nfc = group['non_factored_confidence']
        use_kahan = group.get('kahan_sum', False)

        # Collect phase: build lists of FP32 tensors on compute device
        p_fp32_list = []
        grad_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        exp_avg_sq_hat_list = [] if use_amsbound else None
        exp_avg_res_list = [] if nfc else None
        kahan_comp_list = [] if use_kahan else None
        kahan_sim_list = [] if use_kahan else None
        param_kahan_flags = []  # per-param: whether this param has kahan enabled
        state_list = []
        srng_bufs = []  # stochastic rounding scratch per parameter

        for p in active_params:
            if p.grad is None:
                continue

            state = self.state[p]
            state_list.append(state)
            param_kahan = use_kahan and p.dtype in {torch.float16, torch.bfloat16}
            param_kahan_flags.append(param_kahan)

            if self._use_staging:
                staging = self._get_staging(p, compute_device, factored=False,
                                            ams_bound=use_amsbound, nfc=nfc, kahan=param_kahan)
                # Copy data into pre-allocated staging buffers (avoids .to() allocation)
                staging['p_fp32'].copy_(p, non_blocking=True)
                staging['grad'].copy_(p.grad.data, non_blocking=True)  # single .to(), no intermediate
                staging['exp_avg'].copy_(state["exp_avg"], non_blocking=True)
                staging['exp_avg_sq'].copy_(state["exp_avg_sq"], non_blocking=True)

                p_fp32_list.append(staging['p_fp32'])
                grad_list.append(staging['grad'])
                exp_avg_list.append(staging['exp_avg'])
                exp_avg_sq_list.append(staging['exp_avg_sq'])

                if use_amsbound:
                    staging['exp_avg_sq_hat'].copy_(state["exp_avg_sq_hat"], non_blocking=True)
                    exp_avg_sq_hat_list.append(staging['exp_avg_sq_hat'])

                if nfc:
                    staging['exp_avg_res'].copy_(state['exp_avg_res'], non_blocking=True)
                    exp_avg_res_list.append(staging['exp_avg_res'])

                if param_kahan:
                    staging['kahan_comp'].copy_(state['kahan_comp'], non_blocking=True)
                    kahan_comp_list.append(staging['kahan_comp'])
                    kahan_sim_list.append(staging['kahan_sim'])

                # Pre-allocate stochastic rounding scratch (shared across write-backs for this param)
                srng_bufs.append(self._get_srng_buf(staging['exp_avg']))
            else:
                # Fallback: .to() path (no-op when already on compute device in float32)
                p_fp32_list.append(p.to(compute_device, dtype=torch.float32, non_blocking=True))
                grad_list.append(p.grad.data.to(compute_device, dtype=torch.float32, non_blocking=True))
                exp_avg_list.append(state["exp_avg"].to(compute_device, non_blocking=True, dtype=torch.float32))
                exp_avg_sq_list.append(state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32))

                if use_amsbound:
                    exp_avg_sq_hat_list.append(
                        state["exp_avg_sq_hat"].to(compute_device, non_blocking=True, dtype=torch.float32)
                    )

                if nfc:
                    exp_avg_res_list.append(
                        state['exp_avg_res'].to(compute_device, non_blocking=True, dtype=torch.float32)
                    )

                if param_kahan:
                    kahan_comp_list.append(
                        state['kahan_comp'].to(compute_device, non_blocking=True, dtype=torch.float32)
                    )
                    kahan_sim_list.append(torch.empty_like(p, dtype=torch.float32, device=compute_device))

                srng_bufs.append(None)

        if not p_fp32_list:
            return

        # ---- Kahan pre-compensation (per-param, before foreach batch) ----
        if use_kahan and kahan_comp_list:
            for i, p in enumerate(active_params):
                if p.grad is None:
                    continue
                if param_kahan_flags[i]:
                    # Map from active_params index to kahan_comp_list index
                    k_idx = sum(1 for j in range(i) if param_kahan_flags[j])
                    p_fp32_list[i].add_(kahan_comp_list[k_idx])

        # ---- Batch compute phase ----

        # 1. EMA of squared gradient: exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * (grad^2 + eps1)
        update_list = torch._foreach_mul(grad_list, grad_list)  # grad^2 — batched
        torch._foreach_add_(update_list, group['eps1'])  # grad^2 + eps1

        torch._foreach_mul_(exp_avg_sq_list, beta2)
        torch._foreach_add_(exp_avg_sq_list, update_list, alpha=1.0 - beta2)
        # exp_avg_sq_list now holds correct EMA — preserved for write-back

        # 2. Compute denom = rsqrt(EMA) into update_list (reuse allocation, avoids new allocs)
        for _i in range(len(update_list)):
            update_list[_i].copy_(exp_avg_sq_list[_i])
        torch._foreach_rsqrt_(update_list)

        # AMSBound — update hat state, then compute denom from hat
        if use_amsbound:
            # hat = max(hat, 1/denom) = max(hat, sqrt(EMA))
            torch._foreach_max_(exp_avg_sq_hat_list, [1.0 / d for d in update_list])
            # Compute denom = rsqrt(hat / beta2) into update_list (reuse allocation)
            for _i in range(len(update_list)):
                update_list[_i].copy_(exp_avg_sq_hat_list[_i])
            torch._foreach_div_(update_list, beta2)
            torch._foreach_rsqrt_(update_list)
        denom_list = update_list

        # 3. Precondition: denom *= grad
        torch._foreach_mul_(denom_list, grad_list)

        # 4. RMS clip (per-tensor since norm/numel differ)
        for upd in denom_list:
            rms = upd.norm(2) / math.sqrt(upd.numel())
            clip_factor = max(rms / group['clip_threshold'], 1.0)
            upd.div_(clip_factor)

        # Save pre-momentum update values for optional confidence computation.
        # When nfc=True, use pre-allocated staging scratch to avoid N clone allocations.
        if nfc:
            if self._use_staging:
                # Use dedicated pre_mom_scratch buffer (separate from exp_avg_res to avoid aliasing)
                pre_momentum_updates = []
                for _i, p in enumerate(active_params):
                    if p.grad is None:
                        continue
                    staging = self._get_staging(p, compute_device, factored=False,
                                                ams_bound=use_amsbound, nfc=nfc)
                    scratch = staging['pre_mom_scratch']
                    scratch.copy_(denom_list[_i])
                    pre_momentum_updates.append(scratch)
            else:
                pre_momentum_updates = [d.clone() for d in denom_list]

        # 4. Momentum: exp_avg = beta1 * exp_avg + (1-beta1) * update
        torch._foreach_mul_(exp_avg_list, beta1)
        torch._foreach_add_(exp_avg_list, denom_list, alpha=1.0 - beta1)

        # 5. Confidence residual modulation (non-factored) — batched via foreach
        if nfc:
            # Reuse pre_momentum_updates as scratch buffer (clones of denom_list, no longer needed)
            # Compute residual in-place: res = (pre_momentum_update - exp_avg)^2 + eps2
            torch._foreach_sub_(pre_momentum_updates, exp_avg_list)
            torch._foreach_mul_(pre_momentum_updates, pre_momentum_updates)
            torch._foreach_add_(pre_momentum_updates, group['eps2'])

            # EMA of residual: exp_avg_res = beta3 * exp_avg_res + (1-beta3) * res
            torch._foreach_mul_(exp_avg_res_list, beta3)
            torch._foreach_add_(exp_avg_res_list, pre_momentum_updates, alpha=1.0 - beta3)

            # final_update = exp_avg / (sqrt(exp_avg_res) + eps2)
            # Copy exp_avg_res into scratch, then sqrt in-place to preserve exp_avg_res state
            for i in range(len(pre_momentum_updates)):
                pre_momentum_updates[i].copy_(exp_avg_res_list[i])
            torch._foreach_sqrt_(pre_momentum_updates)
            torch._foreach_add_(pre_momentum_updates, group['eps2'])
            # Reciprocal in-place then multiply by exp_avg to avoid _foreach_div allocation
            torch._foreach_pow_(pre_momentum_updates, -1.0)
            torch._foreach_mul_(pre_momentum_updates, exp_avg_list)
            final_update_list = pre_momentum_updates
        else:
            # update = exp_avg (no confidence)
            final_update_list = exp_avg_list

        # 6. Weight decay (inlined foreach variant — mirrors _core_unfactored_full_fp32)
        if cwd:
            # Cautious weight decay: apply WD only where gradient and param agree in sign
            wd_scaled = wd * lr
            for i in range(len(p_fp32_list)):
                cwd_mask = (grad_list[i] * p_fp32_list[i] >= 0).to(p_fp32_list[i].dtype)
                p_fp32_list[i].mul_(1.0 - wd_scaled * cwd_mask)
        elif wd_decouple:
            wd_factor = 1.0 if fixed_decay else lr
            torch._foreach_mul_(p_fp32_list, 1.0 - wd * wd_factor)
        elif wd > 0.0:
            # Standard (non-decoupled) weight decay: add scaled parameter to gradient
            torch._foreach_add_(grad_list, p_fp32_list, alpha=wd)

        # 7. LR scale
        torch._foreach_mul_(final_update_list, lr)

        # 8. Update strategy (inlined foreach variant — mirrors _core_unfactored_full_fp32)
        if update_strategy in ('cautious', 'both'):
            # Cautious: mask = (update * grad > 0), normalized by mean
            mask_list = []
            for i in range(len(final_update_list)):
                mask = (final_update_list[i] * grad_list[i] > 0).to(grad_list[i].dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                mask_list.append(mask)
            torch._foreach_mul_(final_update_list, mask_list)
        if update_strategy in ('grams', 'both'):
            # Grams: update = sign(grad) * |update|
            # In-place variant: 1N allocations (sign) instead of 3N when nfc=True,
            # or 2N instead of 3N when nfc=False (abs list + in-place mul).
            sign_list = torch._foreach_sign(grad_list)  # N allocs (unavoidable — can't in-place on grad_list)
            if nfc:
                # Safe to mutate in-place: final_update_list points to pre_momentum scratch buffers
                torch._foreach_abs_(final_update_list)
                torch._foreach_mul_(final_update_list, sign_list)
            else:
                # Can't mutate in-place: final_update_list aliases exp_avg_list (optimizer state)
                abs_list = torch._foreach_abs(final_update_list)  # N allocs
                torch._foreach_mul_(abs_list, sign_list)  # in-place on abs_list
                final_update_list = abs_list

        # 9. Apply: p -= update
        torch._foreach_add_(p_fp32_list, final_update_list, alpha=-1.0)

        # ---- Write-back phase with sync chunking ----
        kahan_iter = 0  # tracks position in kahan_comp_list / kahan_sim_list
        for i, state in enumerate(state_list):
            p = active_params[i]
            p_fp32 = p_fp32_list[i]
            device = p.device
            srng = srng_bufs[i] if i < len(srng_bufs) else None

            # Parameter write-back (with optional Kahan compensation)
            param_kahan = param_kahan_flags[i]
            if param_kahan:
                kahan_sim = kahan_sim_list[kahan_iter]
                kahan_comp = kahan_comp_list[kahan_iter]
                kahan_iter += 1

                # Simulate rounding to compute new compensation
                kahan_sim.copy_(p_fp32)
                if p.dtype == torch.bfloat16:
                    # Simulate stochastic rounding (same bit manipulation as copy_stochastic_)
                    sim_int = kahan_sim.view(dtype=torch.int32)
                    if srng is not None:
                        srng.random_(0, 1 << 16)
                        sim_int.add_(srng)
                    else:
                        sim_int.add_(torch.randint_like(sim_int, 0, 1 << 16))
                    sim_int.bitwise_and_(-65536)
                else:
                    # fp16: simulate deterministic rounding
                    kahan_sim.copy_(p_fp32.to(p.dtype).to(torch.float32))

                # Write rounded value to parameter
                if device.type == "cpu":
                    p.data.copy_(kahan_sim)
                else:
                    p.data.copy_(kahan_sim, non_blocking=True)

                # Compute new compensation: rounded - exact
                kahan_sim.sub_(p_fp32)

                # Store compensation back to state
                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state['kahan_comp'], kahan_sim, scratch=srng)
                else:
                    state['kahan_comp'].copy_(kahan_sim, non_blocking=True)
            else:
                # Standard parameter write-back
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
            exp_avg = exp_avg_list[i]
            exp_avg_sq = exp_avg_sq_list[i]

            if self.state_storage_dtype == torch.bfloat16:
                copy_stochastic_(state["exp_avg"], exp_avg, scratch=srng)
                copy_stochastic_(state["exp_avg_sq"], exp_avg_sq, scratch=srng)
                if use_amsbound:
                    copy_stochastic_(state["exp_avg_sq_hat"], exp_avg_sq_hat_list[i], scratch=srng)
            else:
                state["exp_avg"].copy_(exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)
                if use_amsbound:
                    state["exp_avg_sq_hat"].copy_(exp_avg_sq_hat_list[i], non_blocking=True)

            if nfc:
                ear = exp_avg_res_list[i]
                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state['exp_avg_res'], ear, scratch=srng)
                else:
                    state['exp_avg_res'].copy_(ear, non_blocking=True)

            # Sync chunking
            if (i + 1) % group.get('sync_chunk_size', 256) == 0:
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

            beta1, beta2, beta3 = group['betas']
            use_foreach = group.get('foreach', False)
            nfc = group.get('non_factored_confidence', False)
            use_kahan = group.get('kahan_sum', False)
            update_strategy = group['update_strategy']

            # Lazily compile core functions on first step
            if self._compiled_unfactored is None:
                self._compile_core_fns()

            # Select compiled or uncompiled callables
            core_unfactored_fn = self._compiled_unfactored
            core_factored_fn = self._compiled_factored

            # Pre-compute group_idx once (avoids O(n) list.index() per parameter)
            group_idx = self.param_groups.index(group)

            # Bucket params for foreach (unfactored only, numel >= 16)
            # Bypass foreach for 0D and very small tensors (numel < 16) where
            # kernel launch overhead dominates and per-param compiled path is faster.
            unfactored_foreach_params = []
            if use_foreach:
                compute_device_for_foreach = None
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_shape = p.grad.shape
                    if not self.get_options(grad_shape) and p.numel() >= 16:
                        unfactored_foreach_params.append(p)
                        if compute_device_for_foreach is None:
                            first_device = p.device
                            compute_device_for_foreach = (
                                torch.cuda.current_device() if first_device.type == "cpu" else first_device
                            )

                if unfactored_foreach_params and compute_device_for_foreach is not None:
                    self._foreach_unfactored_step(
                        group, unfactored_foreach_params, beta1, beta2, beta3, compute_device_for_foreach
                    )

            # Per-parameter loop for factored params or when foreach is disabled
            processed = 0
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                # Skip unfactored params that were already handled by foreach
                # (only those with numel  and p.numel() >= 16; small/0D params fall through to per-param path)
                if use_foreach and not self.get_options(grad.shape) and p.numel() >= 16:
                    continue

                state = self.state[p]
                device = p.device

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                if len(state) == 0:
                    self._init_param_state(p, group, grad_shape, factored)

                # ========= Determine compute device =========
                if device.type == "cpu":
                    compute_device = torch.cuda.current_device()
                else:
                    compute_device = device

                # ========= Per-parameter Kahan flag =========
                param_kahan = use_kahan and p.dtype in {torch.float16, torch.bfloat16}

                # ========= Transfer state to compute device (staging or .to()) =========
                if self._use_staging:
                    staging = self._get_staging(p, compute_device, factored,
                                                group['ams_bound'], nfc, kahan=param_kahan)
                    staging['exp_avg'].copy_(state["exp_avg"], non_blocking=True)
                    exp_avg = staging['exp_avg']

                    if factored:
                        staging['exp_avg_sq_row'].copy_(state["exp_avg_sq_row"], non_blocking=True)
                        staging['exp_avg_sq_col'].copy_(state["exp_avg_sq_col"], non_blocking=True)
                        staging['exp_avg_res_row'].copy_(state["exp_avg_res_row"], non_blocking=True)
                        staging['exp_avg_res_col'].copy_(state["exp_avg_res_col"], non_blocking=True)
                        exp_avg_sq_row = staging['exp_avg_sq_row']
                        exp_avg_sq_col = staging['exp_avg_sq_col']
                        exp_avg_res_row = staging['exp_avg_res_row']
                        exp_avg_res_col = staging['exp_avg_res_col']
                    else:
                        staging['exp_avg_sq'].copy_(state["exp_avg_sq"], non_blocking=True)
                        exp_avg_sq = staging['exp_avg_sq']

                    if group['ams_bound']:
                        staging['exp_avg_sq_hat'].copy_(state["exp_avg_sq_hat"], non_blocking=True)
                        exp_avg_sq_hat = staging['exp_avg_sq_hat']

                    staging['grad'].copy_(grad, non_blocking=True)  # single .to(), no intermediate
                    grad = staging['grad']
                    staging['p_fp32'].copy_(p, non_blocking=True)
                    p_fp32 = staging['p_fp32']

                    if param_kahan:
                        staging['kahan_comp'].copy_(state['kahan_comp'], non_blocking=True)
                        kahan_comp = staging['kahan_comp']
                        kahan_sim = staging['kahan_sim']

                    # Stochastic rounding scratch for this parameter
                    srng = self._get_srng_buf(exp_avg)
                else:
                    # Fallback: .to() path (no-op when already on compute device in float32)
                    exp_avg = state["exp_avg"].to(
                        compute_device, non_blocking=True, dtype=torch.float32
                    )
                    if factored:
                        exp_avg_sq_row = state["exp_avg_sq_row"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )
                        exp_avg_sq_col = state["exp_avg_sq_col"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )
                        exp_avg_res_row = state["exp_avg_res_row"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )
                        exp_avg_res_col = state["exp_avg_res_col"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )
                    else:
                        exp_avg_sq = state["exp_avg_sq"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )

                    if group['ams_bound']:
                        exp_avg_sq_hat = state["exp_avg_sq_hat"].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )

                    grad = grad.to(compute_device, dtype=torch.float32, non_blocking=True)  # single .to()
                    p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)
                    srng = None

                    if param_kahan:
                        kahan_comp = state['kahan_comp'].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )
                        kahan_sim = torch.empty_like(p_fp32)

                # ========= Kahan pre-compensation =========
                if param_kahan:
                    p_fp32.add_(kahan_comp)

                # ========= Core computation (compiled or uncompiled) =========
                if factored:
                    scalars = self._get_scalar_tensors(compute_device, group_idx, group)

                    ams_hat = exp_avg_sq_hat if group['ams_bound'] else self._get_empty_tensor(compute_device)

                    core_factored_fn(
                        grad, exp_avg,
                        exp_avg_sq_row, exp_avg_sq_col,
                        exp_avg_res_row, exp_avg_res_col,
                        ams_hat,
                        scalars['beta1_t'], scalars['beta2_t'], scalars['beta3_t'],
                        scalars['eps1_t'], scalars['eps2_t'], scalars['clip_t'],
                        group['ams_bound'],
                        scalars['lr_t'], scalars['wd_t'],
                        group['weight_decouple'],
                        group['fixed_decay'],
                        group['cautious_weight_decay'],
                        update_strategy in {'cautious', 'both'},
                        update_strategy in {'grams', 'both'},
                        p_fp32,
                    )
                else:
                    # Unfactored path (not handled by foreach) -- use full compiled step
                    scalars = self._get_scalar_tensors(compute_device, group_idx, group)

                    ams_hat = exp_avg_sq_hat if group['ams_bound'] else self._get_empty_tensor(compute_device)

                    if nfc:
                        if 'exp_avg_res' not in state:
                            # Lazy creation if state was already initialized without nfc
                            state['exp_avg_res'] = torch.zeros_like(
                                grad, dtype=torch.float32, device=self.state_storage_device
                            )
                            if self.state_storage_device == "cpu":
                                state['exp_avg_res'] = state['exp_avg_res'].pin_memory()
                        if self._use_staging:
                            staging_buf = self._get_staging(p, compute_device, factored=False,
                                                            ams_bound=group['ams_bound'], nfc=True)
                            staging_buf['exp_avg_res'].copy_(state['exp_avg_res'], non_blocking=True)
                            exp_avg_res_nonfac = staging_buf['exp_avg_res']
                        else:
                            exp_avg_res_nonfac = state['exp_avg_res'].to(
                                compute_device, non_blocking=True, dtype=torch.float32
                            )
                    else:
                        exp_avg_res_nonfac = self._get_empty_tensor(compute_device)

                    core_unfactored_fn(
                        grad, exp_avg, exp_avg_sq,
                        exp_avg_res_nonfac,
                        ams_hat,
                        scalars['beta1_t'], scalars['beta2_t'], scalars['beta3_t'],
                        scalars['eps1_t'], scalars['eps2_t'], scalars['clip_t'],
                        group['ams_bound'],
                        nfc,
                        scalars['lr_t'], scalars['wd_t'],
                        group['weight_decouple'],
                        group['fixed_decay'],
                        group['cautious_weight_decay'],
                        update_strategy in {'cautious', 'both'},
                        update_strategy in {'grams', 'both'},
                        p_fp32,
                    )

                # ========= Write-back (with optional Kahan compensation) =========
                if param_kahan:
                    # Kahan write-back: simulate rounding, compute compensation, write parameter
                    kahan_sim.copy_(p_fp32)
                    if p.dtype == torch.bfloat16:
                        # Simulate stochastic rounding (same bit manipulation as copy_stochastic_)
                        sim_int = kahan_sim.view(dtype=torch.int32)
                        if srng is not None:
                            srng.random_(0, 1 << 16)
                            sim_int.add_(srng)
                        else:
                            sim_int.add_(torch.randint_like(sim_int, 0, 1 << 16))
                        sim_int.bitwise_and_(-65536)
                    else:
                        # fp16: simulate deterministic rounding
                        kahan_sim.copy_(p_fp32.to(p.dtype).to(torch.float32))

                    # Write rounded value to parameter
                    if device.type == "cpu":
                        p.data.copy_(kahan_sim)
                    else:
                        p.data.copy_(kahan_sim, non_blocking=True)

                    # Compute new compensation: rounded - exact
                    kahan_sim.sub_(p_fp32)

                    # Store compensation back to state
                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state['kahan_comp'], kahan_sim, scratch=srng)
                    else:
                        state['kahan_comp'].copy_(kahan_sim, non_blocking=True)
                else:
                    # Standard parameter write-back
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
                    if not factored:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq, scratch=srng)
                    if group['ams_bound']:
                        copy_stochastic_(state["exp_avg_sq_hat"], exp_avg_sq_hat, scratch=srng)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    if not factored:
                        state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)
                    if group['ams_bound']:
                        state["exp_avg_sq_hat"].copy_(exp_avg_sq_hat, non_blocking=True)

                if factored:
                    state["exp_avg_sq_row"].copy_(exp_avg_sq_row, non_blocking=True)
                    state["exp_avg_sq_col"].copy_(exp_avg_sq_col, non_blocking=True)
                    state["exp_avg_res_row"].copy_(exp_avg_res_row, non_blocking=True)
                    state["exp_avg_res_col"].copy_(exp_avg_res_col, non_blocking=True)

                # Non-factored confidence write-back
                if not factored and nfc and 'exp_avg_res' in state:
                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state['exp_avg_res'], exp_avg_res_nonfac, scratch=srng)
                    else:
                        state['exp_avg_res'].copy_(exp_avg_res_nonfac, non_blocking=True)

                # ========= Sync chunking (use processed count, not raw loop index) =========
                processed += 1
                if processed % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization
            torch.cuda.synchronize()

        return loss
