# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

from .utils import copy_stochastic_

class RAdamScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free RAdam
    Neither warmup hyperparameter nor scheduler is needed with this optimizer.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default False).
        silent_sgd_phase (bool): If True, the optimizer will not use the first SGD phase of RAdam.
            This means that the optimizer will not update model parameters during the early training
            steps (e.g., < 5 when β_2 = 0.999), but just update the momentum values of the optimizer.
            This helps stabilize training by ensuring smoother warmup behavior and more reliable
            calculation of the moving average coefficient (`ckp1`). Recommended to set to True
            (default True).
        sync_chunk_size (int): Size of chunks to sync between devices (default: 128)
        state_storage_dtype (str|torch.dtype): Data type for storing optimizer state (default: bfloat16)
        state_storage_device (str|torch.device): Device for storing optimizer state (default: cpu)
    """

    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = False,
                 silent_sgd_phase: bool = True,
                 sync_chunk_size: int = 128,
                 state_storage_dtype: Union[str, torch.dtype] = torch.bfloat16,
                 state_storage_device: Union[str, torch.device] = "cpu",
                 ):
        
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

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach,
                        silent_sgd_phase=silent_sgd_phase,
                        sync_chunk_size=sync_chunk_size,
                        state_storage_dtype=final_dtype,
                        state_storage_device=state_storage_device)
        super().__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['k'] = 0
            for p in group['params']:
                state = self.state[p]

                state['z'] = torch.clone(p, memory_format=torch.preserve_format).to(
                    dtype=self.state_storage_dtype, device=self.state_storage_device)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(
                    dtype=self.state_storage_dtype, device=self.state_storage_device)

                if str(self.state_storage_device) == "cpu":
                    state['z'] = state['z'].pin_memory()
                    state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p_fp32 = p
                        z = state["z"]

                        # Set p to x
                        if p.dtype == torch.bfloat16:
                            z = z.to(device=p.device, dtype=torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)
                            p_fp32.lerp_(end=z, weight=1 - 1 / beta1)
                            copy_stochastic_(p, p_fp32)
                        else:
                            z = z.to(device=p.device, dtype=torch.float32)
                            p.lerp_(end=z, weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p_fp32 = p
                        z = state["z"]

                        # Set p to y
                        if p.dtype == torch.bfloat16:
                            z = z.to(device=p.device, dtype=torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)
                            p_fp32.lerp_(end=z, weight=1 - beta1)
                            copy_stochastic_(p, p_fp32)
                        else:
                            z = z.to(device=p.device, dtype=torch.float32)
                            p.lerp_(end=z, weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            silent_sgd_phase = group["silent_sgd_phase"]
            k = group["k"]  # current steps
            step = k + 1
            r = group['r']
            weight_lr_power = group['weight_lr_power']
            state_storage_dtype = group['state_storage_dtype']
            state_storage_device = group['state_storage_device']

            beta2_t = beta2**step
            bias_correction2 = 1 - beta2_t

            # maximum length of the approximated SMA
            rho_inf = 2 / (1 - beta2) - 1
            # compute the length of the approximated SMA
            rho_t = rho_inf - 2 * step * beta2_t / bias_correction2
            rect = (
                ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                if rho_t > 4.0
                else float(not silent_sgd_phase)
            )

            lr = group["lr"] * rect
            group["scheduled_lr"] = lr  # For logging purposes

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = (step**r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            ckp1 = weight / weight_sum if weight_sum != 0 else 0

            adaptive_y_lr = lr * (beta1 * (1 - ckp1) - 1)
            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    # Create states on the storage device/dtype
                    storage_z = torch.clone(p, memory_format=torch.preserve_format).to(
                        dtype=state_storage_dtype, device=state_storage_device)
                    storage_exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format).to(
                        dtype=state_storage_dtype, device=state_storage_device)

                    if str(state_storage_device) == "cpu":
                        storage_z = storage_z.pin_memory()
                        storage_exp_avg_sq = storage_exp_avg_sq.pin_memory()

                    self.state[p]["z"] = storage_z
                    self.state[p]["exp_avg_sq"] = storage_exp_avg_sq

            if group["foreach"] and len(active_p) > 0:
                # ========= Foreach path: move all states to compute device as fp32 =========
                y_fp32_list = []
                grad_fp32_list = []
                z_fp32_list = []
                exp_avg_sq_fp32_list = []

                for p in active_p:
                    # Determine compute device
                    if p.device.type == "cpu":
                        compute_device = torch.cuda.current_device()
                    else:
                        compute_device = p.device

                    state = self.state[p]

                    z_fp32 = state["z"].to(compute_device, non_blocking=True, dtype=torch.float32)
                    exp_avg_sq_fp32 = state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32)
                    grad_fp32 = p.grad.to(dtype=torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                    y_fp32_list.append(p_fp32)
                    grad_fp32_list.append(grad_fp32)
                    z_fp32_list.append(z_fp32)
                    exp_avg_sq_fp32_list.append(exp_avg_sq_fp32)

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq_fp32_list, beta2)
                torch._foreach_addcmul_(exp_avg_sq_fp32_list, grad_fp32_list, grad_fp32_list, value=1 - beta2)

                if rho_t > 4.0:
                    # Adam step
                    denom = torch._foreach_div(exp_avg_sq_fp32_list, bias_correction2)
                    torch._foreach_sqrt_(denom)
                    torch._foreach_add_(denom, eps)

                    # Normalize grad in-place for memory efficiency
                    torch._foreach_div_(grad_fp32_list, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad_fp32_list, y_fp32_list, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y_fp32_list, z_fp32_list, weight=ckp1)
                torch._foreach_add_(y_fp32_list, grad_fp32_list, alpha=adaptive_y_lr)

                # z step
                torch._foreach_sub_(z_fp32_list, grad_fp32_list, alpha=lr)

                # ========= Copy results back =========
                for i, p in enumerate(active_p):
                    state = self.state[p]

                    # Copy p back (with stochastic rounding if bf16)
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, y_fp32_list[i])
                    else:
                        p.data.copy_(y_fp32_list[i], non_blocking=True)

                    # Copy z back to storage
                    if state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["z"], z_fp32_list[i])
                    else:
                        state["z"].copy_(z_fp32_list[i], non_blocking=True)

                    # Copy exp_avg_sq back to storage
                    if state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq_fp32_list[i])
                    else:
                        state["exp_avg_sq"].copy_(exp_avg_sq_fp32_list[i], non_blocking=True)

                    # Synchronize after processing chunks
                    if (i + 1) % self.sync_chunk_size == 0:
                        torch.cuda.synchronize()
            else:
                for i, p in enumerate(active_p):
                    # Determine compute device
                    if p.device.type == "cpu":
                        compute_device = torch.cuda.current_device()
                    else:
                        compute_device = p.device

                    grad = p.grad
                    state = self.state[p]

                    # Move states to compute device as fp32
                    z_fp32 = state["z"].to(compute_device, non_blocking=True, dtype=torch.float32)
                    exp_avg_sq_fp32 = state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32)
                    grad_fp32 = grad.to(dtype=torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                    exp_avg_sq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                    if rho_t > 4.0:
                        # Adam step
                        denom = exp_avg_sq_fp32.div(bias_correction2).sqrt_().add_(eps)

                        # Reuse grad buffer for memory efficiency
                        grad_normalized = grad_fp32.div_(denom)
                    else:
                        # Fall back to SGD (or nothing)
                        grad_normalized = grad_fp32

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(p_fp32, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    p_fp32.lerp_(end=z_fp32, weight=ckp1)
                    p_fp32.add_(grad_normalized, alpha=adaptive_y_lr)

                    # z step
                    z_fp32.sub_(grad_normalized, alpha=lr)

                    # Copy p back (with stochastic rounding if bf16)
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

                    # Copy z back to storage
                    if state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["z"], z_fp32)
                    else:
                        state["z"].copy_(z_fp32, non_blocking=True)

                    # Copy exp_avg_sq back to storage
                    if state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq_fp32)
                    else:
                        state["exp_avg_sq"].copy_(exp_avg_sq_fp32, non_blocking=True)

                    # Synchronize after processing chunks
                    if (i + 1) % self.sync_chunk_size == 0:
                        torch.cuda.synchronize()

            # Final synchronization for the group
            torch.cuda.synchronize()

            group["k"] = k + 1
        return loss
