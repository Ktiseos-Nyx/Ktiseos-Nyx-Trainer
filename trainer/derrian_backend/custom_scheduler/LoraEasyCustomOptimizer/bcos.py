# Source: https://github.com/facebookresearch/bcos
# 
# MIT License
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This license applies to all files in this repository except for those located
# in a directory with its own license file. In such cases, the license found in
# the closest ancestor directory to the file takes precedence.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.optim import Optimizer

from .utils import copy_stochastic_


class BCOS(Optimizer):
    def __init__(self, 
                params, 
                lr, 
                beta=0.9, 
                beta2=None, 
                eps=1e-6, 
                weight_decay=0.1, 
                mode='c', 
                decouple_wd=True, 
                simple_cond=False,
                sync_chunk_size: int = 128,
                state_storage_dtype: str|torch.dtype = torch.bfloat16,
                state_storage_device: str|torch.device = "cpu",
                **kwargs): 
        
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

        defaults = dict(lr=lr, beta=beta, beta2=beta2, eps=eps, wd=weight_decay, sync_chunk_size=sync_chunk_size, state_storage_dtype=final_dtype, state_storage_device=state_storage_device) 
        super().__init__(params, defaults)

        if mode not in ['g', 'm', 'c']:
            raise ValueError(f"BCOS mode {mode} not supported")
        self.mode = mode
        self.decouple_wd = decouple_wd      # True for BCOSW
        self.simple_cond = simple_cond # True for simple alternative v estimator in 'c' mode

    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta = group["beta"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["wd"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                device = p.device
                grad = p.grad.data

                state = self.state[p]

                # initialize optimizer states for specific modes
                if self.state_storage_device == "cpu":
                    if self.mode in ['m', 'c'] and 'm' not in state:
                        state["m"] = grad.detach().to(dtype=self.state_storage_dtype, 
                                                    device=self.state_storage_device).pin_memory()
                    if self.mode in ['g', 'm'] and 'v' not in state:
                        state["v"] = grad.detach().to(dtype=self.state_storage_dtype, 
                                                    device=self.state_storage_device).pin_memory()
                else:
                    if self.mode in ['m', 'c'] and 'm' not in state:
                        state["m"] = grad.detach().to(dtype=self.state_storage_dtype, 
                                                    device=self.state_storage_device)
                    if self.mode in ['g', 'm'] and 'v' not in state:
                        state["v"] = grad.detach().to(dtype=self.state_storage_dtype, 
                                                    device=self.state_storage_device)
                    
                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                if self.mode in ['m', 'c']:
                    m = state["m"].to(
                        compute_device, 
                        non_blocking=True, 
                        dtype=torch.float32
                    )
                if self.mode in ['g', 'm']:
                    v = state["v"].to(
                        compute_device, 
                        non_blocking=True, 
                        dtype=torch.float32
                    )
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                # decoupled weight decay or absorb in gradient
                if self.decouple_wd:    # p := (1 - lr * wd) * p
                    p_fp32.data.mul_(1 - lr * wd)
                else:                   # g := g + wd * p
                    grad.data.add_(p_fp32.data, alpha = wd)
                
                if self.mode in ['m', 'c']:
                    m = state['m']
                    if self.mode == 'c':    # conditional estimator
                        if not self.simple_cond:
                            # BCOS-c
                            v = (3 * beta**2 - 2 * beta**3) * m.square() + (1 - beta)**2 * grad.detach().square() + 2 * beta * (1-beta)**2 * m * grad.detach()
                        else:
                            # simple alternative:
                            betav = 1 - (1 - beta)**2 if beta2 is None else beta2
                            g2 = grad.detach().square()
                            v = betav * m.square() + (1 - betav) * g2
                    # update momentum
                    m.mul_(beta).add_(grad.detach(), alpha=1 - beta) 
                    d = m
                else:
                    d = grad.detach()
                
                if self.mode in ['g', 'm']:     # EMA estimator
                    v = state['v']
                    betav = beta if beta2 is None else beta2
                    v.mul_(betav).add_(d.square(), alpha=1 - betav)

                # BCOS update: p := p - lr * (d / (sqrt(v) + eps))
                p_fp32.data.add_(d.div(v.sqrt() + eps), alpha= - lr)

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
                    if self.mode in ['m', 'c']:
                        copy_stochastic_(state["m"], m)
                    if self.mode in ['g', 'm']:
                        copy_stochastic_(state["v"], v)
                else:
                    if self.mode in ['m', 'c']:
                        state["m"].copy_(m, non_blocking=True)
                    if self.mode in ['g', 'm']:
                        state["v"].copy_(v, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss