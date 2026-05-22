import torch
from torch.optim import Optimizer

from .utils import copy_stochastic_, adaptive_eps

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

# FishMonger from https://github.com/Clybius/Personalized-Optimizers/blob/main/FishMonger.py by Clybius
class FishMonger(Optimizer):
    r"""
    FishMonger: Screw it, fisher everything.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.001)
        betas (Tuple[float, float, float], optional):
            coefficients used for computing running averages of
            fim, momentum, and its square (default: (0.9, 0.99, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-30).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
        diff_amp (float):
            Accelerate the difference between the current and past gradient by this multiplicative value (default: 1.0).
        diff_amp_beta (float):
            Coefficient used for computing running average of the current and past gradients (default: 0.999).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99, 0.999),
        eps=1e-8,
        eps2=0.01,
        eps_floor=1e-16,
        weight_decay=0.0,
        clip=1.0,
        centralization=1.0,
        diff_amp=1.0,
        diff_amp_beta=0.999,
        **kwargs,
    ):
        
        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
            diff_amp=diff_amp,
            diff_amp_beta=diff_amp_beta,
        )

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        super(FishMonger, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FishMonger'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            clip = 1.0 if (group["clip"] <= 0) else group["clip"]
            centralization = group["centralization"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FishMonger does not support sparse gradients")

                state = self.state[p]

                # State initialization
                diff_amp = group["diff_amp"]
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p.data)
                    state["momentum_slow"] = torch.zeros_like(p.data)
                    state["momentum_slow_squared"] = torch.zeros_like(p.data)
                    # Fisher Information Matrix
                    state["fim"] = torch.ones_like(p.data)
                    # Previous grad
                    if diff_amp:
                        state["ema_diff"] = torch.zeros_like(p.data)
                        state["previous_grad"] = grad.data.clone().mul_(-1.0)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32).data
                    momentum, momentum_slow, momentum_slow_squared, fim = state["momentum"].to(torch.float32), state["momentum_slow"].to(torch.float32), state["momentum_slow_squared"].to(torch.float32), state["fim"].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema_diff = state["ema_diff"].to(torch.float32) if diff_amp else 0
                else:
                    grad = grad.data
                    momentum, momentum_slow, momentum_slow_squared, fim = state["momentum"], state["momentum_slow"], state["momentum_slow_squared"], state["fim"]
                    ema_diff = state["ema_diff"] if diff_amp else 0

                if diff_amp:
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff = state["previous_grad"].to(torch.float32)
                    else:
                        grad_diff = state["previous_grad"]
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff.add_(grad)

                    # Smooth the difference between previous grad and current grad
                    ema_diff.mul_(group["diff_amp_beta"]).add_(grad_diff, alpha=1 - group["diff_amp_beta"])

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["previous_grad"], -grad)
                    else:
                        state["previous_grad"].copy_(-grad)

                # Momentumize the gradient
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)

                # bias correction step size
                # soft warmup
                fim_beta = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0) # Start at 0 beta (I believe this is from Adafactor?)
                bias_correction = 1 - beta1**group["step"]

                # Update fim
                fim.mul_(fim_beta).addcmul_(momentum, momentum, value=1 - fim_beta)

                curr_eps = adaptive_eps(grad, group)

                fim_base = fim.sqrt() + curr_eps

                # Compute natural gradient from momentum
                grad_nat = momentum / fim_base

                # Clip with rms
                rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat.div_(divisor)

                # Moving average of natural momentum
                momentum_slow.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # Update momentumized natural fim
                squared_fim_beta = (beta3**group["step"] - beta3) / (beta3**group["step"] - 1.0)
                momentum_slow_squared.mul_(squared_fim_beta).addcmul_(momentum_slow, momentum_slow, value=1 - squared_fim_beta)

                fim_slow_base = momentum_slow_squared.sqrt() + curr_eps

                # Compute natural gradient using both FIMs
                grad_nat_2 = grad / fim_base / fim_slow_base

                rms = grad_nat_2.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat_2.div_(divisor)

                # Weight decay
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = p_fp32.data / fim_base / fim_slow_base
                else:
                    grad_weights = p.data / fim_base / fim_slow_base

                rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_weights.div_(divisor)

                # Differential amplification
                diff_weights = ema_diff / fim_base / fim_slow_base if diff_amp else 0
                if diff_amp:
                    rms = diff_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    diff_weights.div_(divisor)
                
                full_step = grad_nat_2 + (weight_decay * grad_weights) - (diff_amp * diff_weights)

                # Centralize the gradient vector
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                    )

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.add_(full_step, alpha=-lr / bias_correction)
                else:
                    p.data.add_(full_step, alpha=-lr / bias_correction)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["momentum_slow"], momentum_slow)
                    copy_stochastic_(state["momentum_slow_squared"], momentum_slow_squared)
                    copy_stochastic_(state["fim"], fim)
                    if diff_amp:
                        copy_stochastic_(state["ema_diff"], ema_diff)
                    copy_stochastic_(p, p_fp32)

        return loss

# FishMonger from https://github.com/Clybius/Personalized-Optimizers/blob/main/FishMonger.py by Clybius
class FishMonger8BitBNB(Optimizer):
    r"""
    FishMonger8BitBNB: Screw it, fisher everything.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.001)
        betas (Tuple[float, float, float], optional):
            coefficients used for computing running averages of
            fim, momentum, and its square (default: (0.9, 0.99, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
        diff_amp (float):
            Accelerate the difference between the current and past gradient by this multiplicative value (default: 1.0).
        diff_amp_beta (float):
            Coefficient used for computing running average of the current and past gradients (default: 0.999).
        quantization_group_size (int):
            number of quant group (default: 64).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        clip=1.0,
        centralization=1.0,
        diff_amp=1.0,
        diff_amp_beta=0.999,
        quantization_group_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
            diff_amp=diff_amp,
            diff_amp_beta=diff_amp_beta,
            group_size=quantization_group_size,
        )
        super(FishMonger8BitBNB, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FishMonger8BitBNB'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FishMonger8BitBNB does not support sparse gradients")

                state = self.state[p]

                # State initialization
                diff_amp = group["diff_amp"]
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )
                    state["momentum_slow"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )                    
                    state["momentum_slow_squared"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )                    
                    # Fisher Information Matrix
                    state["fim"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )     
                    # Previous grad
                    if diff_amp:
                        state["ema_diff"] = quantize_blockwise(
                            torch.zeros_like(p.data),
                            blocksize=group["group_size"],
                        )
                        state["previous_grad"] = quantize_blockwise(
                            grad.data.clone().mul_(-1.0),
                            blocksize=group["group_size"],
                        )    

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.to(torch.float32)
                    grad = grad.to(torch.float32).data
                else:
                    grad = grad.data

                momentum = dequantize_blockwise(*state["momentum"])
                momentum_slow = dequantize_blockwise(*state["momentum_slow"])
                momentum_slow_squared = dequantize_blockwise(*state["momentum_slow_squared"])
                fim = dequantize_blockwise(*state["fim"])
                ema_diff = dequantize_blockwise(*state["ema_diff"]) if diff_amp else 0

                beta1, beta2, beta3 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                clip = 1.0 if (group["clip"] <= 0) else group["clip"]
                centralization = group["centralization"]
                
                if diff_amp:
                    grad_diff = dequantize_blockwise(*state["previous_grad"])
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff.add_(grad)

                    # Smooth the difference between previous grad and current grad
                    ema_diff.mul_(group["diff_amp_beta"]).add_(grad_diff, alpha=1 - group["diff_amp_beta"])

                    state["previous_grad"] = quantize_blockwise(
                        -grad,
                        blocksize=group["group_size"],
                    )

                # Momentumize the gradient
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)

                # bias correction step size
                # soft warmup
                fim_beta = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0) # Start at 0 beta (I believe this is from Adafactor?)
                bias_correction = 1 - beta1**group["step"]

                # Update fim
                fim.mul_(fim_beta).addcmul_(momentum, momentum, value=1 - fim_beta)

                curr_eps = adaptive_eps(grad, group)

                fim_base = fim.sqrt() + curr_eps

                # Compute natural gradient from momentum
                grad_nat = momentum / fim_base

                # Clip with rms
                rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat.div_(divisor)

                # Moving average of natural momentum
                momentum_slow.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # Update momentumized natural fim
                squared_fim_beta = (beta3**group["step"] - beta3) / (beta3**group["step"] - 1.0)
                momentum_slow_squared.mul_(squared_fim_beta).addcmul_(momentum_slow, momentum_slow, value=1 - squared_fim_beta)

                fim_slow_base = momentum_slow_squared.sqrt() + curr_eps

                # Compute natural gradient using both FIMs
                grad_nat_2 = grad / fim_base / fim_slow_base

                rms = grad_nat_2.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat_2.div_(divisor)

                # Weight decay
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = p_fp32.data / fim_base / fim_slow_base
                else:
                    grad_weights = p.data / fim_base / fim_slow_base

                rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_weights.div_(divisor)

                # Differential amplification
                diff_weights = ema_diff / fim_base / fim_slow_base if diff_amp else 0
                if diff_amp:
                    rms = diff_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    diff_weights.div_(divisor)
                
                full_step = grad_nat_2 + (weight_decay * grad_weights) - (diff_amp * diff_weights)

                # Centralize the gradient vector
                if centralization > 0.0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                    )

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.add_(full_step, alpha=-lr / bias_correction)
                else:
                    p.data.add_(full_step, alpha=-lr / bias_correction)

                # pack
                state["momentum"] = quantize_blockwise(
                    momentum,
                    blocksize=group["group_size"],
                )
                state["momentum_slow"] = quantize_blockwise(
                    momentum_slow,
                    blocksize=group["group_size"],
                )                    
                state["momentum_slow_squared"] = quantize_blockwise(
                    momentum_slow_squared,
                    blocksize=group["group_size"],
                )                    
                # Fisher Information Matrix
                state["fim"] = quantize_blockwise(
                    fim,
                    blocksize=group["group_size"],
                )     
                # Previous grad
                if diff_amp:
                    state["ema_diff"] = quantize_blockwise(
                        ema_diff,
                        blocksize=group["group_size"],
                    )

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss