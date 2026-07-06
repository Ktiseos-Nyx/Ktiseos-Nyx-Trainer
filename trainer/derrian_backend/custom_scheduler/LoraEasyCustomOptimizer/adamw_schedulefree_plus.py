# Derived from https://github.com/facebookresearch/schedule_free/blob/43e5c2d978920e1a3707a7609662c72b56401e62/schedulefree/adamc_schedulefree_plus_paper.py 
# Apache License 
# Version 2.0, January 2004 
# http://www.apache.org/licenses/

from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math
import logging
from .utils import copy_stochastic_


class AdamWScheduleFreePlus(torch.optim.Optimizer):
    r"""
    ScheduleFree+ AdamW (AdamC + Schedule-Free + Polyak Step Size)

    This optimizer implements the full ScheduleFree+ method from:
    "ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to
    Large Language Models" (Defazio, 2026).

    It combines three key ideas:

    1. **Schedule-Free Learning** with inner momentum (Adam's β₁) for
       large-batch robustness, optional β_sf annealing for long training runs,
       c_warmup for improved early convergence, and configurable polynomial
       weighting (r parameter).

    2. **Polyak Step Size** (learning-rate-free): The per-step learning rate
       is set adaptively using a Polyak-style rule based on the function value
       and an EMA of the gradient L1 norm. This eliminates the need to tune
       the learning rate. The Polyak step also automatically scales with
       batch size.

    3. **AdamC Fully-Decoupled Weight Decay**: Weight decay is applied at
       the fast iterate z by default (decay_at_z), which is provably stable
       for all training horizons (Lemma 3.1, Apte et al. 2026). Set
       ``weight_decay_at_y=True`` to use the original decay-at-y behavior.
       Decay is scaled by ``lr_effective`` (AdamC style) so that the
       effective decay strength is independent of the adaptive learning rate.
       Typical values are 5-50 for Polyak-based training (default 0).

    This optimizer requires that ``.train()`` and ``.eval()`` be called before
    training and evaluation respectively. The optimizer should also be placed
    in eval mode when saving checkpoints.

    The step interface uses ``step_func(function_value)`` instead of the
    standard ``step(closure)``, because the Polyak step size requires the
    current loss value.

    **Memory-efficient design**: This implementation uses a single ``z`` state
    (like ``adamw_schedulefree.py``) rather than storing explicit ``x`` and
    ``y`` tensors.  During training the parameter ``p`` *is* ``y``; during
    evaluation ``p`` is switched to ``x`` via a lerp from ``z``.  This saves
    two full-sized parameter tensors per parameter (~40 % optimizer memory
    reduction) while producing identical optimisation updates.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Base learning rate multiplier (default 1.0). The effective step
            size is ``lr * polyak_lr`` where ``polyak_lr`` is computed
            adaptively from the function value.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.95)).
        sf_beta1 (float):
            Schedule-Free outer momentum used for the ``x -> y``
            extrapolation (default 0.9). This is distinct from Adam's
            first-moment beta1.
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability (default 1e-8).
        weight_decay (float):
            Decoupled weight decay coefficient. Applied at the fast iterate z
            by default (``weight_decay_at_y=False``), scaled by the effective
            learning rate (AdamC style). Typical values are 5-50 for
            Polyak-based training (default 0).
        weight_decay_at_y (bool):
            If True, applies weight decay at y (the gradient evaluation
            point) — the original behavior from Defazio et al. (2024).
            If False (default), applies weight decay at z (the fast iterate),
            which is provably stable for all training horizons (Lemma 3.1,
            Apte et al. 2026). Decay at y can lead to divergence for
            long-horizon training with β > 0.
        r (float):
            Polynomial weighting power for the Schedule-Free average.
            ``r=1`` is recommended for long-duration training runs
            (default 1).
        weight_lr_power (float):
            During warmup, the weights in the average will be equal to lr
            raised to this power. Set to 0 for no weighting
            (default 2.0).
        polyak_beta (float):
            EMA decay for the running estimate of the gradient L1 norm used
            in the Polyak step size. 0 means no smoothing (default 0.9).
        polyak_f_ema (float):
            EMA coefficient for smoothing the stochastic function value in
            the Polyak step-size numerator. The paper (Section 6)
            recommends applying EMA to stochastic estimates for stability.
            A value of 0.95 provides strong smoothing; set to 0 to disable
            and use the raw function value (not recommended for small batch
            sizes). (default 0.95).
        max_polyak_lr (float):
            Upper bound on the Polyak step-size scalar ``polyak_lr`` to
            prevent blow-up when the gradient L1 norm is small (e.g. with
            LoRA-scale training or small batch sizes). Set to 0 or
            ``math.inf`` to disable the cap (default 10.0).
        c_warmup (int):
            Number of initial steps during which the averaging weight
            ``ckp1`` is forced to 1.0 (i.e. ``x`` tracks ``z`` exactly).
            Recommended: 2x the lr warmup duration (default 0).
        warmup_steps (int):
            Enables a linear learning rate warmup (default 0).
        sf_beta1_anneal_steps (int):
            If greater than zero, ``sf_beta1`` is annealed from ``sf_beta1``
            to ``sf_beta1_max`` over this many steps using log-linear
            interpolation. Set to 0 to disable (default 0).
        sf_beta1_max (float):
            Target value for ``sf_beta1`` at the end of the annealing
            schedule (default 0.965).

    State maintained per parameter:
        z (Tensor): the unaveraged AdamW iterate.
        exp_avg (Tensor): AdamW first-moment buffer.
        exp_avg_sq (Tensor): AdamW second-moment buffer.

    The parameter buffer ``p`` serves as ``y`` during training and ``x``
    during evaluation.  The ``.train()`` / ``.eval()`` methods switch ``p``
    between them via a lerp from ``z``.
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 sf_beta1: float = 0.9,
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 weight_decay_at_y: bool = False,
                 r: float = 1.0,
                 weight_lr_power: float = 2.0,
                 polyak_beta: float = 0.9,
                 polyak_f_ema: float = 0.95,
                 max_polyak_lr: float = 10.0,
                 c_warmup: int = 0,
                 warmup_steps: int = 0,
                 sf_beta1_anneal_steps: int = 0,
                 sf_beta1_max: float = 0.965,
                 **kwargs,):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        defaults = dict(lr=lr,
                        betas=betas,
                        sf_beta1=sf_beta1,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=eps,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        weight_decay_at_y=weight_decay_at_y,
                        polyak_beta=polyak_beta,
                        polyak_f_ema=polyak_f_ema,
                        f_ema=None,  # initialized lazily on first step
                        max_polyak_lr=max_polyak_lr,
                        grad_l1_ema=0.0,
                        c_warmup=c_warmup,
                        sf_beta1_anneal_steps=sf_beta1_anneal_steps,
                        sf_beta1_max=sf_beta1_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                sf_beta1_k = group.get('sf_beta1_k', group['sf_beta1'])
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x = lerp(y, z, 1-1/sf_beta1_k)
                        if sf_beta1_k > 0:
                            p.lerp_(end=state['z'].to(p.device),
                                    weight=1 - 1 / sf_beta1_k)
                        else:
                            # When sf_beta1_k = 0, y = z and x is undefined;
                            # just set p = z since all iterates coincide.
                            p.copy_(state['z'].to(p.device))
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                sf_beta1_k = group.get('sf_beta1_k', group['sf_beta1'])
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y = lerp(x, z, 1-sf_beta1_k)
                        p.lerp_(end=state['z'].to(p.device),
                                weight=1 - sf_beta1_k)
                group['train_mode'] = True

    @torch.no_grad()
    def step_func(self, function_value: float) -> float:
        """Performs a single optimization step using the Polyak step size.

        This method takes the current loss function value (required for the
        Polyak step-size rule) and performs the full ScheduleFree+ update:
        Polyak-based adaptive learning rate, AdamC weight decay (at z or y), inner
        momentum, Schedule-Free averaging with optional β annealing, and
        c_warmup.

        Arguments:
            function_value (float): The current value of the loss function
                at the training point y. Required for computing the Polyak
                step size numerator.

        Returns:
            The function_value that was passed in, for convenience.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        grad_l1_ema = self.param_groups[0]['grad_l1_ema']
        polyak_beta = self.param_groups[0]['polyak_beta']
        k = self.param_groups[0]['k']

        # ---- Schedule-Free β annealing ----
        sf_beta1 = self.param_groups[0]['sf_beta1']
        sf_beta1_max = self.param_groups[0]['sf_beta1_max']
        sf_beta1_anneal_steps = self.param_groups[0]['sf_beta1_anneal_steps']

        if sf_beta1_anneal_steps > 0:
            progress = min(k / sf_beta1_anneal_steps, 1.0)
            sf_beta1_k = 1 - math.exp(
                math.log(1 - sf_beta1) * (1 - progress) +
                math.log(1 - sf_beta1_max) * progress)
        else:
            sf_beta1_k = sf_beta1

        # ---- Collect gradient L1 norms and inner product correction ----
        grad_l1_list = []
        ip_term_list = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                grad_l1_list.append(torch.linalg.vector_norm(grad, ord=1))
                if 'z' in state:
                    # ip_term = sf_beta1_k * <grad, z - x>
                    # Using identity  z - x = (z - y) / sf_beta1_k(prev),
                    # and p = y in train mode.
                    ip_term_list.append((grad * (state['z'] - p)).sum())

        # ---- Compute global Polyak step size ----
        local_grad_l1 = torch.stack(grad_l1_list).sum() if grad_l1_list else torch.tensor(0.0)
        local_ip_term = torch.stack(ip_term_list).sum() if ip_term_list else torch.tensor(0.0)

        grad_l1 = local_grad_l1.item()
        ip_term = local_ip_term.item()

        # Update gradient L1 norm EMA with bias correction
        grad_l1_ema = polyak_beta * grad_l1_ema + (1 - polyak_beta) * grad_l1 * math.sqrt(math.pi / 2)
        grad_l1_ema_corr = grad_l1_ema / (1 - polyak_beta ** (k + 1))

        # ---- Function value EMA for stochastic stability ----
        # The paper (Section 6) recommends applying EMA to stochastic
        # function-value estimates, critical for small-batch training.
        f_ema_coeff = self.param_groups[0]['polyak_f_ema']
        f_ema = self.param_groups[0]['f_ema']
        if f_ema is None:
            # First step: initialize EMA to the raw value
            f_ema = function_value
        elif f_ema_coeff > 0:
            f_ema = f_ema_coeff * f_ema + (1 - f_ema_coeff) * function_value
        # else: f_ema_coeff == 0 → use raw function_value
        self.param_groups[0]['f_ema'] = f_ema

        polyak_lr = max(0, f_ema + ip_term) / max(grad_l1_ema_corr, 1e-12)

        # ---- Polyak LR cap to prevent blow-up ----
        _max_polyak = self.param_groups[0]['max_polyak_lr']
        if _max_polyak > 0:
            polyak_lr = min(polyak_lr, _max_polyak)

        # ---- Per-group parameter updates ----
        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            c_warmup = group['c_warmup']

            bias_correction1 = 1 - beta1 ** (k + 1)
            bias_correction2 = 1 - beta2 ** (k + 1)

            # Linear warmup applied to the base learning rate
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            group_lr = lr * sched

            # Effective learning rate: base_lr * polyak_lr
            alpha = group_lr * polyak_lr

            # For logging purposes
            group['grad_l1_ema'] = grad_l1_ema
            group['grad_l1_ema_corr'] = grad_l1_ema_corr
            group['function_value_raw'] = function_value
            group['function_value_ema'] = f_ema
            group['function_value_with_correction'] = f_ema + ip_term
            group['ip_term'] = ip_term
            group['polyak_lr'] = polyak_lr
            group['scheduled_lr'] = alpha
            group['sf_beta1_k'] = sf_beta1_k

            lr_max = group['lr_max'] = max(alpha, group['lr_max'])

            # ---- Schedule-Free averaging weight ----
            if k < c_warmup:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight
                try:
                    ckp1 = weight / weight_sum
                except ZeroDivisionError:
                    ckp1 = 0

            # ---- Per-parameter AdamC + Schedule-Free updates ----
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if 'z' not in state:
                    state['z'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Cast to fp32 for numerically stable computation
                # (same pattern as came.py: compute in fp32, write back with stochastic rounding)
                p_fp32 = p.to(dtype=torch.float32)
                grad_fp32 = grad.to(dtype=torch.float32)
                z_fp32 = state['z'].to(dtype=torch.float32)
                exp_avg_fp32 = state['exp_avg'].to(dtype=torch.float32)
                exp_avg_sq_fp32 = state['exp_avg_sq'].to(dtype=torch.float32)

                # Update first moment (Adam inner momentum)
                exp_avg_fp32.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                exp_avg_corr = exp_avg_fp32.div(bias_correction1)

                # Update second moment
                exp_avg_sq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)
                denom = exp_avg_sq_fp32.div(bias_correction2).sqrt_().add_(eps)

                # Adam-preconditioned gradient
                grad_normalized = exp_avg_corr.div(denom)

                # Schedule-Free interpolation weight
                A = 1 - sf_beta1_k * (1 - ckp1)

                if group.get('weight_decay_at_y', False):
                    # ---- Original behavior: decay at y ----
                    # Adds AdamC weight decay using the gradient-evaluation
                    # point y (p = y in train mode).  Both y and z receive
                    # the decay-modified combined gradient.
                    # g_combined = g + alpha * decay * y
                    if decay != 0:
                        grad_normalized.add_(p_fp32,
                                             alpha=alpha * decay)  # p = y in train mode
                    p_fp32.lerp_(end=z_fp32, weight=ckp1)
                    p_fp32.add_(grad_normalized, alpha=-A * alpha)
                    z_fp32.sub_(grad_normalized, alpha=alpha)
                else:
                    # ---- New behavior (default): decay at z ----
                    # Weight decay is applied directly to the fast iterate z
                    # BEFORE the gradient step, yielding a geometric
                    # contraction that provably bounds all iterates
                    # (Lemma 3.1, Apte et al. 2026).
                    # z_new = (1 - alpha*lambda) * z  -  alpha * g
                    # y_new = (1-c)*y + c*z  -  A*alpha*g   (pure gradient, no decay)
                    p_fp32.lerp_(end=z_fp32, weight=ckp1)
                    p_fp32.add_(grad_normalized, alpha=-A * alpha)
                    if decay != 0:
                        z_fp32.mul_(1.0 - alpha * decay)
                    z_fp32.sub_(grad_normalized, alpha=alpha)

                # Write back parameter and state with stochastic rounding for bf16
                if p.dtype == torch.bfloat16:
                    copy_stochastic_(p.data, p_fp32)
                    copy_stochastic_(state['z'], z_fp32)
                    copy_stochastic_(state['exp_avg'], exp_avg_fp32)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq_fp32)
                else:
                    p.data.copy_(p_fp32)
                    state['z'].copy_(z_fp32)
                    state['exp_avg'].copy_(exp_avg_fp32)
                    state['exp_avg_sq'].copy_(exp_avg_sq_fp32)
               

            group['k'] = k + 1

        # Persist EMA at group level for next step
        self.param_groups[0]['grad_l1_ema'] = grad_l1_ema
        return function_value

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Raises an error. Use ``step_func(function_value)`` instead.

        The Polyak step size requires the function value to be passed in
        explicitly. Use ``step_func(loss.item())`` after computing the loss.
        """
        raise NotImplementedError(
            "AdamWScheduleFreePlus uses the Polyak step size and requires "
            "the function value. Use step_func(function_value) instead of "
            "step(closure). Example:\n"
            "    loss = model(input)\n"
            "    loss.backward()\n"
            "    optimizer.step_func(loss.item())")
