# Based on: "Anytime Training with Schedule-Free Spectral Optimization"
# (Apte, Deshpande, Kumar, Chakrabarti, Kim — JPMorganChase, May 2026)
# arXiv:2605.23061v1
#
# Implements Algorithm 1 (SF-NorMuon) from the paper with the memory-efficient
# single-z-buffer design from Appendix F's reference PyTorch implementation.
#
# Key design choices from the paper:
#   - Polar decomposition via 5-step Newton-Schulz (coefficients from Appendix F)
#   - Explicit momentum buffer (μ) smooths gradient before polar factor (§2.1)
#   - Row-wise adaptive normalization for per-neuron step sizes (§2.2)
#   - Weight decay at fast iterate Z (NOT at Y) for long-horizon stability (§3)
#   - 1D parameters use SF-AdamW fallback (§4, page 10)
#
# Original schedule-free framework: Defazio et al. "The Road Less Scheduled" (2024)
# NorMuon: Li et al. "NorMuon: Making Muon more efficient and scalable" (2025)
# Muon: Jordan et al. "Muon: An optimizer for hidden layers" (2024)

from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any, List
from typing_extensions import TypeAlias
import torch
import torch.optim
import math
import logging

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

from .utils import copy_stochastic_


@torch.no_grad()
def _zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    r"""Approximate the polar factor of G using 5th-order Newton-Schulz iteration.

    Computes an approximation to :math:`P = \text{polar}(G) = UV^\top` where
    :math:`G = U \Sigma V^\top` is the SVD.  The iteration uses the quintic
    polynomial coefficients :math:`(a, b, c) = (3.4445, -4.7750, 2.0315)` that
    maximize the slope at zero for rapid convergence.

    For tall matrices (rows > cols), the iteration is applied on the transpose
    to reduce FLOPs, then transposed back.

    :param G: 2-D input tensor (momentum buffer).
    :param steps: Number of Newton-Schulz iterations (default 5).
    :param eps: Small constant for numerical stability in normalization.
    :param compute_dtype: dtype used for the internal Newton-Schulz iteration
        (default ``torch.bfloat16``).  Pass ``torch.float32`` when the
        originating parameter is fp16, since bf16 hardware support cannot be
        assumed in that case.
    :returns: Approximate polar factor, same shape and dtype as G.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.to(dtype=compute_dtype)
    X /= (X.norm() + eps)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B

    if transposed:
        X = X.T

    return X.to(G.dtype)


class NorMuonScheduleFree(torch.optim.Optimizer):
    r"""Schedule-Free NorMuon (SF-NorMuon) optimizer.

    A schedule-free spectral optimizer that applies polar decomposition
    (steepest descent under the spectral norm) combined with row-wise
    adaptive normalization, explicit momentum, and schedule-free weight
    averaging.

    This optimizer requires that ``.train()`` and ``.eval()`` be called before
    training and evaluation respectively. The optimizer should also be placed
    in eval mode when saving checkpoints.

    **Parameter handling:**

    * **2-D parameters** (weight matrices in attention/MLP layers): Use the
      full SF-NorMuon spectral update with Newton-Schulz polar decomposition,
      explicit momentum, and row-wise adaptive normalization.

    * **Non-2-D parameters** (embeddings, biases, layer norms, etc.): Fall back
      to a schedule-free AdamW update with weight decay at Z.

    **Memory footprint** (per matrix parameter of shape m × n):

    * State: z (m×n) + mom (m×n) + v (m) + s (scalar) ≈ 2mn + m
    * Compared to AdamW: z (m×n) + exp_avg (m×n) + exp_avg_sq (m×n) = 3mn
    * The parameter buffer p serves as Y during training and X during evaluation.

    :param params: Iterable of parameters to optimize or dicts defining
        parameter groups.
    :param lr: Base learning rate (default: 0.008). Can be shared with SF-AdamW
        thanks to the NorMuon RMS scaling factor.
    :param betas: Tuple of (β₁, β₂) where β₁ is the schedule-free interpolation
        parameter and β₂ is the row-wise second moment EMA coefficient
        (default: (0.9, 0.95)).
    :param momentum: Explicit momentum coefficient μ for smoothing the gradient
        before computing the polar factor (default: 0.8). Critical for stable
        spectral updates — ablating to 0 significantly degrades performance.
    :param eps: Term added to the denominator for numerical stability
        (default: 1e-8).
    :param weight_decay: Decoupled weight decay coefficient λ. Applied at the
        fast iterate Z (NOT at Y), which is essential for long-horizon training
        stability (default: 0.05).
    :param warmup_steps: Number of steps for linear learning rate warmup
        (default: 0).
    :param eta_scale: Scaling factor for the NorMuon learning rate adjustment:
        :math:`\hat{\eta} = 0.2 \cdot \eta \cdot \sqrt{mn} / \|\hat{P}\|_F`
        (default: 0.2). This normalizes the effective step size to be comparable
        to Adam's RMS-normalized updates.
    :param ns_steps: Number of Newton-Schulz iterations for the polar
        decomposition approximation (default: 5).

    References:
        - Apte et al. "Anytime Training with Schedule-Free Spectral
          Optimization" (2026). arXiv:2605.23061v1.
        - Defazio et al. "The Road Less Scheduled" (2024).
        - Li et al. "NorMuon: Making Muon more efficient and scalable" (2025).
        - Jordan et al. "Muon: An optimizer for hidden layers" (2024).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        momentum: float = 0.8,
        eps: float = 1e-8,
        weight_decay: float = 0.05,
        warmup_steps: int = 0,
        eta_scale: float = 0.2,
        ns_steps: int = 5,
        **kwargs,
    ):
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            momentum=momentum,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eta_scale=eta_scale,
            ns_steps=ns_steps,
            k=0,
            train_mode=False,
            weight_sum=0.0,
            s_sum=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        """Reset all optimizer state (useful for re-initialization)."""
        for group in self.param_groups:
            group["k"] = 0
            group["weight_sum"] = 0.0
            group["s_sum"] = 0.0
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    state["z"].copy_(p)
                    if p.ndim == 2:
                        state["v"].zero_()
                        state["mom"].zero_()
                    else:
                        state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def eval(self):
        r"""Switch to evaluation mode: set p to the averaged iterate X_t.

        Computes :math:`X_t = (Y_t - (1 - \beta_1) Z_t) / \beta_1` on the
        fly from the stored Z_t and the live weights Y_t (=p).

        When :math:`\beta_1 = 0`, all iterates coincide so p is simply set
        to Z_t.
        """
        for group in self.param_groups:
            train_mode = group["train_mode"]
            if train_mode:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        z = state["z"]
                        if p.dtype in (torch.bfloat16, torch.float16):
                            z_fp32 = z.to(device=p.device, dtype=torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)
                            if beta1 > 0:
                                p_fp32.lerp_(end=z_fp32, weight=1.0 - 1.0 / beta1)
                            else:
                                p_fp32.copy_(z_fp32)
                            copy_stochastic_(p.data, p_fp32)
                        else:
                            z_dev = z.to(device=p.device)
                            if beta1 > 0:
                                p.lerp_(end=z_dev, weight=1.0 - 1.0 / beta1)
                            else:
                                p.copy_(z_dev)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        r"""Switch to training mode: set p to the gradient-evaluation point Y_t.

        Computes :math:`Y_t = (1 - \beta_1) Z_t + \beta_1 X_t` which, given
        that the previous eval call set p to X_t, is simply a lerp from X_t
        toward Z_t.
        """
        for group in self.param_groups:
            train_mode = group["train_mode"]
            if not train_mode:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        z = state["z"]
                        if p.dtype in (torch.bfloat16, torch.float16):
                            z_fp32 = z.to(device=p.device, dtype=torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)
                            p_fp32.lerp_(end=z_fp32, weight=1.0 - beta1)
                            copy_stochastic_(p.data, p_fp32)
                        else:
                            z_dev = z.to(device=p.device)
                            p.lerp_(end=z_dev, weight=1.0 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Perform a single SF-NorMuon optimization step.

        Follows Algorithm 1 from the paper. The gradient must have been
        computed at the current parameter values (which represent Y_t in
        training mode).

        :param closure: Optional closure that reevaluates the model and
            returns the loss.
        :returns: The loss value if a closure was provided.
        """
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = closure() if closure else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            mu = group["momentum"]
            eps = group["eps"]
            eta_scale = group["eta_scale"]
            decay = group["weight_decay"]
            k = group["k"]
            warmup_steps = group["warmup_steps"]
            ns_steps = group["ns_steps"]
            step_num = k + 1

            # Learning rate with linear warmup (Algorithm 1, line 12)
            if warmup_steps > 0:
                sched = min(1.0, step_num / warmup_steps)
            else:
                sched = 1.0
            lr = group["lr"] * sched

            # Schedule-free averaging coefficient (Algorithm 1, lines 14-15)
            # s_t = s_{t-1} + η_t²,  c_{t+1} = η_t² / s_t
            weight = lr * lr
            s_sum = group["s_sum"] = group["s_sum"] + weight
            ckp1 = weight / s_sum if s_sum > 0 else 0

            # Process each parameter
            active_params = [p for p in group["params"] if p.grad is not None]

            for p in active_params:
                grad = p.grad
                state = self.state[p]

                if p.ndim == 2:
                    self._step_2d(p, grad, state, group, lr, beta1, beta2,
                                  mu, eps, eta_scale, decay, ckp1, ns_steps)
                else:
                    self._step_1d(p, grad, state, group, lr, beta1, beta2,
                                  eps, decay, ckp1)

            group["k"] = step_num

        return loss

    def _step_2d(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        mu: float,
        eps: float,
        eta_scale: float,
        decay: float,
        ckp1: float,
        ns_steps: int,
    ):
        """SF-NorMuon spectral update for 2-D weight matrix parameters.

        Implements the full Algorithm 1 path:
        momentum → polar decomposition → row-wise normalization →
        NorMuon LR scaling → weight decay at Z → polar update → SF average.
        """
        # Initialize state on first step
        if "z" not in state:
            state["z"] = torch.clone(p.detach(), memory_format=torch.preserve_format)
            state["v"] = torch.zeros(
                p.shape[0], device=p.device, dtype=torch.float32
            )
            state["mom"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        z = state["z"]
        v = state["v"]       # row-wise second moment (m,)
        mom = state["mom"]    # explicit momentum buffer (m × n)

        # Work in fp32 for numerical stability
        p_fp32 = p.to(dtype=torch.float32)
        grad_fp32 = grad.to(dtype=torch.float32)
        z_fp32 = z.to(device=p.device, dtype=torch.float32)
        mom_fp32 = mom.to(device=p.device, dtype=torch.float32)

        # Algorithm 1, line 5: explicit momentum buffer
        # M_t = μ · M_{t-1} + (1 - μ) · G_t
        mom_fp32.mul_(mu).add_(grad_fp32, alpha=1.0 - mu)

        # Algorithm 1, line 7: polar factor via Newton-Schulz
        # P_t = polar(M_t)
        # Use fp32 for Newton-Schulz when param is fp16 (bf16 assumed unsupported)
        ns_dtype = torch.float32 if p.dtype == torch.float16 else torch.bfloat16
        P = _zeropower_via_newtonschulz5(mom_fp32, steps=ns_steps, compute_dtype=ns_dtype)

        # Algorithm 1, line 8: row-wise second moment EMA
        # v_t = β₂ · v_{t-1} + (1 - β₂) · mean_cols(P_t ⊙ P_t)
        row_ms = (P * P).mean(dim=1).float()  # (m,)
        v.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)

        # Algorithm 1, line 10: adaptive row-wise normalization
        # P̂_t = P_t / (√V_t + ε)
        denom = v.sqrt().add_(eps).to(P.dtype)  # (m,)
        Phat = P / denom.unsqueeze(1)

        # Algorithm 1, line 13: NorMuon learning rate scaling
        # η̂_t = 0.2 · η_t · √(mn) / ‖P̂_t‖_F
        m, n = p.shape
        Phat_norm = Phat.float().norm()
        eta_hat = eta_scale * lr * math.sqrt(m * n) / max(1e-12, Phat_norm)

        # Recover X_t from Y_t and Z_t:
        # Y_t = (1-β₁)Z_t + β₁X_t  =>  X_t = (Y_t - (1-β₁)Z_t) / β₁
        if beta1 > 0:
            x_t = (p_fp32 - (1.0 - beta1) * z_fp32) / beta1
        else:
            # When β₁=0, Y_t = Z_t = X_t
            x_t = z_fp32.clone()

        # Algorithm 1, line 17: update fast iterate
        # Z_{t+1} = Z_t - η·λ·Z_t - η̂_t·P̂_t
        # Weight decay is applied at Z (NOT at Y), critical for stability (§3)
        if decay != 0:
            z_fp32.sub_(z_fp32, alpha=lr * decay)
        z_fp32.sub_(Phat.float(), alpha=eta_hat)

        # Algorithm 1, line 18: schedule-free average
        # X_{t+1} = (1 - c_{t+1}) · X_t + c_{t+1} · Z_{t+1}
        x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z_fp32

        # Set Y_{t+1} = (1-β₁)Z_{t+1} + β₁X_{t+1} for next step
        p_new = (1.0 - beta1) * z_fp32 + beta1 * x_tp1

        # Write back with stochastic rounding for bf16/fp16
        if p.dtype in (torch.bfloat16, torch.float16):
            copy_stochastic_(p.data, p_new)
            copy_stochastic_(z, z_fp32)
            copy_stochastic_(mom, mom_fp32)
        else:
            p.data.copy_(p_new)
            z.copy_(z_fp32)
            mom.copy_(mom_fp32)

    def _step_1d(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        decay: float,
        ckp1: float,
    ):
        """Schedule-Free AdamW update for non-matrix (1-D) parameters.

        Embeddings, biases, layer norms, and other non-matrix parameters use
        standard Adam-style second-moment normalization with schedule-free
        averaging and weight decay at Z, consistent with the paper's setup.
        """
        # Initialize state on first step
        if "z" not in state:
            state["z"] = torch.clone(p.detach(), memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        z = state["z"]
        exp_avg_sq = state["exp_avg_sq"]

        # Work in fp32
        p_fp32 = p.to(dtype=torch.float32)
        grad_fp32 = grad.to(dtype=torch.float32)
        z_fp32 = z.to(device=p.device, dtype=torch.float32)
        exp_avg_sq_fp32 = exp_avg_sq.to(device=p.device, dtype=torch.float32)

        # Update second moment (Adam-style)
        # v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
        exp_avg_sq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
        denom = exp_avg_sq_fp32.sqrt().add_(eps)

        # Normalized gradient
        grad_normalized = grad_fp32 / denom

        # Recover X_t
        if beta1 > 0:
            x_t = (p_fp32 - (1.0 - beta1) * z_fp32) / beta1
        else:
            x_t = z_fp32.clone()

        # Weight decay at Z
        if decay != 0:
            z_fp32.sub_(z_fp32, alpha=lr * decay)

        # Adam step on fast iterate
        z_fp32.sub_(grad_normalized, alpha=lr)

        # Schedule-free average
        x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z_fp32

        # Set Y_{t+1}
        p_new = (1.0 - beta1) * z_fp32 + beta1 * x_tp1

        # Write back with stochastic rounding for bf16/fp16
        if p.dtype in (torch.bfloat16, torch.float16):
            copy_stochastic_(p.data, p_new)
            copy_stochastic_(z, z_fp32)
            copy_stochastic_(exp_avg_sq, exp_avg_sq_fp32)
        else:
            p.data.copy_(p_new)
            z.copy_(z_fp32)
            exp_avg_sq.copy_(exp_avg_sq_fp32)
