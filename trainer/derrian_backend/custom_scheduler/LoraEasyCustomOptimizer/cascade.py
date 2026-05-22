import torch
from torch.optim import Optimizer
from typing import Tuple, Iterable
import math

from .utils import copy_stochastic_


@torch.no_grad()
def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, eps=1e-16):
    """
    CAME-style factorized denominator computation.
    Combines row-wise and column-wise variance factors.
    Uses in-place operations for memory efficiency.
    """
    # Row factor with epsilon to prevent division by zero
    r_factor = exp_avg_sq_row + eps
    r_factor.sqrt_()  # In-place sqrt
    r_factor.unsqueeze_(-1)  # In-place unsqueeze to [m, 1]

    # Column factor with epsilon (normalized by mean for stability)
    c_factor = (
        (exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)
    ).unsqueeze(-2)  # Unsqueeze to [..., 1, n]
    c_factor.sqrt_()  # In-place sqrt

    # Combine with broadcasting support: [..., m, 1] * [..., 1, n] = [..., m, n]
    return torch.mul(r_factor, c_factor)


@torch.no_grad()
def _reconstruct_factorized(row_factor, col_factor):
    """
    Reconstruct a full tensor from row-wise and column-wise factors.
    Similar to _approx_sq_grad but without sqrt and epsilon.
    """
    r = row_factor.unsqueeze(-1)  # [..., m, 1]
    c = col_factor.unsqueeze(-2)  # [..., 1, n]
    return torch.mul(r, c)

def reshape_to_2d(grad):
    """Reshape a tensor to 2D for matrix operations."""
    dimcount = len(grad.shape)
    if dimcount > 2:
        grad_2d = grad.reshape(len(grad), -1)
    elif dimcount < 2:
        grad_2d = grad.reshape(1, -1)
    else:
        grad_2d = grad
    return grad_2d

GRAM_NEWTON_SCHULZ_2STEP_COEFFS1 = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]

@torch.no_grad()
def gram_newton_schulz_2step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    5-step Gram Newton-Schulz with pre-optimized unconstrained coefficients.

    Uses 5-step accumulated iteration with coefficients derived from pure
    optimization (no h(1)=1 or h'(1)=-0.5 constraints enforced).

    Coefficients optimized for spectral range [0.2, 1.8] with convergence
    target ||r_final - 1|| < 1e-4.

    Args:
        M: Input matrix [n, m] to orthogonalize
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality after odd iterations
        tolerance: Orthonormality error threshold for early stopping

    Returns:
        Orthonormal matrix [n, m]
    """
    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL-Gram folding
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s
    R = s * A * s.mT

    n, m = X.shape
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I.clone()

    # 5-step iteration with pre-optimized coefficients
    for step_idx, (a, b, c) in enumerate(GRAM_NEWTON_SCHULZ_2STEP_COEFFS1):
        # Cubic polynomial on Gram matrix
        R2 = R @ R
        z = a * I + b * R + c * R2

        # Accumulated updates
        Q = Q @ z
        R = z @ R @ z

    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)

class CASCADE(Optimizer):
    r"""
    CASCADE: Cascaded Adaptive Second-moment Conditioning with Adaptive Dual Estimation.

    A variant of Adam & CAME utilizing cascaded second-moment conditioning for
    memory-efficient adaptive optimization. Features three CAME-factorized momentums:
    - Factorized momentum (tracks gradient EMA)
    - First denominator (tracks deviation from factorized momentum)
    - Second denominator (tracks preconditioned gradient)

    Key features:
    1. Cascaded conditioning: gradient first conditioned by deviation from factorized
       momentum, then by its own magnitude
    2. CAME-style factorized memory for all second-moment estimates
    3. Polynomial beta scheduling for de-biased early training
    4. Compensated cautious masking for stable updates
    5. Stochastic rounding for BF16 precision

    The optimizer state consists of:
    - exp_avg: Standard momentum (init zeros, normal beta1)
    - exp_avg_fac_row/col: Factorized momentum (init zeros, polybeta1)
    - exp_avg_sq_row1/col1: First denominator (init ones, polybeta2)
    - exp_avg_sq_row2/col2: Second denominator (init ones, polybeta2)

    Update flow:
    grad → update first denom (grad - fac_momentum).pow(2) → update fac_momentum →
    precondition (grad / denom1.sqrt()) → update full momentum →
    update second denom → cautious masking → weight decay → apply update

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for momentum (beta1) and
            denominator (beta2) EMAs (default: (0.9, 0.999)).
        eps (float): Term added to denominator for numerical stability (default: 1e-8).
        weight_decay (float): Decoupled weight decay (default: 0.0).
        cautious (bool): Apply compensated sign-masking to updates (default: True).
        stochastic_fp (bool): Use stochastic rounding for BF16 tensors (default: True).

    Example:
        >>> optimizer = CASCADE(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious: bool = True,
        muon: bool = True,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if len(betas) != 2:
            raise ValueError(f"Invalid betas: expected 2 values, got {len(betas)}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")

        if muon:
            try:
                self.ortho_func = torch.compile(gram_newton_schulz_2step, dynamic=True, mode="default")
            except:
                self.ortho_func = gram_newton_schulz_2step

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            cautious=cautious,
            muon=muon,
            stochastic_fp=stochastic_fp,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            cautious = group["cautious"]
            muon = group["muon"]
            stochastic_fp = group["stochastic_fp"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CASCADE does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    # Standard momentum (init zeros, normal beta1)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Factorized momentum (init zeros, polybeta1)
                    if p.ndim >= 2:
                        state["exp_avg_fac_row"] = torch.zeros(
                            p.shape[:-1], device=p.device, dtype=p.dtype
                        )
                        col_shape = p.shape[:-2] + p.shape[-1:]
                        state["exp_avg_fac_col"] = torch.zeros(
                            col_shape, device=p.device, dtype=p.dtype
                        )
                    else:
                        state["exp_avg_fac"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    # First denominator (init ones, polybeta2) - deviation tracking
                    if p.ndim >= 2:
                        state["exp_avg_sq_row1"] = torch.ones(
                            p.shape[:-1], device=p.device, dtype=p.dtype
                        )
                        state["exp_avg_sq_col1"] = torch.ones(
                            col_shape, device=p.device, dtype=p.dtype
                        )
                    else:
                        state["exp_avg_sq1"] = torch.ones_like(
                            p, memory_format=torch.preserve_format
                        )

                    # Second denominator (init ones, polybeta2) - preconditioned grad tracking
                    if p.ndim >= 2:
                        state["exp_avg_sq_row2"] = torch.ones(
                            p.shape[:-1], device=p.device, dtype=p.dtype
                        )
                        state["exp_avg_sq_col2"] = torch.ones(
                            col_shape, device=p.device, dtype=p.dtype
                        )
                    else:
                        state["exp_avg_sq2"] = torch.ones_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]

                # Mixed precision handling
                use_stochastic = stochastic_fp and p.dtype == torch.bfloat16

                # Initialize work variables
                p_work = p.data
                grad_work = grad
                exp_avg_work = exp_avg

                if use_stochastic:
                    p_work = p_work.to(torch.float32)
                    grad_work = grad_work.to(torch.float32)
                    exp_avg_work = exp_avg_work.to(torch.float32)

                # Initialize factorized state work variables
                if p.ndim >= 2:
                    exp_avg_fac_row = state["exp_avg_fac_row"]
                    exp_avg_fac_col = state["exp_avg_fac_col"]
                    exp_avg_sq_row1 = state["exp_avg_sq_row1"]
                    exp_avg_sq_col1 = state["exp_avg_sq_col1"]
                    exp_avg_sq_row2 = state["exp_avg_sq_row2"]
                    exp_avg_sq_col2 = state["exp_avg_sq_col2"]

                    exp_avg_fac_row_work = exp_avg_fac_row
                    exp_avg_fac_col_work = exp_avg_fac_col
                    exp_avg_sq_row1_work = exp_avg_sq_row1
                    exp_avg_sq_col1_work = exp_avg_sq_col1
                    exp_avg_sq_row2_work = exp_avg_sq_row2
                    exp_avg_sq_col2_work = exp_avg_sq_col2

                    if use_stochastic:
                        exp_avg_fac_row_work = exp_avg_fac_row_work.to(torch.float32)
                        exp_avg_fac_col_work = exp_avg_fac_col_work.to(torch.float32)
                        exp_avg_sq_row1_work = exp_avg_sq_row1_work.to(torch.float32)
                        exp_avg_sq_col1_work = exp_avg_sq_col1_work.to(torch.float32)
                        exp_avg_sq_row2_work = exp_avg_sq_row2_work.to(torch.float32)
                        exp_avg_sq_col2_work = exp_avg_sq_col2_work.to(torch.float32)
                else:
                    exp_avg_fac = state["exp_avg_fac"]
                    exp_avg_sq1 = state["exp_avg_sq1"]
                    exp_avg_sq2 = state["exp_avg_sq2"]

                    exp_avg_fac_work = exp_avg_fac
                    exp_avg_sq1_work = exp_avg_sq1
                    exp_avg_sq2_work = exp_avg_sq2

                    if use_stochastic:
                        exp_avg_fac_work = exp_avg_fac_work.to(torch.float32)
                        exp_avg_sq1_work = exp_avg_sq1_work.to(torch.float32)
                        exp_avg_sq2_work = exp_avg_sq2_work.to(torch.float32)

                # Polynomial beta schedule (de-biased, from WiwiOpt)
                poly_beta1 = (beta1 ** (step) - beta1) / (beta1 ** (step) - 1.0)
                poly_beta2 = (beta2 ** (step) - beta2) / (beta2 ** (step) - 1.0)

                # === Cascaded Conditioning Update ===

                if p_work.ndim >= 2:
                    # Step 1: Update first denominator with (grad - fac_momentum)^2
                    # Reconstruct factorized momentum
                    fac_momentum_work = _reconstruct_factorized(
                        exp_avg_fac_row_work, exp_avg_fac_col_work
                    )

                    # Compute deviation from factorized momentum
                    grad_err = grad_work - fac_momentum_work
                    grad_err_sq = grad_err.pow(2)

                    # Update first denominator (row-wise and column-wise)
                    exp_avg_sq_row1_work.lerp_(
                        grad_err_sq.mean(dim=-1), weight=1.0 - poly_beta2
                    )
                    if grad_err_sq.ndim > 2:
                        exp_avg_sq_col1_work.lerp_(
                            grad_err_sq.mean(dim=-2), weight=1.0 - poly_beta2
                        )
                    else:
                        exp_avg_sq_col1_work.lerp_(
                            grad_err_sq.mean(dim=0), weight=1.0 - poly_beta2
                        )

                    # Compute first denominator
                    denom1 = _approx_sq_grad(
                        exp_avg_sq_row1_work, exp_avg_sq_col1_work, eps
                    )

                    # Step 2: Update factorized momentum (polybeta1)
                    exp_avg_fac_row_work.lerp_(
                        grad_work.mean(dim=-1), weight=1.0 - poly_beta1
                    )
                    if grad_work.ndim > 2:
                        exp_avg_fac_col_work.lerp_(
                            grad_work.mean(dim=-2), weight=1.0 - poly_beta1
                        )
                    else:
                        exp_avg_fac_col_work.lerp_(
                            grad_work.mean(dim=0), weight=1.0 - poly_beta1
                        )

                    # Step 3: Precondition gradient
                    precond_grad = grad_work / denom1.clamp_min_(eps)

                    if muon:
                        precond_grad_2d = reshape_to_2d(precond_grad)

                        precond_grad = self.ortho_func(precond_grad_2d, eps=eps).view_as(precond_grad)

                    # Step 4: Update second denominator with precond_grad^2
                    precond_grad_sq = (precond_grad - exp_avg_work).pow(2)
                    exp_avg_sq_row2_work.lerp_(
                        precond_grad_sq.mean(dim=-1), weight=1.0 - poly_beta2
                    )
                    if precond_grad_sq.ndim > 2:
                        exp_avg_sq_col2_work.lerp_(
                            precond_grad_sq.mean(dim=-2), weight=1.0 - poly_beta2
                        )
                    else:
                        exp_avg_sq_col2_work.lerp_(
                            precond_grad_sq.mean(dim=0), weight=1.0 - poly_beta2
                        )

                    # Compute second denominator
                    denom2 = _approx_sq_grad(
                        exp_avg_sq_row2_work, exp_avg_sq_col2_work, eps
                    )

                    # Step 4: Update full momentum (normal beta1, NOT polybeta1)
                    exp_avg_work.lerp_(precond_grad, weight=1.0 - beta1)
                else:
                    # 1D/0D tensors: simplified path
                    # Step 1: Update first denominator
                    grad_err = grad_work - exp_avg_fac_work
                    exp_avg_sq1_work.lerp_(grad_err.pow(2), weight=1.0 - poly_beta2)
                    denom1 = exp_avg_sq1_work.sqrt().clamp_min_(eps)

                    # Step 2: Update factorized momentum
                    exp_avg_fac_work.lerp_(grad_work, weight=1.0 - poly_beta1)

                    # Step 3: Precondition gradient
                    precond_grad = grad_work / denom1

                    # Step 4: Update second denominator
                    exp_avg_sq2_work.lerp_((precond_grad - exp_avg_work).pow(2), weight=1.0 - poly_beta2)
                    denom2 = exp_avg_sq2_work.sqrt()

                    # Step 4: Update full momentum
                    exp_avg_work.lerp_(precond_grad, weight=1.0 - beta1)

                # Step 6: Compute update direction
                update = exp_avg_work / denom2.clamp_min_(eps)

                # Step 7: Cautious masking (compensated, from ProjAdamW)
                if cautious:
                    mask = (grad_work * update > 0).to(grad_work.dtype)
                    num_agree = mask.sum()
                    dim = update.numel()
                    alpha = dim / (num_agree + 1.0)
                    update = update * mask * alpha

                # Step 8: Weight decay (decoupled)
                if weight_decay != 0:
                    p_work.mul_(1.0 - lr * weight_decay)

                # Step 9: Apply update
                p_work.add_(update, alpha=-lr)

                # State sync (stochastic rounding for BF16)
                if use_stochastic:
                    copy_stochastic_(exp_avg, exp_avg_work)
                    copy_stochastic_(p, p_work)

                    if p.ndim >= 2:
                        copy_stochastic_(exp_avg_fac_row, exp_avg_fac_row_work)
                        copy_stochastic_(exp_avg_fac_col, exp_avg_fac_col_work)
                        copy_stochastic_(exp_avg_sq_row1, exp_avg_sq_row1_work)
                        copy_stochastic_(exp_avg_sq_col1, exp_avg_sq_col1_work)
                        copy_stochastic_(exp_avg_sq_row2, exp_avg_sq_row2_work)
                        copy_stochastic_(exp_avg_sq_col2, exp_avg_sq_col2_work)
                    else:
                        copy_stochastic_(exp_avg_fac, exp_avg_fac_work)
                        copy_stochastic_(exp_avg_sq1, exp_avg_sq1_work)
                        copy_stochastic_(exp_avg_sq2, exp_avg_sq2_work)
                else:
                    exp_avg.copy_(exp_avg_work)
                    p.copy_(p_work)

                    if p.ndim >= 2:
                        exp_avg_fac_row.copy_(exp_avg_fac_row_work)
                        exp_avg_fac_col.copy_(exp_avg_fac_col_work)
                        exp_avg_sq_row1.copy_(exp_avg_sq_row1_work)
                        exp_avg_sq_col1.copy_(exp_avg_sq_col1_work)
                        exp_avg_sq_row2.copy_(exp_avg_sq_row2_work)
                        exp_avg_sq_col2.copy_(exp_avg_sq_col2_work)
                    else:
                        exp_avg_fac.copy_(exp_avg_fac_work)
                        exp_avg_sq1.copy_(exp_avg_sq1_work)
                        exp_avg_sq2.copy_(exp_avg_sq2_work)

        return loss
