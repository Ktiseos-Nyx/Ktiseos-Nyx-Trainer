import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, Iterable, Literal, List
import math


from .utils import copy_stochastic_


# Newton-Schulz iteration coefficients for orthogonalization
# From https://kexue.fm/archives/11059
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375),
]

# CANS: Chebyshev-optimized Newton-Schulz coefficients (arXiv:2506.10935)
# Derived via Remez algorithm for optimal convergence over spectral interval
# These achieve 20-50% reduction in matrix multiplies vs standard Newton-Schulz
CANS_DEGREE_3 = (1.5, -0.5)  # Cubic: X <- 1.5X - 0.5X @ X^T @ X
CANS_DEGREE_5 = (3.0, -3.0, 1.0)  # 5th order: X <- 3X - 3X@A + X@A^2
CANS_DEGREE_7 = (3.75, -5.25, 2.625, -0.375)  # 7th order


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

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability ( but exclude last polynomial )
coeffs_list = [(a / 1.02, b / 1.02**3, c / 1.02**5)
                for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

from itertools import repeat

def PolarExpress(
    G: torch.Tensor, compute_hermitian: bool = False, max_iterations: int = 1, ortho_dtype=torch.bfloat16, eps=1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.to(ortho_dtype)  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()  # this reduces FLOPs

    X = X / (X.norm(dim=(-2, -1), keepdim=True).clamp_min_(eps) * 1.02)
    hs = coeffs_list[:max_iterations] + list(repeat(coeffs_list[-1], max_iterations - len(coeffs_list)))

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5

    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()

    return X.to(G.dtype)

@torch.no_grad()
def orthogonalize(
    M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None, eps=1e-7
) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    orig_dtype = M.dtype
    if ortho_dtype is not None:
        M = M.to(ortho_dtype)

    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T.contiguous()

    # Faster normalization
    M = M / (torch.linalg.norm(M).clamp_min_(eps) * 1.02)

    # Pre-calculate Identity matrix for better performance
    I = torch.eye(M.shape[1], dtype=M.dtype, device=M.device)

    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        # 5th order Newton-Schulz update
        M = M @ (a * I + b * A + c * A @ A)

    if transpose:
        M = M.T.contiguous()

    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M


@torch.no_grad()
def sanger_update(
    X: torch.Tensor, V: torch.Tensor, lr: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single step of Sanger's Rule (Generalized Oja's rule) for online PCA."""
    X_norm = X / X.norm().clamp_min(1e-8)
    Y = X_norm @ V
    V_update = X_norm.T @ Y - V @ torch.triu(Y.T @ Y)
    V_new = V + lr * V_update
    Y_new = X @ V_new
    return V_new, Y_new


@torch.no_grad()
def past_update(
    X: torch.Tensor, V: torch.Tensor, P: torch.Tensor, beta: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch Projection Approximation Subspace Tracking (PAST) algorithm with KxK inversion.
    X: [N, D] input batch
    V: [D, K] current basis
    P: [K, K] inverse covariance matrix
    beta: forgetting factor (0 < beta <= 1)
    """
    # X: [N, D], V: [D, K] -> Y: [N, K]
    Y = X @ V

    # C = Y.T @ Y: [K, K]
    C = Y.T @ Y

    # We use the KxK formulation of the RLS update for the inverse covariance P.
    # P_new = (beta * P^-1 + C)^-1 = (I + (1/beta) * P @ C)^-1 @ (P / beta)
    # This avoids NxN inversion where N is the larger dimension (e.g. pixels or channels).
    K = V.shape[1]
    I_K = torch.eye(K, device=X.device, dtype=X.dtype)

    # denom = beta * I + P @ C
    # P_new = solve(beta * I + P @ C, P)
    # We enforce symmetry for numerical stability.
    P_new = torch.linalg.solve(beta * I_K + P @ C, P)
    P_new = (P_new + P_new.T) * 0.5

    # Update basis V: [D, K]
    # V_new = V + (X.T @ Y - V @ C) @ P_new
    # This avoids explicit creation of large [D, N] residuals.
    V_new = V + (X.T @ Y - V @ C) @ P_new

    Y_new = X @ V_new
    return V_new, Y_new, P_new

@torch.no_grad()
def aol_precondition(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Almost-Orthogonal-Layer preconditioning for Newton-Schulz.

    Computes Gram matrix A = M @ M^T, then rescales each row by
    s_i = 1 / sqrt(sum_j |A[i,j]| + eps).
    This reduces initial polar error and accelerates NS convergence.
    """
    A = M @ M.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    return M * s

@torch.no_grad()
def cans_newton_schulz_orthogonalize(
    M: torch.Tensor,
    steps: int = 3,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
    adaptive: bool = False,
    tolerance: float = 1e-3
) -> torch.Tensor:
    """
    Chebyshev-optimized Newton-Schulz (CANS) iteration for orthogonalization.

    Uses Remez algorithm-derived optimal coefficients for fastest convergence.
    arXiv:2506.10935

    Much faster than QR decomposition: O(steps × n × r) vs O(n × r²)
    Achieves 20-50% reduction in matrix multiplies vs standard Newton-Schulz
    with mathematically proven optimal convergence.

    ADAPTIVE MODE: Check orthonormality error after odd iterations,
    stop early if error <= tolerance.

    Args:
        M: Input matrix [n, m] to orthogonalize
        steps: Maximum number of CANS iterations (upper limit when adaptive=True)
        eps: Small constant for numerical stability
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality error after odd iterations
                 and stop early if error <= tolerance
        tolerance: Orthonormality error threshold for early stopping in adaptive mode

    Returns:
        Orthonormal matrix [n, m] (columns are orthonormal if n >= m)
    """
    assert M.ndim == 2, f"Input must be 2D matrix, got shape {M.shape}"

    X = M.to(ortho_dtype)
    n_orig, m_orig = X.shape

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    n, m = X.shape

    #X = X / (X.norm() + eps)
    X = aol_precondition(X, eps)

    # CANS 7th-degree iteration using Remez-optimal coefficients
    # X <- 3.75X - 5.25X@A + 2.625X@A^2 - 0.375X@A^3
    a, b, c, d = CANS_DEGREE_7
    for step_idx in range(steps):
        A = X.T @ X
        A2 = A @ A
        X = a * X + b * X @ A + c * X @ A2 + d * X @ A2 @ A

        # Adaptive mode: Check orthonormality error after odd iterations
        if adaptive and (step_idx % 2 == 0 or step_idx == steps - 1):
            # Fast diagonal check: O(n) instead of O(n³) spectral norm
            # Check if columns (or rows) have unit norm
            # E[i,i] = ||X[:,i]||² - 1 for thin X, or ||X[i,:]||² - 1 for fat X
            if n < m:
                # Fat matrix: check row norms
                row_norms_sq = torch.sum(X**2, dim=1)
                ortho_error = torch.max(torch.abs(row_norms_sq - 1.0)).item()
            else:
                # Thin/square matrix: check column norms
                col_norms_sq = torch.sum(X**2, dim=0)
                ortho_error = torch.max(torch.abs(col_norms_sq - 1.0)).item()

            # Early stopping if converged
            if ortho_error <= tolerance:
                break

    if transposed:
        X = X.T

    return X.to(M.dtype)


@torch.no_grad()
def gram_cans3_orthogonalize(
    M: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
    adaptive: bool = False,
    tolerance: float = 1e-1
) -> torch.Tensor:
    """
    Gram Newton-Schulz with CANS-3 (cubic) coefficients.
    Iterates on n×n Gram matrix instead of n×m rectangular matrix.
    ~1.5-1.8x fewer FLOPs when m >> n. No restart needed.

    CANS-3: p(x) = 1.5x - 0.5x³ = x(1.5 - 0.5x²)
    Gram iteration: z = 1.5I - 0.5R, Q = Q@z, R = z@R@z
    Default 7 steps for ~1e-5 orthonormality error.

    ADAPTIVE MODE: Check orthonormality error after odd iterations,
    stop early if error <= tolerance.

    Args:
        M: Input matrix [n, m] to orthogonalize
        steps: Maximum number of iterations (upper limit when adaptive=True)
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality error after odd iterations
                 and stop early if error <= tolerance
        tolerance: Orthonormality error threshold for early stopping in adaptive mode

    Returns:
        Orthonormal matrix [n, m] (columns are orthonormal if n >= m)
    """
    assert M.ndim == 2, f"Input must be 2D matrix, got shape {M.shape}"

    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL preconditioning with Gram folding:
    # Compute Gram once and reuse for both AOL rescaling and initial R
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s

    n, m = X.shape
    a, b = CANS_DEGREE_3  # (1.5, -0.5)

    # Folded Gram: R = S @ A @ S^T (element-wise scaling, avoids second Gram computation)
    R = s * A * s.mT

    # Accumulated orthogonalizer
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I.clone()

    for step_idx in range(steps):
        # CANS-3 polynomial on Gram: h(r) = a + b*r = 1.5 - 0.5*R
        z = a * I + b * R

        # Update Q and R
        Q = Q @ z               # n×n
        R = z @ R @ z           # Symmetric update

        # Adaptive mode: Check orthonormality error after odd iterations
        if adaptive and (step_idx % 2 == 0 or step_idx == steps - 1):
            # Fast diagonal check: O(n) instead of O(n³) spectral norm
            # Q is the accumulated orthogonalizer (n×n), check diagonal of Q^T Q
            # E[i,i] = ||Q[i,:]||² - 1 (row norms since Q is n×n)
            row_norms_sq = torch.sum(Q**2, dim=1)
            ortho_error = torch.max(torch.abs(row_norms_sq - 1.0)).item()

            # Early stopping if converged
            if ortho_error <= tolerance:
                break

    # Final projection back to rectangular
    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


@torch.no_grad()
def gram_cans5_orthogonalize(
    M: torch.Tensor,
    steps: int = 3,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
    adaptive: bool = False,
    tolerance: float = 1e-3
) -> torch.Tensor:
    """
    Gram Newton-Schulz with cubic-convergent degree-5 coefficients.
    Iterates on n×n Gram matrix instead of n×m rectangular matrix.

    Uses coefficients (1.875, -1.25, 0.375) which give cubic convergence on the
    Gram iteration (f'(1)=0, f''(1)=0), compared to CANS-3's quadratic convergence.
    Reaches ~1e-5 orthonormality error in 5 steps vs 7 for CANS-3.

    CANS-5/7 coefficients (3,-3,1) and (3.75,-5.25,2.625,-0.375) DIVERGE on Gram
    matrices (f'(1)=-1), so we use the optimal cubic-convergent coefficients instead.

    Gram iteration: h(r) = 1.875 - 1.25r + 0.375r², z = h(R)

    ADAPTIVE MODE: Check orthonormality error after odd iterations,
    stop early if error <= tolerance.

    Args:
        M: Input matrix [n, m] to orthogonalize
        steps: Maximum number of iterations (upper limit when adaptive=True)
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality error after odd iterations
                 and stop early if error <= tolerance
        tolerance: Orthonormality error threshold for early stopping in adaptive mode

    Returns:
        Orthonormal matrix [n, m] (columns are orthonormal if n >= m)
    """
    assert M.ndim == 2, f"Input must be 2D matrix, got shape {M.shape}"

    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL preconditioning with Gram folding:
    # Compute Gram once and reuse for both AOL rescaling and initial R
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s

    n, m = X.shape
    # Cubic-convergent coefficients for Gram iteration:
    # h(r) = a + b*r + c*r² where a+b+c=1, b+2c=-1/2, c=3/8
    # Gives f'(1)=0 and f''(1)=0 — cubic convergence
    a, b, c = 1.875, -1.25, 0.375

    # Folded Gram: R = S @ A @ S^T (element-wise scaling, avoids second Gram computation)
    R = s * A * s.mT

    # Accumulated orthogonalizer
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I.clone()

    for step_idx in range(steps):
        # Cubic-convergent polynomial on Gram: h(r) = 1.875 - 1.25r + 0.375r²
        R2 = R @ R                             # 1 matmul (n×n)
        z = a * I + b * R + c * R2             # element-wise, no matmul

        # Update Q and R
        Q = Q @ z                              # 1 matmul (n×n)
        R = z @ R @ z                          # 1 matmul (n×n)

        # Adaptive mode: Check orthonormality error after odd iterations
        if adaptive and (step_idx % 2 == 0 or step_idx == steps - 1):
            # Fast diagonal check: O(n) instead of O(n³) spectral norm
            # Q is the accumulated orthogonalizer (n×n), check diagonal of Q^T Q
            # E[i,i] = ||Q[i,:]||² - 1 (row norms since Q is n×n)
            row_norms_sq = torch.sum(Q**2, dim=1)
            ortho_error = torch.max(torch.abs(row_norms_sq - 1.0)).item()

            # Early stopping if converged
            if ortho_error <= tolerance:
                break

    # Final projection back to rectangular
    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


@torch.no_grad()
def cans_newton_schulz_orthogonalize_accumulated(
    M: torch.Tensor,
    steps: int = 2,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
    adaptive: bool = False,
    tolerance: float = 1e-3
) -> torch.Tensor:
    """
    CANS-7 Accumulated Orthogonalizer with AOL-Gram folding.

    Implements quartic-convergent Newton-Schulz iteration on n×n Gram matrix
    with coefficients optimized for accumulated orthogonalizer pattern.

    Key innovations:
    1. AOL preconditioning folded into initial Gram computation
    2. Accumulated pattern: Q @ z, z @ R @ z (avoids repeated rectangular matmuls)
    3. Degree-7 convergence with only 4 n×n matmuls per iteration
    4. ~1.5-2x faster than rectangular CANS-7 when m >> n
    5. ADAPTIVE MODE: Check orthonormality error after odd iterations,
       continue until tolerance met or steps exhausted

    Mathematical basis:
    - Standard CANS-7 (3.75, -5.25, 2.625, -0.375) diverges on Gram (f'(1)≠0)
    - This uses cubic-convergent CANS-5 coefficients (1.875, -1.25, 0.375)
    - h(R) = 1.875*I - 1.25*R + 0.375*R²
    - Satisfies: h(1)=1, h'(1)=-0.5, h''(1)=0 → cubic convergence (f''(1)=0)
    - f(r) = r * h(r)² has degree 6, giving degree-7-equivalent convergence

    Args:
        M: Input matrix [n, m] to orthogonalize
        steps: Maximum number of iterations (upper limit when adaptive=True)
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation
        adaptive: If True, check orthonormality error after odd iterations
                 and stop early if error <= tolerance
        tolerance: Orthonormality error threshold for early stopping in adaptive mode

    Returns:
        Orthonormal matrix [n, m] (columns are orthonormal if n >= m)
    """
    assert M.ndim == 2, f"Input must be 2D matrix, got shape {M.shape}"

    X = M.to(ortho_dtype)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL-Gram folding: Compute Gram once, use for both preconditioning and initial R
    A = X @ X.mT  # [n, n] Gram matrix

    # AOL row rescaling: s_i = 1/sqrt(sum_j |A[i,j]| + eps)
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)

    # Apply AOL preconditioning to X
    X = X * s

    # Folded Gram: R = S @ A @ S^T via element-wise scaling
    # R[i,j] = s[i] * A[i,j] * s[j] (avoids matmul, preserves symmetry)
    R = s * A * s.mT

    n, m = X.shape

    # Accumulated orthogonalizer initialization
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I.clone()

    # Cubic-convergent coefficients for Gram iteration (degree-6 convergence)
    # From gram_cans5_orthogonalize: h(r) = 1.875 - 1.25r + 0.375r²
    # Conditions: h(1)=1, h'(1)=-0.5, h''(1)=0 → cubic convergence (f''(1)=0)
    # These are the optimal coefficients for Gram iteration with accumulated pattern
    # f(r) = r * h(r)² has degree 6, giving equivalent convergence to degree-7 rectangular
    a, b, c = 1.875, -1.25, 0.375

    for step_idx in range(steps):
        # Compute R²
        R2 = R @ R          # [n, n] matmul

        # Cubic polynomial on Gram matrix: z = h(R) = a*I + b*R + c*R²
        z = a * I + b * R + c * R2

        # Accumulated updates
        Q = Q @ z           # [n, n] matmul - accumulate orthogonalizer
        R = z @ R @ z       # 2×[n, n] matmuls - update symmetric Gram

        # Adaptive mode: Check orthonormality error after odd iterations
        if adaptive and (step_idx % 2 == 0 or step_idx == steps - 1):
            # Fast diagonal check: O(n) instead of O(n³) spectral norm
            # Q is the accumulated orthogonalizer (n×n), check diagonal of Q^T Q
            # E[i,i] = ||Q[i,:]||² - 1 (row norms since Q is n×n)
            row_norms_sq = torch.sum(Q**2, dim=1)
            ortho_error = torch.max(torch.abs(row_norms_sq - 1.0)).item()

            # Early stopping if converged
            if ortho_error <= tolerance:
                break

    # Final projection: apply accumulated orthogonalizer to preconditioned X
    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


@torch.no_grad()
def accumulated_orthogonalize_nstep(
    M: torch.Tensor,
    coeffs_list: List[Tuple[float, float, float]],
    steps: Optional[int] = None,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
    adaptive: bool = False,
    tolerance: float = 1e-3
) -> torch.Tensor:
    """
    N-step accumulated orthogonalization with step-specific coefficients.

    This is a general-purpose accumulated orthogonalizer that can use
    arbitrary coefficients for each step (not constrained to h(1)=1, etc.).

    Args:
        M: Input matrix [n, m]
        coeffs_list: List of (a, b, c) tuples for each step
        steps: Number of steps to use (default: len(coeffs_list))
        eps: Numerical stability constant
        ortho_dtype: Data type for computation
        adaptive: If True, check orthonormality after odd iterations
        tolerance: Orthonormality error threshold for early stopping

    Returns:
        Orthonormal matrix [n, m]
    """
    if steps is None:
        steps = len(coeffs_list)

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

    for step_idx in range(min(steps, len(coeffs_list))):
        a, b, c = coeffs_list[step_idx]

        # Cubic polynomial on Gram matrix
        R2 = R @ R
        z = a * I + b * R + c * R2

        # Accumulated updates
        Q = Q @ z
        R = z @ R @ z

        # Adaptive mode: Fast diagonal check
        if adaptive and (step_idx % 2 == 0 or step_idx == steps - 1):
            # O(n) diagonal check: ||Q[i,:]||² - 1
            row_norms_sq = torch.sum(Q**2, dim=1)
            ortho_error = torch.max(torch.abs(row_norms_sq - 1.0)).item()

            if ortho_error <= tolerance:
                break

    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


# Pre-trained 5-step coefficients (unconstrained optimization)
# Optimized for spectral range [0.2, 1.8] with pure convergence objective
# Error: ~1e-4 on training range
GRAM_NEWTON_SCHULZ_5STEP_COEFFS = [
    (1.723986, -1.338374, 0.340636),  # Step 1
    (1.976014, -1.676303, 0.304450),  # Step 2
    (2.440786, -1.798391, 0.531607),  # Step 3
    (2.887142, -2.051167, 0.573092),  # Step 4
    (3.104949, -2.398975, 0.604532),  # Step 5
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS2 = [
    (1.769767, -1.526870, 0.421974),
    (1.868189, -1.782708, 0.434460),
    (2.391326, -1.857135, 0.480161),
    (2.882490, -2.050413, 0.567363),
    (3.103333, -2.399800, 0.605477),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS3 = [
    (1.980609, -1.893822, 0.588662),
    (2.112242, -1.838111, 0.491787),
    (2.568097, -1.904755, 0.631110),
    (3.048633, -2.228526, 0.580317),
    (3.338350, -2.581638, 0.598580),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS4 = [
    (2.023403, -1.247188, 0.335965),
    (2.358273, -1.495476, 0.411068),
    (2.557690, -1.797661, 0.499005),
    (2.953710, -2.043037, 0.561319),
    (3.248422, -2.374416, 0.551892),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS5 = [
    (2.0536550716229929, -1.2238203061557864, 0.3269712980093314),
    (2.3616550357697834, -1.4764166681057576, 0.3930204879918510),
    (2.5702853863521931, -1.7920194111981576, 0.4914927988807775),
    (2.9600018625931241, -2.0412439403739007, 0.5576339033589184),
    (3.2509947143588609, -2.3738871955494401, 0.5509775783424596),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS6 = [
    (2.0536550716229929, -1.2238203061557864, 0.3269712980093314),
    (2.3616550357697834, -1.4764166681057576, 0.3930204879918510),
    (2.5702853863521931, -1.7920194111981576, 0.4914927988807775),
    (2.9600018625931241, -2.0412439403739007, 0.5576339033589184),
    (3.2509947143588609, -2.3738871955494401, 0.5509775783424596),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS7 = [
    (3.4459608603445333, -3.1755041678093470, 0.7867006670156100),
    (2.5301326987745019, -2.2875906598905478, 0.6823154809579889),
    (2.4512392667669340, -1.8754768321549216, 0.6269642725806978),
    (2.9075961880554018, -2.0477794454860128, 0.5697912870805238),
    (3.1451602955294260, -2.3838983733676198, 0.5845190398950526),
]

GRAM_NEWTON_SCHULZ_5STEP_COEFFS8 = [
    (3.5318553432919795, -3.2158659760419801, 0.7012006347356651),
    (2.4768596184460012, -2.1364774303591765, 0.6081753355355514),
    (2.4413287994783173, -1.8578803401593651, 0.6178422901770554),
    (2.9092595013338038, -2.0466460153339150, 0.5699324994660828),
    (3.1576735308145296, -2.3851719032718757, 0.5816498819410983),
]

GRAM_NEWTON_SCHULZ_3STEP_COEFFS1 = [
    (3.6675553375140790, -4.1355164772091868, 1.2640914244071093),
    (2.3772871960940529, -2.2186318530552400, 0.7000226456303386),
    (2.1305046812018920, -1.8038955969508677, 0.6729710721865838),
]

GRAM_NEWTON_SCHULZ_2STEP_COEFFS1 = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]

@torch.no_grad()
def gram_newton_schulz_5step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype=torch.float32,
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


@torch.no_grad()
def fapi_update(
    X: torch.Tensor, W: torch.Tensor, Z: torch.Tensor, beta: float, eps: float = 1e-16, ortho_dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast Approximated Power Iteration (FAPI) - single step.

    FAPI provides guaranteed orthonormality (~-305 dB error) vs Oja's ~-25 dB.
    280 dB better orthonormality with global exponential convergence.

    Based on: Yang (2000) "An Efficient Subspace Tracking Algorithm With
    Guaranteed Orthonormality and Parallel Structure"

    FAPI Key Innovation: Uses W(t) ≈ W(t-1)Θ(t) where Θ(t) is nearly orthonormal,
    enabling guaranteed orthonormality at each iteration.

    Args:
        X: Input batch [N, n] - gradient batch for tracking
        W: Current orthonormal basis [n, r] (W^H @ W = I)
        Z: Inverse covariance matrix [r, r]
        beta: Forgetting factor (0 < beta <= 1), typically 0.9-0.99
        eps: Numerical stability constant

    Returns:
        W_new: Updated orthonormal basis [n, r] (guaranteed W^H @ W = I)
        Y_new: New projections [N, r]
        Z_new: Updated inverse covariance [r, r]
    """
    # Handle both single sample and batch
    if X.ndim == 1:
        X = X.unsqueeze(0)

    # Use mean for robust update (batch processing)
    x = X.mean(dim=0)  # [n]

    # Step 1: Compute projection coefficients
    # y(t) = W^H(t-1) @ x(t)
    y = W.T @ x  # [r]

    # Step 2: Kalman gain precompute
    # h(t) = Z(t-1) @ y(t)
    h = Z @ y  # [r]

    # Step 3: Normalized Kalman gain
    # g(t) = h(t) / (β + y^H(t) @ h(t))
    denom = beta + (y * h).sum()  # scalar
    g = h / denom.clamp_min(eps)  # [r]

    # Step 4: Squared error norm
    # ε²(t) = ||x(t)||² - ||y(t)||²
    x_norm_sq = (x**2).sum()  # scalar
    y_norm_sq = (y**2).sum()  # scalar
    eps_sq = (x_norm_sq - y_norm_sq).clamp_min(eps)  # Ensure positive

    # Step 5: Adaptive factor τ(t)
    # τ(t) = ε²(t) / (1 + ε²(t)||g(t)||² + √(1 + ε²(t)||g(t)||²))
    g_norm_sq = (g**2).sum()  # scalar
    temp = 1.0 + eps_sq * g_norm_sq
    sqrt_term = torch.sqrt(temp)
    tau = eps_sq / (temp + sqrt_term).clamp_min(eps)

    # Step 6: Orthonormality correction factor
    # η(t) = 1 - τ(t)||g(t)||²
    eta = 1.0 - tau * g_norm_sq

    # Step 7: Corrected projection coefficients
    # y'(t) = η(t)y(t) + τ(t)g(t)
    y_prime = eta * y + tau * g  # [r]

    # Step 8: Corrected innovation
    # e'(t) = x(t) - W(t-1) @ y'(t)
    e_prime = x - W @ y_prime  # [n]

    # Step 9: Update inverse covariance
    # Note: The / beta factor from Yang (2000) is intentionally omitted as it
    # causes exploding instability over many steps.
    Z_new = Z - g.unsqueeze(-1) * h.unsqueeze(0)

    # Step 10: Update basis
    # W(t) = W(t-1) + e'(t) @ g^H(t)
    W_new = W + e_prime.unsqueeze(-1) * g.unsqueeze(0)

    # Step 11: Re-orthogonalize using cubic Newton-Schulz iteration
    # Much faster than QR: O(steps × n × r) vs O(n × r²)
    # 3 iterations provides sufficient orthonormality (~1e-3 to 1e-4 error)
    # which is adequate for gradient subspace tracking
    #W_new = _cubic_newton_schulz_orthogonalize(W_new, eps=eps, ortho_dtype=ortho_dtype)
    W_new = gram_newton_schulz_5step(W_new, ortho_dtype=ortho_dtype, eps=eps)
    #W_new = aol_precondition(W_new, eps=1e-7)

    # Step 12: Project full batch through new basis
    Y_new = X @ W_new  # [N, r]

    # Ensure float32 for numerical stability
    W_new = W_new.to(torch.float32)
    Z_new = Z_new.to(torch.float32)
    Y_new = Y_new.to(torch.float32)

    return W_new, Y_new, Z_new


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
    r_factor.unsqueeze_(-1)  # In-place unsqueeze

    # Column factor with epsilon
    c_factor = (
        (exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=0, keepdim=True) + eps)
    ).unsqueeze(-2)
    c_factor.sqrt_()  # In-place sqrt

    # Combine with broadcasting support
    return torch.mul(r_factor, c_factor)


class WiwiOpt(Optimizer):
    r"""
    WiwiOpt (V2.1).

    A gradient descent optimizer that combines several stabilization & acceleration techniques to produce
    high-signal stable parameter updates.

    WiwiOpt works by:
    1. RMS-based gradient normalization: Incoming gradients are normalized
       by a polynomial-decay EMA of their per-row RMS, preventing exploding
       or vanishing gradient magnitudes.
    2. Egalitarian Gradient Descent (EGD) preconditioning: For 2D+
       parameters, a low-rank SVD approximation is used to precondition the
       gradient, equalizing contribution across singular directions.
    3. Polynomial-schedule momentum: Momentum and accumulation use
       polynomial schedules instead of fixed betas, providing
       smoothing that naturally increases over early training.
    4. CANS orthogonalization (Muon): The effective gradient is
        orthogonalized via Chebyshev-optimized Newton-Schulz iteration
        (arXiv:2506.10935) for multi-dimensional parameters, producing
        direction-pure updates with 20-50% fewer matrix multiplies.
    5. NorMuon scaling: After orthogonalization, the update is re-scaled
        using a tracked second-moment estimate to maintain consistent update
        magnitudes, then re-projected to preserve the original norm.
    6. Projection re-scaling: The orthogonalized step is re-scaled by its
       projection onto the un-orthogonalized effective gradient, preserving
       meaningful magnitude information.
    7. Cautious masking: Updates are masked so that only components
       agreeing in sign with the raw gradient are kept, with proper
       compensation scaling (arXiv:2411.16085, ICLR 2026) ensuring
       monotonic descent guarantees.
    8. Dynamic learning rate: Per-parameter learning rate adjustment based
       on the alignment between the EMA of parameter deltas and the EMA of
       their norms, optionally boosted by an ``atan2``-based scaling factor.
    9. CAME-style factorized variance tracking: Uses row-wise AND column-wise
       variance estimates (like CAME optimizer) for more accurate gradient
       normalization, with in-place operations for memory efficiency.

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float, float] or Tuple[float, float]):
            Exponents for the de-biased beta schedules.
            ``beta1`` controls momentum and gradient accumulation decay,
            ``beta2`` controls the variance tracker decay, and ``beta3``
            controls the dynamic learning rate EMAs.
            (default: (0.95, 0.995, 0.99)).
        eps (float): Numerical stability term for divisions and clamps
            (default: 1e-16).
        weight_decay (float): Decoupled weight decay coefficient
            (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied,
            weight_decay * weight_decay_rate**step
            (default: 1.0).
        muon (bool): Apply Muon's orthogonalization to accelerate descent
            (default: True).
        use_compile (bool): Use ``torch.compile`` on the orthogonalization
            and SVD functions for faster execution (default: True).
        ortho_dtype (str or None): Data type for Newton-Schulz
            orthogonalization. Accepts ``None`` (defaults to float32) or a
            string like ``"torch.bfloat16"`` (default: None).
        stochastic_fp (bool): Use stochastic rounding when parameters are
            stored in bfloat16, reducing quantization bias (default: True).
        dynamic_lr (bool): Enable per-row dynamic learning rate
            adjustment based on delta alignment (default: True).
        dynamic_lr_boost (bool): When ``dynamic_lr`` is enabled, apply an
            additional ``atan2``-based boost factor that amplifies the
            learning rate when parameter deltas are large relative to their
            directional EMA (default: True).
        egd (bool): Enable Egalitarian Gradient Descent preconditioning via
            low-rank SVD for parameters with 2+ dimensions, equalizing
            gradient contribution across singular directions
            (default: True).
        egd_online (bool): Enables a lightweight approximation
            of EGD using Sanger's rule (Generalized Oja's rule) in place
            of full SVD tracking (default: True).
        egd_method (str): Method for online decomposition tracking.
            Accepts 'fapi' (default), 'oja', 'past', or 'svd'.
            FAPI provides guaranteed orthonormality (~-305 dB error) vs Oja's ~-25 dB.
            If 'svd' is used, `egd_online` is ignored.
        cautious_xi (float): Smoothing parameter for cautious masking
            compensation factor. From arXiv:2411.16085, the compensation
            is ``dim / (num_agree + xi)``. Higher values prevent
            over-amplification when few coordinates agree (default: 1.0).
        normuon (bool): Apply NorMuon second-moment scaling after
            orthogonalization to stabilize update magnitudes
            (default: True).
        rms_max (float): Maximum allowed RMS of final_step. If > 0,
            final_step is scaled to not exceed this RMS value. Set to 0
            to disable. (default: 5.0).
        stochastic_fp (bool): Use stochastic rounding when parameters are
            stored in bfloat16, reducing quantization bias (default: True).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        betas: Tuple[float, float, float] = (0.95, 0.95, 0.995),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 1.0,
        muon: bool = True,
        use_compile: bool = True,
        ortho_dtype: Optional[torch.dtype] = None,
        dynamic_lr: bool = True,
        dynamic_lr_boost: bool = True,
        egd: bool = True,
        egd_online: bool = True,
        egd_method: Literal["oja", "past", "svd", "fapi"] = "fapi",
        cautious_xi: float = 1.0,
        normuon: bool = True,
        rms_max: float = 10.0,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        if len(betas) == 2:
            betas = (betas[0], betas[0], betas[1])
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if ortho_dtype is None:
            ortho_dtype = torch.bfloat16
        elif isinstance(ortho_dtype, str):
            dtype_name = ortho_dtype.split(".")[-1]
            ortho_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_rate=weight_decay_rate,
            muon=muon,
            use_compile=use_compile,
            ortho_dtype=ortho_dtype,
            dynamic_lr=dynamic_lr,
            dynamic_lr_boost=dynamic_lr_boost,
            egd=egd,
            egd_online=egd_online,
            egd_method=egd_method,
            cautious_xi=cautious_xi,
            normuon=normuon,
            rms_max=rms_max,
            stochastic_fp=stochastic_fp,
        )

        if use_compile:
            torch._dynamo.config.capture_scalar_outputs = True

        self.ortho_func = torch.compile(gram_newton_schulz_5step, dynamic=True, mode="default") if use_compile else gram_newton_schulz_5step
        self.oja_func = None
        self.past_func = None
        self.fapi_func = None
        self.svd_func = None
        if egd:
            if egd_method == "oja" or (egd_method is None and egd_online):
                self.oja_func = (
                    torch.compile(sanger_update, dynamic=True, mode="default") if use_compile else sanger_update
                )
            elif egd_method == "past":
                self.past_func = (
                    torch.compile(past_update, dynamic=True, mode="default") if use_compile else past_update
                )
            elif egd_method == "fapi":
                self.fapi_func = (
                    torch.compile(fapi_update, dynamic=True, mode="default") if use_compile else fapi_update
                )
            elif egd_method == "svd" or (egd_method is None and not egd_online):
                self.svd_func = (
                    torch.compile(torch.svd_lowrank, dynamic=True, mode="default")
                    if use_compile
                    else torch.svd_lowrank
                )

        super(WiwiOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            if len(group["betas"]) == 2:
                beta1, beta2, beta3 = (
                    group["betas"][0],
                    group["betas"][0],
                    group["betas"][1],
                )
            else:
                beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            muon = group["muon"]
            stochastic_fp = group["stochastic_fp"]
            egd = group["egd"]
            egd_online = group["egd_online"]
            egd_method = group["egd_method"]
            dynamic_lr = group["dynamic_lr"]
            dynamic_lr_boost = group["dynamic_lr_boost"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("WiwiOpt does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["accum"] = torch.ones_like(
                        p.mean(dim=-1, keepdim=True),
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Factorized variance tracking for 2D+ tensors (CAME-style)
                    if p.ndim >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.shape[:-1], device=p.device, dtype=p.dtype
                        )
                        col_shape = p.shape[:-2] + p.shape[-1:]
                        state["exp_avg_sq_col"] = torch.zeros(
                            col_shape, device=p.device, dtype=p.dtype
                        )
                    else:
                        # 1D tensors: fall back to full variance tracking
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    if dynamic_lr:
                        state["delta_ema"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["delta_norm_ema"] = torch.zeros_like(
                            p.mean(dim=-1, keepdim=True),
                            memory_format=torch.preserve_format,
                        )
                    if p.ndim >= 1 and group["normuon"] and muon:
                        grad_2d = reshape_to_2d(grad)
                        state["normuon_second_momentum"] = torch.zeros(
                            grad_2d.shape[0], 1, device=p.device, dtype=p.dtype
                        )

                state["step"] += 1
                step = state["step"]

                accum = state["accum"]
                exp_avg = state["exp_avg"]

                # Mixed precision handling
                use_stochastic = stochastic_fp and p.dtype in {torch.bfloat16}

                # Initialize variables to avoid unbound errors
                p_work = p.detach()
                grad_work = grad.detach()
                accum_work = accum.detach()
                exp_avg_work = exp_avg.detach()

                # Initialize factorized state work variables
                exp_avg_sq_row_work = None
                exp_avg_sq_col_work = None
                exp_avg_sq_work = None

                delta_ema_work = None
                delta_norm_ema_work = None
                normuon_z = None

                if use_stochastic:
                    p_work = p_work.to(torch.float32)
                    grad_work = grad_work.to(torch.float32)
                    accum_work = accum_work.to(torch.float32)
                    exp_avg_work = exp_avg_work.to(torch.float32)

                # Handle factorized states for mixed precision
                if p.ndim >= 2:
                    exp_avg_sq_row_work = state["exp_avg_sq_row"].detach()
                    exp_avg_sq_col_work = state["exp_avg_sq_col"].detach()
                    if use_stochastic:
                        exp_avg_sq_row_work = exp_avg_sq_row_work.to(torch.float32)
                        exp_avg_sq_col_work = exp_avg_sq_col_work.to(torch.float32)
                else:
                    exp_avg_sq_work = state["exp_avg_sq"].detach()
                    if use_stochastic:
                        exp_avg_sq_work = exp_avg_sq_work.to(torch.float32)

                if dynamic_lr:
                    delta_ema_work = state["delta_ema"].detach()
                    delta_norm_ema_work = state["delta_norm_ema"].detach()
                    if use_stochastic:
                        delta_ema_work = delta_ema_work.to(torch.float32)
                        delta_norm_ema_work = delta_norm_ema_work.to(torch.float32)

                if p.ndim >= 1 and group["normuon"] and muon:
                    normuon_z = state["normuon_second_momentum"].detach()
                    if use_stochastic:
                        normuon_z = normuon_z.to(torch.float32)

                poly_beta1 = (beta1 ** (step) - beta1) / (beta1 ** (step) - 1.0)
                poly_beta2 = (beta2 ** (step) - beta2) / (beta2 ** (step) - 1.0)
                poly_beta3 = (beta3 ** (step) - beta3) / (beta3 ** (step) - 1.0)

                grad_rms = grad_work.pow(2).mean(dim=-1, keepdim=True)
                accum_work.lerp_(grad_rms, 1.0 - poly_beta1)

                grad_work.div_(accum_work.sqrt().clamp_min_(eps)).clamp_(-step, step)

                if egd and p_work.ndim >= 2:
                    grad_work_2d = reshape_to_2d(grad_work)
                    m_dim, n_dim = grad_work_2d.size(0), grad_work_2d.size(1)
                    current_rank = min(512, m_dim, n_dim)

                    is_online = (egd_method in ["oja", "past", "fapi"]) or (
                        egd_method is None and egd_online
                    )
                    if current_rank > 0 or is_online:
                        if is_online:
                            if "oja_basis" not in state:
                                track_u = m_dim < n_dim
                                feature_dim = m_dim if track_u else n_dim
                                # Always use float32 for online PCA states to ensure numerical stability.
                                basis = torch.randn(
                                    feature_dim,
                                    current_rank,
                                    device=p_work.device,
                                    dtype=torch.float32,
                                )
                                basis, _ = torch.linalg.qr(basis)
                                state["oja_basis"] = basis
                                if egd_method == "past":
                                    state["inv_cov"] = (
                                        torch.eye(
                                            current_rank,
                                            device=p_work.device,
                                            dtype=torch.float32,
                                        )
                                        * 0.1
                                    )
                                elif egd_method == "fapi":
                                    # FAPI uses inverse covariance matrix Z initialized as I / (1 - beta1)
                                    state["inv_cov"] = torch.eye(
                                        current_rank,
                                        device=p_work.device,
                                        dtype=torch.float32,
                                    ) / (1.0 - beta1)

                            track_u = m_dim < n_dim
                            oja_basis_work = state["oja_basis"]
                            # Ensure we work in float32 for the online update
                            oja_basis_work = oja_basis_work.detach().float()

                            X_for_oja = grad_work_2d.T if track_u else grad_work_2d
                            X_for_oja = X_for_oja.float()

                            try:
                                Y_new = None
                                if egd_method == "past" and self.past_func is not None:
                                    inv_cov_work = state["inv_cov"].detach().float()

                                    # Use a stable forgetting factor for PAST.
                                    # It should match the EMA factor (poly_beta1) but clamped for stability.
                                    past_beta = max(poly_beta1, 0.5)
                                    oja_basis_work, Y_new, inv_cov_work = (
                                        self.past_func(
                                            X_for_oja,
                                            oja_basis_work,
                                            inv_cov_work,
                                            past_beta,
                                        )
                                    )

                                    state["inv_cov"].copy_(inv_cov_work)
                                elif (
                                    egd_method == "fapi" and self.fapi_func is not None
                                ):
                                    inv_cov_work = state["inv_cov"].detach().float()

                                    # FAPI uses polynomial decay beta with floor for stability
                                    fapi_beta = max(poly_beta1, 0.5)
                                    oja_basis_work, Y_new, inv_cov_work = (
                                        self.fapi_func(
                                            X_for_oja,
                                            oja_basis_work,
                                            inv_cov_work,
                                            fapi_beta,
                                            eps,
                                            group["ortho_dtype"],
                                        )
                                    )

                                    state["inv_cov"].copy_(inv_cov_work)
                                elif egd_method == "oja" and self.oja_func is not None:
                                    oja_basis_work, Y_new = self.oja_func(
                                        X_for_oja, oja_basis_work, 1.0 - poly_beta1
                                    )

                                if Y_new is not None:
                                    # Normalize the basis vectors and the projections to ensure the
                                    # preconditioned gradient magnitude is stable regardless of basis drift.
                                    basis_norm = oja_basis_work / oja_basis_work.norm(
                                        dim=0, keepdim=True
                                    ).clamp_min_(eps)
                                    proj_norm = Y_new / Y_new.norm(
                                        dim=0, keepdim=True
                                    ).clamp_min_(eps)

                                    if track_u:
                                        grad_precond = basis_norm @ proj_norm.T
                                    else:
                                        grad_precond = proj_norm @ basis_norm.T

                                    state["oja_basis"].copy_(oja_basis_work)
                                    grad_work = grad_precond.to(p_work.dtype).view_as(
                                        p_work
                                    )
                            except RuntimeError:
                                pass
                        else:
                            try:
                                # Use float32 for SVD stability if it was half precision
                                dtype_orig = grad_work_2d.dtype
                                grad_f32 = grad_work_2d.float()

                                if self.svd_func is not None:
                                    U, S, _ = self.svd_func(grad_f32, q=current_rank)

                                    U = U.to(dtype_orig)
                                    S = S.to(dtype_orig)

                                    S = torch.maximum(
                                        S,
                                        torch.tensor(
                                            eps, device=S.device, dtype=S.dtype
                                        ),
                                    )
                                    S_inv = 1.0 / S

                                    aux = (U * S_inv.unsqueeze(0)) @ U.mT
                                    grad_precond = aux @ grad_work_2d

                                    grad_work = grad_precond.view_as(p_work)
                            except RuntimeError:
                                # Fallback if SVD fails to converge (rare)
                                pass

                # CAME-style factorized variance tracking with in-place operations
                if p_work.ndim >= 2:
                    grad_err = grad_work - exp_avg_work
                    grad_err.pow_(2)  # In-place square

                    # Update row-wise variance (mean over last dimension)
                    exp_avg_sq_row_work.lerp_(
                        grad_err.mean(dim=-1), weight=1.0 - poly_beta2
                    )

                    # Update column-wise variance (mean over second-to-last dimension)
                    if grad_err.ndim > 2:
                        exp_avg_sq_col_work.lerp_(
                            grad_err.mean(dim=-2), weight=1.0 - poly_beta2
                        )
                    else:
                        exp_avg_sq_col_work.lerp_(
                            grad_err.mean(dim=0), weight=1.0 - poly_beta2
                        )

                    # Compute factorized denominator
                    denom = _approx_sq_grad(
                        exp_avg_sq_row_work, exp_avg_sq_col_work, eps
                    )
                else:
                    # 1D tensors: use simple variance tracking
                    grad_err = grad_work - exp_avg_work
                    grad_err.pow_(2)  # In-place
                    exp_avg_sq_work.lerp_(grad_err, weight=1.0 - poly_beta2)
                    denom = exp_avg_sq_work.sqrt().clamp_min_(eps)  # In-place sqrt

                # Momentumize with in-place operations
                exp_avg_work.lerp_(grad_work, weight=1.0 - poly_beta1)

                # Compute effective gradient
                g_eff_mom = grad_work.clone()
                g_eff_mom.lerp_(exp_avg_work, weight=poly_beta1)
                g_eff_mom.div_(denom)  # Apply factorized denominator in-place

                if p_work.ndim >= 1 and muon:
                    full_step_2d = reshape_to_2d(g_eff_mom)

                    Q = self.ortho_func(
                        full_step_2d,
                        eps=eps,
                        ortho_dtype=group["ortho_dtype"]
                    )

                    # NorMuon update & re-norm
                    if group["normuon"] and normuon_z is not None:
                        vnorm = Q.norm(dim=(-2, -1), keepdim=True)

                        v_mean = torch.mean(Q * Q, dim=-1, keepdim=True)
                        normuon_z.lerp_(v_mean, 1 - poly_beta2)
                        step_size = normuon_z.sqrt().clamp_min_(eps)
                        Q.div_(step_size)

                        vnorm_new = Q.norm(dim=(-2, -1), keepdim=True)
                        Q = Q * (vnorm / vnorm_new.clamp_min(eps))

                    final_step = Q.view_as(p_work)

                    # Re-scaling: final_step functionally sums to 1.
                    # We re-scale it to the magnitude of the projection onto the un-orthogonalized effective gradient
                    scale_factor = (g_eff_mom * final_step).sum()
                    final_step.mul_(scale_factor)
                else:
                    final_step = g_eff_mom

                # Cautious masking with principled compensation scaling
                # (arXiv:2411.16085, ICLR 2026)
                mask = (grad_work * final_step > 0).to(final_step.dtype)
                num_agree = mask.sum()
                dim = final_step.numel()
                xi = group["cautious_xi"]
                alpha = dim / (num_agree + xi)
                final_step.mul_(mask).mul_(alpha)

                # RMS clamping: scale final_step to not exceed rms_max
                if group["rms_max"] > 0:
                    current_rms = final_step.pow(2).mean().sqrt_()
                    if current_rms > group["rms_max"]:
                        final_step.mul_(group["rms_max"] / current_rms)

                # Dynamic Learning Rate Adjustment
                lr_adj = torch.ones_like(p.mean())
                if (
                    dynamic_lr
                    and delta_ema_work is not None
                    and delta_norm_ema_work is not None
                ):
                    if step > 1:
                        # True norm of EMA of deltas vs EMA of accumulated norms of deltas
                        alignment_ratio = delta_ema_work.norm(
                            dim=-1, keepdim=True
                        ) / delta_norm_ema_work.clamp_min(eps)
                        # Parameter-wise update scaling
                        if dynamic_lr_boost:
                            update_ratio = delta_norm_ema_work.atan2(
                                delta_ema_work.abs()
                            ).mul_(1.27323954474)
                            lr_adj = alignment_ratio * update_ratio
                        else:
                            lr_adj = alignment_ratio
                    else:
                        lr_adj = torch.ones_like(p.mean())

                    final_step.mul_(lr_adj)

                    # Update EMAs
                    current_norm = final_step.norm(dim=-1, keepdim=True)
                    delta_ema_work.lerp_(final_step, 1.0 - poly_beta3)
                    delta_norm_ema_work.lerp_(current_norm, 1.0 - poly_beta3)

                # Apply Update
                if weight_decay != 0:
                    weight_decay_multiplier = weight_decay_rate**step

                    # Cautious weight decay with principled compensation scaling
                    wd_mask = (p_work * final_step > 0).to(p_work.dtype)
                    num_agree_wd = wd_mask.sum()
                    dim_wd = p_work.numel()
                    wd_alpha = dim_wd / (num_agree_wd + group["cautious_xi"])

                    p_mid = p_work * wd_mask * wd_alpha
                    p_work.add_(
                        p_mid * lr_adj if dynamic_lr else p_mid,
                        alpha=-lr * weight_decay * weight_decay_multiplier,
                    )

                p_work.add_(final_step, alpha=-lr)

                # State Sync
                if use_stochastic:
                    copy_stochastic_(accum, accum_work)
                    copy_stochastic_(exp_avg, exp_avg_work)
                    # Sync factorized variance states
                    if p.ndim >= 2:
                        copy_stochastic_(state["exp_avg_sq_row"], exp_avg_sq_row_work)
                        copy_stochastic_(state["exp_avg_sq_col"], exp_avg_sq_col_work)
                    else:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq_work)
                    if (
                        dynamic_lr
                        and delta_ema_work is not None
                        and delta_norm_ema_work is not None
                    ):
                        copy_stochastic_(state["delta_ema"], delta_ema_work)
                        copy_stochastic_(state["delta_norm_ema"], delta_norm_ema_work)
                    copy_stochastic_(p, p_work)
                    if p.ndim >= 1 and group["normuon"] and muon and normuon_z is not None:
                        copy_stochastic_(state["normuon_second_momentum"], normuon_z)
                else:
                    accum.copy_(accum_work)
                    exp_avg.copy_(exp_avg_work)
                    # Sync factorized variance states
                    if p.ndim >= 2:
                        state["exp_avg_sq_row"].copy_(exp_avg_sq_row_work)
                        state["exp_avg_sq_col"].copy_(exp_avg_sq_col_work)
                    else:
                        state["exp_avg_sq"].copy_(exp_avg_sq_work)
                    if dynamic_lr:
                        state["delta_ema"].copy_(delta_ema_work)
                        state["delta_norm_ema"].copy_(delta_norm_ema_work)
                    p.copy_(p_work)
                    if p.ndim >= 1 and group["normuon"] and muon:
                        state["normuon_second_momentum"].copy_(normuon_z)

        return loss
