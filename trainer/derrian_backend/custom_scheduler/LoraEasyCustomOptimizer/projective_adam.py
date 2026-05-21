import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, Iterable, Literal
import math
import logging

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
    (1.875, -1.25, 0.375)
]

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


@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = M.dtype
        M = M.to(ortho_dtype)
    
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T.contiguous()
    
    # Pre-calculate Identity matrix for better performance
    I = torch.eye(M.shape[1], dtype=M.dtype, device=M.device)
    
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        # Faster normalization
        M = M / (torch.linalg.norm(M).clamp_min_(1e-8))
        A = M.T @ M
        # 5th order Newton-Schulz update
        M = M @ (a * I + b * A + c * A @ A)
    
    if transpose:
        M = M.T.contiguous()
    
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M


class ProjectiveAdam(Optimizer):
    r"""
    ProjectiveAdam: An Adam-based optimizer with selectable geometric projections.
    
    This optimizer maps gradients onto a geometric manifold using one of several
    projection types, tracks momentum on that manifold, and reconstructs the
    update via inverse projection.
    
    Supported Projections:
        - 'stereographic': Maps R^n -> S^n via stereographic projection from the south pole.
        - 'gnomonic': Maps R^n -> Hemisphere via central/gnomonic projection.
        - 'hyperbolic': Maps R^n -> Poincaré Ball (Hyperbolic space) via tanh scaling.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for EMAs (default: (0.95, 0.999)).
        eps (float): Numerical stability term (default: 1e-16).
        weight_decay (float): Weight decay coefficient (default: 0.0).
        projection (str): Projection type: 'stereographic', 'gnomonic', 'hyperbolic' (default: 'gnomonic').
        input_norm (bool): Normalize RMS by last 2D dimension if True, otherwise tensor-wise RMS (default: True).
        normuon (bool): Use NorMuon update scaling (default: True).
        use_compile (bool): Use torch.compile on orthogonalization for faster execution (default: True).
        ortho_dtype (str): Data type for Newton-Schulz orthogonalization (default: None (torch.float32)).
        stochastic_fp (bool): Use stochastic rounding for half-precision (default: True).
    """

    PROJECTION_TYPES = ['stereographic', 'gnomonic', 'hyperbolic']

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        projection: Literal['stereographic', 'gnomonic', 'hyperbolic'] = 'gnomonic',
        input_norm: bool = True,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: Optional[str] = None,
        stochastic_fp: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if projection not in self.PROJECTION_TYPES:
            raise ValueError(f"Invalid projection type: {projection}. Choose from {self.PROJECTION_TYPES}")
        
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

        if ortho_dtype is None:
            ortho_dtype = torch.float32
        elif isinstance(ortho_dtype, str):
            dtype_name = ortho_dtype.split('.')[-1]
            ortho_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            projection=projection,
            input_norm=input_norm,
            normuon=normuon,
            use_compile=use_compile,
            ortho_dtype=ortho_dtype,
            stochastic_fp=stochastic_fp,
            sync_chunk_size = sync_chunk_size,
            state_storage_dtype = final_dtype,
            state_storage_device = state_storage_device,
        )
        self.ortho_func = torch.compile(orthogonalize, mode="default") if use_compile else orthogonalize
        super(ProjectiveAdam, self).__init__(params, defaults)

    # =========================================================================
    # Projection Functions, yeah i gemini'd these functions, yeah i abuse my $20 google subscription, yeah i love experimental and maybe wonky math.
    # =========================================================================

    @staticmethod
    def _stereographic_project(g: torch.Tensor, eps: float):
        """
        Element-wise Stereographic Projection from South Pole.
        Maps each element g_i -> (y_i, z_i) on a 2D circle.
        
        proj(g_i) = (2*g_i / (g_i^2 + 1), (g_i^2 - 1) / (g_i^2 + 1))
        Returns: (Y, z) where both are tensor-shaped.
        """
        g_sq = g.pow(2)
        denom = g_sq + 1.0
        y = (2.0 * g) / denom
        z = (g_sq - 1.0) / denom
        return y, z

    @staticmethod
    def _stereographic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float):
        """
        Inverse Stereographic Projection.
        Maps S^n -> R^n.
        
        inv(Y, z) = Y / (1 - z)
        """
        return y / (1.0 - z).clamp_min(eps)

    @staticmethod
    def _gnomonic_project(g: torch.Tensor, eps: float):
        """
        Element-wise Gnomonic (Central) Projection.
        Maps each element g_i -> (y_i, z_i) on the hemisphere.
        
        proj(g_i) = (g_i / sqrt(1 + g_i^2), 1 / sqrt(1 + g_i^2))
        Returns: (Y, z) where both are tensor-shaped.
        """
        g_sq = g.pow(2)
        inv_sqrt = torch.rsqrt(1.0 + g_sq)
        y = g * inv_sqrt
        z = inv_sqrt
        return y, z

    @staticmethod
    def _gnomonic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float):
        """
        Inverse Gnomonic Projection.
        Maps Hemisphere -> R^n.
        
        inv(Y, z) = Y / z
        """
        return y / z.clamp_min(eps)

    @staticmethod
    def _hyperbolic_project(g: torch.Tensor, eps: float):
        """
        Element-wise Hyperbolic (Poincaré) Projection.
        Maps each element g_i -> tanh(|g_i|) * sign(g_i).
        
        proj(g_i) = tanh(g_i) (element-wise tanh)
        Returns: (Y, z) where Y is the mapped point and z is |tanh(g_i)|.
        """
        y = torch.tanh(g)
        z = y.abs()  # Track magnitude in ball space
        return y, z

    @staticmethod
    def _hyperbolic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float):
        """
        Element-wise Inverse Hyperbolic Projection.
        Maps D^1 -> R^1 using arctanh.
        
        inv(y_i) = arctanh(y_i)
        """
        # Clamp y to be in (-1, 1) to avoid atanh singularity
        y_clamped = y.clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.atanh(y_clamped)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            projection = group['projection']
            stochastic_fp = group['stochastic_fp']

            # Select projection functions
            if projection == 'stereographic':
                project_fn = self._stereographic_project
                inverse_fn = self._stereographic_inverse
            elif projection == 'gnomonic':
                project_fn = self._gnomonic_project
                inverse_fn = self._gnomonic_inverse
            elif projection == 'hyperbolic':
                project_fn = self._hyperbolic_project
                inverse_fn = self._hyperbolic_inverse
            else:
                raise ValueError(f"Unknown projection: {projection}")

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                device = p.device

                grad = p.grad.data


                if grad.is_sparse:
                    raise RuntimeError('ProjectiveAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if self.state_storage_device == "cpu":

                        state["exp_avg_y"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device,
                            memory_format=torch.preserve_format
                        ).pin_memory()

                        state["exp_avg_z"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device,
                            memory_format=torch.preserve_format
                        ).pin_memory()

                        if p.ndim >= 1 and group["normuon"]:
                            grad_2d = reshape_to_2d(grad)
                            state['normuon_second_momentum'] = torch.zeros(grad_2d.shape[0], 1, 
                                                                           dtype=self.state_storage_dtype, 
                                                                           device=self.state_storage_device).pin_memory()
                    else:
                        state["exp_avg_y"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device,
                            memory_format=torch.preserve_format
                        )

                        state["exp_avg_z"] = torch.zeros_like(
                            p.data, 
                            dtype=self.state_storage_dtype, 
                            device=self.state_storage_device,
                            memory_format=torch.preserve_format
                        )

                        if p.ndim >= 1 and group["normuon"]:
                            grad_2d = reshape_to_2d(grad)
                            state['normuon_second_momentum'] = torch.zeros(grad_2d.shape[0], 1, 
                                                                           dtype=self.state_storage_dtype, 
                                                                           device=self.state_storage_device)

                state['step'] += 1
                step = state['step']

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                exp_avg_y = state["exp_avg_y"].detach().to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                exp_avg_z = state["exp_avg_z"].detach().to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                grad = grad.detach().to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.detach().to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                if p_fp32.ndim >= 1 and group["normuon"]:
                    normuon_second_momentum  = state["normuon_second_momentum"].detach().to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                    )

                # RMS Normalization & early clamping to prevent NaN or inf
                if p_fp32.ndim >= 1 and group["input_norm"]:
                    grad_work_2d = reshape_to_2d(grad)
                    grad_work_2d.div_(grad_work_2d.pow(2).mean(dim=-1, keepdim=True).sqrt_().clamp_min_(eps)).clamp_(-step, step)
                    grad = grad_work_2d.view_as(p_fp32)
                else:
                    grad.div_(grad.pow(2).mean().sqrt_().clamp_min_(eps)).clamp_(-step, step)

                # Projection
                y, z = project_fn(grad, eps)

                # EMA Updates
                exp_avg_y.mul_(beta1).add_(y, alpha=1 - beta1)
                exp_avg_z.mul_(beta2).add_(z, alpha=1 - beta2)

                # Inverse Projection with Bias Correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                curr_y = exp_avg_y / bias_correction1
                curr_z = exp_avg_z / bias_correction2

                update = inverse_fn(curr_y, curr_z, eps)

                if p_fp32.ndim >= 1:
                    full_step_2d = reshape_to_2d(update)
                    
                    # Newton-Schulz Orthogonalization
                    Q = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                    # NorMuon update & re-norm
                    if group["normuon"]:
                        vnorm = Q.norm(dim=(-2, -1), keepdim=True)

                        v_mean = torch.mean(Q * Q, dim=-1, keepdim=True)
                        normuon_second_momentum.lerp_(v_mean, 1 - beta2)
                        step_size = normuon_second_momentum.div(bias_correction2).sqrt().clamp_min_(eps)
                        Q.div_(step_size)

                        vnorm_new = Q.norm(dim=(-2, -1), keepdim=True)
                        Q = Q * (vnorm / vnorm_new.clamp_min(eps))

                    final_step = Q.view_as(p_fp32)

                    # Since final_step (Q) sums to 1, this functionally re-scales it back to the pre-ortho update's magnitude
                    scale_factor = (update * final_step).sum()
                    final_step.mul_(scale_factor)
                else:
                    final_step = update

                # Cautious masking
                scale_factor_mask = (grad * final_step > 0).to(final_step.dtype)
                mask_mean = scale_factor_mask.mean().clamp_min_(1e-3)
                scale_factor_mask.div_(mask_mean)
                final_step.mul_(scale_factor_mask)

                # Apply Update
                if weight_decay != 0:
                    p_fp32.add_(p_fp32, alpha=-lr * weight_decay)
                
                p_fp32.add_(final_step, alpha=-lr)


                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16 and stochastic_fp:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    # Original GPU path
                    if p.dtype == torch.bfloat16 and stochastic_fp:
                        copy_stochastic_(p, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

                # Store State
                if self.state_storage_dtype == torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(state["exp_avg_y"], exp_avg_y)
                    copy_stochastic_(state["exp_avg_z"], exp_avg_z)
                    if p.ndim >= 1 and group["normuon"]:
                        copy_stochastic_(state["normuon_second_momentum"], normuon_second_momentum)
                else:
                    state["exp_avg_y"].copy_(exp_avg_y, non_blocking=True)
                    state["exp_avg_z"].copy_(exp_avg_z, non_blocking=True)
                    if p.ndim >= 1 and group["normuon"]:
                        copy_stochastic_(state["normuon_second_momentum"], normuon_second_momentum)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss