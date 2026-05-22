import torch
import torch.optim as optim
import math
import warnings
from typing import Tuple, Dict, Any

from .utils import copy_stochastic_


class AdaGC(optim.Optimizer):
    """
    Implements AdaGC (Adaptive Gradient Clipping based on Local Gradient Norm)
    integrated with AdamW, with optional stochastic rounding and torch.compile support.
    Manages tensors to minimize re-creation, using pre-allocated buffers and scalar tensors.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty)
            (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of Adam
            (default: False)
        Tstart (int, optional): number of warm-up steps using global clipping.
            Set to 0 to disable warm-up. (default: 100)
        Aabs (float, optional): absolute global clipping threshold used during
            warm-up (default: 1.0)
        Arel (float, optional): relative local clipping threshold used in AdaGC
            phase (default: 1.05)
        beta_ema (float, optional): smoothing coefficient for the EMA of
            clipped gradient norms (gamma) (default: 0.98)
        eps_ema (float, optional): term added to the gamma EMA denominator
            to improve numerical stability (default: 1e-6).
        stochastic_rounding (bool, optional): whether to apply stochastic
            rounding when writing back to low-precision parameter and state
            tensors after float32 calculations. Calculations are always done in
            float32 if the parameter's original dtype is not float32.
            (default: False)
        compile_step (bool, optional): whether to torch.compile the core
            optimization step logic (`_single_param_step_fp32`) with
            `fullgraph=True, dynamic=False`. (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, Tstart=100, Aabs=1.0,
                 Arel=1.05, beta_ema=0.98, eps_ema=1e-6,
                 stochastic_rounding=False, compile_step=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 <= Tstart:
            raise ValueError("Invalid Tstart value: {}".format(Tstart))
        if not 0.0 <= Aabs:
             raise ValueError("Invalid Aabs value: {}".format(Aabs))
        if not 0.0 <= Arel:
             raise ValueError("Invalid Arel value: {}".format(Arel))
        if not 0.0 <= beta_ema < 1.0:
            raise ValueError("Invalid beta_ema value: {}".format(beta_ema))
        if not 0.0 <= eps_ema:
            raise ValueError("Invalid eps_ema value: {}".format(eps_ema))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        Tstart=Tstart, Aabs=Aabs, Arel=Arel,
                        beta_ema=beta_ema, eps_ema=eps_ema,
                        stochastic_rounding=stochastic_rounding,
                        compile_step=compile_step)
        super(AdaGC, self).__init__(params, defaults)

        # Global step counter (Python int)
        self._global_step = 0

        # Tensors to hold scalar values for the compiled step function, per device
        # Use dicts keyed by device
        self._scalar_tensors_float: Dict[torch.device, Dict[str, torch.Tensor]] = {}
        self._scalar_tensors_int64: Dict[torch.device, Dict[str, torch.Tensor]] = {}
        self._scalar_tensors_bool: Dict[torch.device, Dict[str, torch.Tensor]] = {}

        # Global clip factor tensor (FP32 scalar)
        self._global_clip_factor_fp32 = torch.tensor(1.0, dtype=torch.float32, device='cpu')

        # Compile the core step logic if requested
        self._single_param_step_callable = self._single_param_step_fp32 # Default to non-compiled
        if compile_step:
             try:
                # Compile the FP32 version of the step function
                with torch._dynamo.utils.disable_cache_limit():
                    self._single_param_step_callable = torch.compile(
                        self._single_param_step_fp32, fullgraph=True, dynamic=False
                        )
                warnings.warn("Core optimization step compiled with torch.compile(fullgraph=True, dynamic=False, mode=\"reduce-overhead\").")
             except Exception as e:
                 warnings.warn(f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. Falling back to non-compiled step.")
                 self._single_param_step_callable = self._single_param_step_fp32 # Fallback

    # @staticmethod removed because it needs access to self._allocate_scalar_tensors
    # Let's make it a regular method that takes parameters as arguments or accessess state via p
    # No, the point was to compile a static method. Keep it static, and pass everything it needs.

    @staticmethod
    # Using **kwargs allows for easier addition of future scalar tensor inputs
    def _single_param_step_fp32(
        p_data_fp32: torch.Tensor,
        grad_fp32: torch.Tensor,
        exp_avg_fp32: torch.Tensor,
        exp_avg_sq_fp32: torch.Tensor,
        gamma_fp32: torch.Tensor, # Gamma is always FP32 state (scalar tensor)
        max_exp_avg_sq_fp32: torch.Tensor, # Max_exp_avg_sq is always FP32 (tensor or placeholder)
        final_clip_factor_fp32: torch.Tensor, # Final clip factor (scalar tensor)
        # Scalar hyperparameters and step count as tensors
        step_t: torch.Tensor, # Parameter step (scalar int tensor)
        lr_t: torch.Tensor, # Scalar float tensor
        beta1_t: torch.Tensor, # Scalar float tensor
        beta2_t: torch.Tensor, # Scalar float tensor
        eps_t: torch.Tensor, # Scalar float tensor
        weight_decay_t: torch.Tensor, # Scalar float tensor
        beta_ema_t: torch.Tensor, # Scalar float tensor
        amsgrad: bool # Scalar bool tensor
    ):
        """
        Core optimization logic for a single parameter, designed to run in FP32.
        All inputs are expected to be tensors.
        Updates inputs tensors in-place.
        """
        with torch.no_grad():
            # --- Apply Final Clipping Factor ---
            # The `final_clip_factor_fp32` is already determined outside
            # Always apply this factor.
            clipped_grad_fp32 = grad_fp32 # Operation modifies in-place
            clipped_grad_fp32.mul_(final_clip_factor_fp32)


            # --- EMA Update (using norm of the *clipped* gradient) ---
            # Calculate the norm of the *clipped* gradient in FP32
            clipped_param_norm_fp32 = torch.linalg.norm(clipped_grad_fp32)
            # Update gamma state (which is an FP32 scalar tensor) in-place
            gamma_fp32.mul_(beta_ema_t).add_(clipped_param_norm_fp32, alpha=1.0 - beta_ema_t)


            # --- Standard AdamW Update ---
            # Use the calculation tensors (FP32) and scalar tensors for updates
            exp_avg_fp32.mul_(beta1_t).add_(clipped_grad_fp32, alpha=1.0 - beta1_t)
            exp_avg_sq_fp32.mul_(beta2_t).addcmul_(clipped_grad_fp32, clipped_grad_fp32, value=1.0 - beta2_t) # Use clipped gradient (FP32) and 1.0 - tensor

            # Use torch.where for amsgrad logic - requires scalar bool tensor to work
            # Need to prepare both amsgrad and non-amsgrad denominator calculations
            if amsgrad:
                denom_fp32 = max_exp_avg_sq_fp32.sqrt().add_(eps_t)
            else:
                denom_fp32 = exp_avg_sq_fp32.sqrt().add_(eps_t)

            # AdamW bias correction factors (use scalar tensor power)
            bias_correction1_t = 1.0 - beta1_t.pow(step_t.float()) # Pow requires float
            bias_correction2_t = 1.0 - beta2_t.pow(step_t.float())
            # Apply bias correction to the denominator (FP32)
            denom_fp32.div_(torch.sqrt(bias_correction2_t)) # torch.sqrt works on tensors

            # Calculate step_size (use scalar tensor division)
            step_size_t = lr_t / bias_correction1_t

            # Parameter update (on the calculation tensor - FP32)
            # Decoupled weight decay - use torch.where for static graph
            p_data_fp32.addcdiv_(exp_avg_fp32, denom_fp32, value=-step_size_t)
            # Add weight decay term using torch.where based on weight_decay_t
            p_data_fp32.add_(
                p_data_fp32,
                alpha=torch.where(weight_decay_t != 0.0, -weight_decay_t * lr_t, torch.tensor(0.0, dtype=torch.float32, device=p_data_fp32.device))
            )


            # Updates happened in-place on the input FP32 tensors.


    def _allocate_scalar_tensors(self, device: torch.device, group_defaults: Dict[str, Any]):
         """Allocates or retrieves scalar tensors for a given device."""
         if device not in self._scalar_tensors_float:
            self._scalar_tensors_float[device] = {}
            self._scalar_tensors_int64[device] = {}
            self._scalar_tensors_bool[device] = {}

            # Initialize float tensors
            self._scalar_tensors_float[device]['lr_t'] = torch.tensor(group_defaults['lr'], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['beta1_t'] = torch.tensor(group_defaults['betas'][0], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['beta2_t'] = torch.tensor(group_defaults['betas'][1], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['eps_t'] = torch.tensor(group_defaults['eps'], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['weight_decay_t'] = torch.tensor(group_defaults['weight_decay'], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['beta_ema_t'] = torch.tensor(group_defaults['beta_ema'], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['eps_ema_t'] = torch.tensor(group_defaults['eps_ema'], dtype=torch.float32, device=device)
            self._scalar_tensors_float[device]['Arel_t'] = torch.tensor(group_defaults['Arel'], dtype=torch.float32, device=device)

            # Initialize int64 tensors
            self._scalar_tensors_int64[device]['step_t'] = torch.tensor(0, dtype=torch.int64, device=device) # Parameter step
            self._scalar_tensors_int64[device]['global_step_t'] = torch.tensor(0, dtype=torch.int64, device=device) # Global step
            self._scalar_tensors_int64[device]['Tstart_t'] = torch.tensor(group_defaults['Tstart'], dtype=torch.int64, device=device)

            # Initialize bool tensors
            self._scalar_tensors_bool[device]['amsgrad_t_bool'] = torch.tensor(group_defaults['amsgrad'], dtype=torch.bool, device=device)

         return (self._scalar_tensors_float[device],
                 self._scalar_tensors_int64[device],
                 self._scalar_tensors_bool[device])


    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1 # Python int global step counter

        # --- Global Clipping Calculation (outside compiled step) ---
        # Calculate global norm and clip factor in FP32 if needed.
        # This result is passed to the compiled function.
        # Find a device to perform global norm calculation on
        global_norm_device = 'cpu'
        has_grad = False
        for group in self.param_groups:
             for p in group['params']:
                 if p.grad is not None:
                      has_grad = True
                      global_norm_device = p.grad.data.device # Use device of first grad found
                      break
             if global_norm_device != 'cpu': break # Found a device

        if self._global_step <= self.defaults['Tstart'] and self.defaults['Tstart'] > 0 and has_grad:
            # Calculate total squared global norm of gradients in FP32 on the selected device
            global_norm_sq_fp32 = torch.tensor(0.0, dtype=torch.float32, device=global_norm_device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        global_norm_sq_fp32.add_(p.grad.data.float().pow(2).sum()) # Ensure sum is in FP32

            if global_norm_sq_fp32 > 0: # Avoid division by zero
                 global_norm_fp32 = torch.sqrt(global_norm_sq_fp32)
                 # Global clip factor computed once in FP32
                 # Ensure eps is a float or FP32 tensor when adding
                 eps_fp32 = torch.tensor(self.defaults['eps'], dtype=torch.float32, device=global_norm_device)
                 self._global_clip_factor_fp32 = torch.tensor(self.defaults['Aabs'], dtype=torch.float32, device=global_norm_device) / (global_norm_fp32 + eps_fp32)
                 self._global_clip_factor_fp32 = torch.min(self._global_clip_factor_fp32, torch.tensor(1.0, device=global_norm_device, dtype=torch.float32))
            else:
                 # If global norm is 0, no clipping is needed, factor is 1.0
                 self._global_clip_factor_fp32 = torch.tensor(1.0, device=global_norm_device, dtype=torch.float32) # Put on device
        else:
             # If not in warm-up or no grads, global clip factor is 1.0 (ensure it's on a device if possible)
             device = global_norm_device # Use the device found earlier, or 'cpu'
             self._global_clip_factor_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)


        # --- AdaGC & AdamW Logic per parameter ---
        for group in self.param_groups:
            # Get group hyperparameters (Python scalars)
            # Access defaults directly to avoid recreating dict each step if needed,
            # or just pull the values out as they are few.
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            lr = group['lr']
            stochastic_rounding = group['stochastic_rounding']
            Tstart = group['Tstart']
            beta_ema = group['beta_ema']
            eps_ema = group['eps_ema']
            Arel = group['Arel']


            # Get the step function to use (compiled or original)
            step_fn = self._single_param_step_callable


            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the parameter's device
                device = p.data.device

                # State initialization (including FP32 buffers)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0 # Python int step
                    # exp_avg and exp_avg_sq match parameter dtype
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # gamma is always FP32 scalar state
                    state['gamma'] = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if amsgrad:
                        # max_exp_avg_sq matches parameter dtype initially
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    # Allocate FP32 buffers in state if original dtype is not FP32
                    if p.data.dtype != torch.float32:
                        state['_p_data_fp32_buf'] = torch.zeros_like(p.data, dtype=torch.float32, memory_format=torch.preserve_format)
                        state['_grad_fp32_buf'] = torch.zeros_like(p.grad.data, dtype=torch.float32, memory_format=torch.preserve_format)
                        state['_exp_avg_fp32_buf'] = torch.zeros_like(state['exp_avg'], dtype=torch.float32, memory_format=torch.preserve_format)
                        state['_exp_avg_sq_fp32_buf'] = torch.zeros_like(state['exp_avg_sq'], dtype=torch.float32, memory_format=torch.preserve_format)
                        if amsgrad:
                             state['_max_exp_avg_sq_fp32_buf'] = torch.zeros_like(state['max_exp_avg_sq'], dtype=torch.float32, memory_format=torch.preserve_format)
                        else:
                             # Placeholder buffer for amsgrad=False, needed for consistent logic flow
                             state['_max_exp_avg_sq_fp32_buf'] = torch.tensor([], dtype=torch.float32, device=device)

                # Update parameter step counter (Python int)
                state['step'] += 1

                # Allocate or get scalar tensors for this device and update their values
                scalar_float, scalar_int64, scalar_bool = self._allocate_scalar_tensors(device, group)
                scalar_float['lr_t'].fill_(lr)
                scalar_float['beta1_t'].fill_(beta1)
                scalar_float['beta2_t'].fill_(beta2)
                scalar_float['eps_t'].fill_(eps)
                scalar_float['weight_decay_t'].fill_(weight_decay)
                scalar_float['beta_ema_t'].fill_(beta_ema)
                scalar_float['eps_ema_t'].fill_(eps_ema)
                scalar_float['Arel_t'].fill_(Arel)
                scalar_int64['step_t'].fill_(self.state[p]['step']) # Use the parameter's internal step counter
                scalar_int64['global_step_t'].fill_(self._global_step) # Use the optimizer's global step counter
                scalar_int64['Tstart_t'].fill_(Tstart)
                scalar_bool['amsgrad_t_bool'].fill_(amsgrad)

                original_dtype = p.data.dtype
                is_original_fp32 = (original_dtype == torch.float32)

                # --- Prepare FP32 calculation tensors (use buffers) ---
                if not is_original_fp32:
                     # Copy low-precision data into FP32 buffers
                     state['_p_data_fp32_buf'].copy_(p.data)
                     state['_grad_fp32_buf'].copy_(p.grad.data)
                     state['_exp_avg_fp32_buf'].copy_(state['exp_avg'])
                     state['_exp_avg_sq_fp32_buf'].copy_(state['exp_avg_sq'])
                     if amsgrad:
                          state['_max_exp_avg_sq_fp32_buf'].copy_(state['max_exp_avg_sq'])

                     p_data_calc = state['_p_data_fp32_buf']
                     grad_calc = state['_grad_fp32_buf']
                     exp_avg_calc = state['_exp_avg_fp32_buf']
                     exp_avg_sq_calc = state['_exp_avg_sq_fp32_buf']
                     gamma_calc = state['gamma'] # gamma is always float32
                     max_exp_avg_sq_calc = state['_max_exp_avg_sq_fp32_buf'] # Buffer or placeholder

                else: # Original dtype is FP32
                     # Use original tensors directly as they are FP32
                     p_data_calc = p.data
                     grad_calc = p.grad.data
                     exp_avg_calc = state['exp_avg']
                     exp_avg_sq_calc = state['exp_avg_sq']
                     gamma_calc = state['gamma'] # gamma is always FP32
                     if amsgrad:
                          max_exp_avg_sq_calc = state['max_exp_avg_sq']
                     else:
                          # Still need a placeholder for the function signature
                          max_exp_avg_sq_calc = torch.tensor([], dtype=torch.float32, device=device)


                # --- Determine the FINAL clipping factor for this parameter (outside compiled step) ---
                # This logic remains outside to avoid tensor-based control flow in compiled graph.
                if self._global_step <= Tstart and Tstart > 0:
                    # Use the pre-calculated global clip factor.
                    # It was already placed on a device during the global norm calculation,
                    # move it to the current parameter's device if needed.
                    final_clip_factor_fp32 = self._global_clip_factor_fp32.to(device)
                else:
                    # Calculate the local AdaGC scaling factor in FP32
                    # Norm of the raw gradient (grad_calc is FP32)
                    param_norm_fp32 = torch.linalg.norm(grad_calc)

                    # Get previous EMA gamma (gamma_calc is FP32) and calculate adaptive threshold (FP32)
                    prev_gamma_fp32 = gamma_calc
                    Arel_t = scalar_float['Arel_t'] # Get the scalar tensor
                    eps_ema_t = scalar_float['eps_ema_t'] # Get the scalar tensor
                    adaptive_threshold_fp32 = Arel_t * (prev_gamma_fp32 + eps_ema_t)

                    # Calculate the static clipping factor: min(1.0, threshold / norm)
                    # Add eps_t to the denominator of the ratio to prevent division by zero
                    eps_t = scalar_float['eps_t'] # Get the scalar tensor
                    ratio_fp32 = adaptive_threshold_fp32 / (param_norm_fp32 + eps_t)
                    # Create 1.0 tensor on the correct device for torch.min
                    one_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)
                    final_clip_factor_fp32 = torch.min(one_fp32, ratio_fp32)

                # --- Call the core step function (potentially compiled) ---
                # Always call the FP32 step function with the FP32 calculation tensors and scalar tensors
                with torch._dynamo.utils.disable_cache_limit():
                    step_fn(
                        p_data_fp32=p_data_calc,
                        grad_fp32=grad_calc,
                        exp_avg_fp32=exp_avg_calc,
                        exp_avg_sq_fp32=exp_avg_sq_calc,
                        gamma_fp32=gamma_calc, # gamma is always FP32
                        max_exp_avg_sq_fp32=max_exp_avg_sq_calc, # will be FP32 tensor or placeholder
                        final_clip_factor_fp32=final_clip_factor_fp32, # Pass the pre-calculated final factor
                        # Scalar tensor inputs (retrieve from allocated tensors)
                        step_t=scalar_int64['step_t'],
                        lr_t=scalar_float['lr_t'],
                        beta1_t=scalar_float['beta1_t'],
                        beta2_t=scalar_float['beta2_t'],
                        eps_t=scalar_float['eps_t'],
                        weight_decay_t=scalar_float['weight_decay_t'],
                        beta_ema_t=scalar_float['beta_ema_t'],
                        amsgrad=amsgrad
                    )

                # --- Write Back to Original Tensors with Rounding ---
                if not is_original_fp32: # If original dtype was low-precision
                    if stochastic_rounding:
                        # Use stochastic rounding to copy FP32 results back to original low-precision tensors
                        copy_stochastic_(p.data, p_data_calc) # p.data is original low-prec tensor
                        copy_stochastic_(state['exp_avg'], exp_avg_calc) # state['exp_avg'] is original low-prec tensor
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq_calc) # state['exp_avg_sq'] is original low-prec tensor
                        if amsgrad:
                            copy_stochastic_(state['max_exp_avg_sq'], max_exp_avg_sq_calc) # state['max_exp_avg_sq'] is original low-prec tensor
                        # gamma was updated in-place as FP32, no copy needed
                    else:
                        # Standard copy (effectively casting FP32 result back using default rounding)
                        p.data.copy_(p_data_calc)
                        state['exp_avg'].copy_(exp_avg_calc)
                        state['exp_avg_sq'].copy_(exp_avg_sq_calc)
                        if amsgrad:
                             state['max_exp_avg_sq'].copy_(max_exp_avg_sq_calc)
                        # gamma was updated in-place as FP32
                # Else: original dtype was FP32, updates were in-place on original tensors, no copy needed


        return loss
