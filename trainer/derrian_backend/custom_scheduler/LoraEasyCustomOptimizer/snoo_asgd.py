import torch
from torch.optim import Optimizer

"""
Usage: Wrap your optimizer of choice with ASGD. The swapping of averaged parameters
for evaluation is handled automatically.

Example 1 (Powered Averaging):
# Uses a decay of 1 / (n_averaged ^ alpha)
optimizer = ASGD(base_optimizer, alpha=0.5, t0=0)

Example 2 (Exponential Moving Average):
# Uses a constant decay factor (lambda). This is often more stable.
optimizer = ASGD(base_optimizer, lambd=0.001, t0=0)
"""

class SNOO_ASGD(Optimizer):
    """
    Implements Averaged Stochastic Gradient Descent (Polyak-Ruppert averaging) as a wrapper
    with automatic parameter swapping for evaluation.

    This optimizer offers two averaging strategies:
    1.  Powered Averaging (using `alpha`): The averaging step size decays as 1/(n^alpha).
        This is closer to the original paper's formulation. alpha=1.0 is a standard
        running average.
    2.  Exponential Moving Average (EMA, using `lambd`): The averaging uses a constant
        decay factor. This is often more robust in practice and prevents the average
        from "freezing" late in training. If `lambd` is specified, it overrides `alpha`.

    The wrapper automatically swaps the model's parameters with the averaged parameters after each
    `step()`, making the model ready for evaluation. It swaps them back at the beginning of the
    next training iteration, triggered by `zero_grad()`.
    """

    @torch.no_grad()
    def __init__(self, optimizer, lr=1.0, alpha: float = 0.75, lambd: float = None, t0: int = 0, accelerate_k: int = 20, accelerate_lr: float = 0.5, accelerate_momentum: float = 0.5, accelerate_nesterov=True) -> None:
        """
        Args:
            optimizer (torch.optim.Optimizer): The base optimizer to be wrapped (e.g., SGD).
            alpha (float, optional): Power for the averaging step size decay (1/n^alpha).
                Defaults to 0.75. Is ignored if `lambd` is not None.
            lambd (float, optional): The decay factor for the Exponential Moving Average.
                If provided, this method is used instead of powered averaging. Defaults to None.
                (Note: 'lambd' is used to avoid conflict with Python's lambda keyword).
            t0 (int, optional): The iteration number to start averaging from. Defaults to 0.
        """
        self.optimizer = optimizer
        self.lr = lr
        self.alpha = alpha
        self.lambd = lambd
        self.t0 = t0
        self.accelerate_k = accelerate_k
        self.accelerate_lr = accelerate_lr
        self.accelerate_momentum = accelerate_momentum
        self.accelerate_nesterov = accelerate_nesterov
        self.current_step = 0
        self.n_averaged = 0
        self.model_params = None
        self.averaged_params_cpu = None
        self.non_averaged_params_cpu = None # Buffer to store original params during eval mode
        self.nesterov_params_cpu = None # Buffer to store nesterov-accelerated params (from SNOO)
        self.nesterov_buffer_cpu = None # Buffer to store nesterov-accelerated params (from SNOO)
        self.is_swapped = False # Tracks if the model params are currently the averaged ones

        if self.optimizer.param_groups:
            self.param_groups = self.optimizer.param_groups

    @torch.no_grad()
    def _initialize_state(self):
        """Initializes the buffers for averaged and non-averaged parameters on the CPU."""
        params = [p for pg in self.optimizer.param_groups for p in pg['params'] if isinstance(p, torch.Tensor)]
        if not params: return

        self.model_params = list(params)
        self.averaged_params_cpu = [p.clone().to('cpu') for p in self.model_params]
        self.non_averaged_params_cpu = [p.clone().to('cpu') for p in self.model_params]
        self.nesterov_params_cpu = [p.clone().to('cpu') for p in self.model_params] if self.accelerate_k > 0 else None
        self.nesterov_buffer_cpu = [torch.zeros_like(p).to('cpu') for p in self.model_params] if self.accelerate_k > 0 else None
        self.param_groups = self.optimizer.param_groups
        del params

    @torch.no_grad()
    def step(self, closure=None):
        if self.averaged_params_cpu is None:
            if self.optimizer.param_groups: self._initialize_state()
            if self.averaged_params_cpu is None: return self.optimizer.step(closure)

        if self.is_swapped:
            self._swap_parameters()

        loss = self.optimizer.step(closure)
        self.current_step += 1

        if self.current_step >= self.t0:
            self.n_averaged += 1

            # Determine the decay factor for the update
            if self.lambd is not None:
                # Use constant decay for EMA
                decay = self.lambd
            else:
                # Use powered decay for standard averaging
                decay = self.lr / (self.n_averaged ** self.alpha)

            # Update the running average of parameters
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu):
                delta = p_gpu.data.to('cpu', non_blocking=True) - p_avg_cpu.data
                p_avg_cpu.data.add_(delta, alpha=decay)
        else:
            # Update the running average of parameters
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu):
                p_avg_cpu.data.copy_(p_gpu.data, non_blocking=True)

        # Update the Nesterov accelerated params
        if self.accelerate_k > 0:
            if self.current_step % self.accelerate_k == 0:
                for p_nesterov_cpu, buffer_nesterov_cpu, p_gpu in zip(self.nesterov_params_cpu, self.nesterov_buffer_cpu, self.model_params):
                    delta = p_gpu.data.to('cpu', non_blocking=True) - p_nesterov_cpu.data
                    if self.accelerate_nesterov:
                        buffer_nesterov_cpu.data.mul_(self.accelerate_momentum).add_(delta)
                        grad = buffer_nesterov_cpu.data.mul(self.accelerate_momentum).add_(delta).mul_(1. - self.accelerate_momentum).to(p_gpu.data.device, non_blocking=True)
                    else:
                        buffer_nesterov_cpu.data.lerp_(delta, weight=1. - self.accelerate_momentum)
                        grad = delta.lerp(buffer_nesterov_cpu.data, weight=self.accelerate_momentum).to(p_gpu.data.device, non_blocking=True)

                    p_nesterov_cpu.data.copy_(p_gpu.data, non_blocking=True)
                    p_gpu.data.add_(grad, alpha=self.accelerate_lr)
                    #print("acceleration:", buffer_nesterov_cpu.data)


        #print(self.model_params)

        if not self.is_swapped:
            self._swap_parameters()

        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        if self.is_swapped:
            self._swap_parameters()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _swap_parameters(self):
        if self.averaged_params_cpu is None: return

        if not self.is_swapped:
            for p_gpu, p_non_avg_cpu in zip(self.model_params, self.non_averaged_params_cpu):
                p_non_avg_cpu.copy_(p_gpu.data, non_blocking=True)
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu):
                p_gpu.copy_(p_avg_cpu.data, non_blocking=True)
            self.is_swapped = True
        else:
            for p_gpu, p_non_avg_cpu in zip(self.model_params, self.non_averaged_params_cpu):
                p_gpu.copy_(p_non_avg_cpu.data, non_blocking=True)
            self.is_swapped = False

    def state_dict(self):
        inner_state_dict = self.optimizer.state_dict()
        wrapper_state_dict = {
            't0': self.t0,
            'lr': self.lr,
            'alpha': self.alpha,
            'lambd': self.lambd,
            'accelerate_k': self.accelerate_k,
            'accelerate_lr': self.accelerate_lr,
            'accelerate_momentum': self.accelerate_momentum,
            'accelerate_nesterov': self.accelerate_nesterov,
            'current_step': self.current_step,
            'n_averaged': self.n_averaged,
            'averaged_params_cpu': self.averaged_params_cpu,
            'non_averaged_params_cpu': self.non_averaged_params_cpu,
            'nesterov_params_cpu': self.nesterov_params_cpu if self.accelerate_k > 0 else None,
            'nesterov_buffer_cpu': self.nesterov_buffer_cpu if self.accelerate_k > 0 else None,
            'is_swapped': self.is_swapped,
        }
        return {'inner_optimizer': inner_state_dict, 'asgd_wrapper': wrapper_state_dict}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['inner_optimizer'])
        wrapper_state = state_dict['asgd_wrapper']
        
        if self.averaged_params_cpu is None:
            if self.optimizer.param_groups: self._initialize_state()

        self.t0 = wrapper_state.get('t0', 0)
        self.lr = wrapper_state.get('lr', 1.0)
        self.alpha = wrapper_state.get('alpha', 0.75)
        self.lambd = wrapper_state.get('lambd', None)
        self.accelerate_k = wrapper_state.get('accelerate_k', 20)
        self.accelerate_lr = wrapper_state.get('accelerate_lr', 0.5)
        self.accelerate_momentum = wrapper_state.get('accelerate_momentum', 0.5)
        self.accelerate_nesterov = wrapper_state.get('accelerate_nesterov', True)
        self.current_step = wrapper_state['current_step']
        self.n_averaged = wrapper_state['n_averaged']
        self.averaged_params_cpu = wrapper_state['averaged_params_cpu']
        self.non_averaged_params_cpu = wrapper_state['non_averaged_params_cpu']
        self.nesterov_params_cpu = wrapper_state['nesterov_params_cpu']
        self.nesterov_buffer_cpu = wrapper_state['nesterov_buffer_cpu']
        self.is_swapped = wrapper_state['is_swapped']

        if self.is_swapped:
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu):
                p_gpu.copy_(p_avg_cpu.data, non_blocking=True)