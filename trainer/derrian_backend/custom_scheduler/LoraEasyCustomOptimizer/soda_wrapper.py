import torch
import torch.optim
from .utils import (clean_dict_params)
from typing import Union, Iterable, Dict, Any
from typing_extensions import TypeAlias
from .utils import copy_stochastic_

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class SODAWrapper(torch.optim.Optimizer):
    r"""
    OBS: Do not use weight decay for the underlying optimizer.
    """
    def __init__(self, 
                 params: ParamsT,
                 base_optimizer : torch.optim.Optimizer,
                 **kwargs,
                 ):
        
        clean_kwargs = clean_dict_params(base_optimizer, kwargs, wrapped=True)
        self.base_optimizer = base_optimizer(params, **clean_kwargs)

        for group in self.base_optimizer.param_groups:
            if 'soda_step' not in group:
                group['soda_step'] = 0

    def add_param_group(self, param_group):
        if 'soda_step' not in param_group:
            param_group['soda_step'] = 0
        return self.base_optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        for group in self.base_optimizer.param_groups:
            if 'soda_step' not in group:
                group['soda_step'] = 0

    def state_dict(self):
        return self.base_optimizer.state_dict()
        
    def zero_grad(self, set_to_none=True):
        return self.base_optimizer.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Keep a local snapshot before base.step(); do not mutate optimizer state yet.
        prev_map = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                prev_map[p] = torch.clone(p, memory_format=torch.preserve_format)

        self.base_optimizer.step()

        for group in self.param_groups:
            soda_step = group['soda_step']
            for p in group['params']:
                if p not in prev_map:
                    continue
                state = self.state[p]
                prev = prev_map[p].to(dtype=torch.float32)

                if 'soda_z0' not in state:
                    state['soda_z0'] = torch.clone(prev, memory_format=torch.preserve_format)

                soda_z = state['soda_z0'].to(dtype=torch.float32)
                soda_z = soda_z.add((soda_step+2) * (p - prev))
                
                x = prev * (1 - 1/(soda_step+2)) + soda_z * (1/(soda_step+2))
                if p.dtype in {torch.bfloat16}:
                    copy_stochastic_(p, x)
                else:
                    p.copy_(x)
            
            group['soda_step'] = soda_step+1

        return loss
