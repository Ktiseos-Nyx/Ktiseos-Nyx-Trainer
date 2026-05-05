import torch
from ramtorch.helpers import replace_linear_with_ramtorch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

def apply_ramtorch_to_module(module: torch.nn.Module|None, 
                             name: str, 
                             device: torch.device|str = "cuda",
                             dtype: torch.dtype = None) -> torch.nn.Module|None:
    """Apply RamTorch to a module if it's a valid torch.nn.Module."""
    if isinstance(module, torch.nn.Module):
        module = replace_linear_with_ramtorch(module, device, dtype)
        logger.info(f"RamTorch applied to {name}.")
    return module