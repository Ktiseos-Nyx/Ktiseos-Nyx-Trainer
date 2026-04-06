from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from ..utils.quant import QuantLinears, log_bypass, log_suspect
from ..logging import logger
from typing import Optional
import math

try:
    from ramtorch.modules.linear import CPUBouncingLinear
except ImportError:
    CPUBouncingLinear = type(None)

class AsyncTensorStreamer:
    def __init__(self, device):
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device=device)
        
        # RING BUFFER SETTINGS
        # Size 3 is safe: [Weight_Layer_N, Bias_Layer_N, Weight_Layer_N+1]
        # This prevents overwriting the weight currently being computed if the 
        # next layer starts transferring immediately.
        self.num_buffers = 3 
        self.idx = 0
        
        # We store (buffer, event) pairs. 
        # buffer: Holds the Tensor memory on GPU
        # event: Records when the Compute Stream is DONE using this buffer
        self.buffers = [None] * self.num_buffers
        self.compute_done_events = [torch.cuda.Event() for _ in range(self.num_buffers)]

    def transfer(self, tensor_cpu: torch.Tensor):
        # 1. Pin Memory (Crucial for Async)
        if not tensor_cpu.is_pinned():
            tensor_cpu = tensor_cpu.pin_memory()

        # 2. Select the next slot in the Ring Buffer
        slot_idx = self.idx
        self.idx = (self.idx + 1) % self.num_buffers
        
        ready_event = self.compute_done_events[slot_idx]
        
        # 3. SYNC: Wait for the PREVIOUS Compute cycle to finish with this specific slot
        # We cannot overwrite this slot if the GPU is still doing Math on the data previously stored here.
        # (For the first run, the event is unrecorded, so this is a no-op).
        self.transfer_stream.wait_event(ready_event)

        with torch.cuda.stream(self.transfer_stream):
            with torch.no_grad():
                # 4. TRANSFER / ALLOCATE
                # We use .to() which uses PyTorch's Caching Allocator. 
                # If self.buffers[slot_idx] existed, it goes back to the pool.
                # We don't manually hold .new_empty() anymore to allow dynamic resizing 
                # if layers have different shapes.
                gpu_tensor = tensor_cpu.to(self.device, non_blocking=True)
                
                # Keep a reference in our ring buffer list so Python doesn't GC it 
                # before the stream operation completes.
                self.buffers[slot_idx] = gpu_tensor
            
            # Record that transfer is finished
            transfer_finished_event = torch.cuda.Event()
            transfer_finished_event.record()

        # 5. SYNC: Tell the Compute Stream (Current Stream) to wait for transfer
        torch.cuda.current_stream().wait_event(transfer_finished_event)
        
        # 6. Mark usage
        # Record an event on the Compute Stream. 
        # The NEXT time we try to write to 'slot_idx', we will wait for this event.
        ready_event.record()
        
        return self.buffers[slot_idx]

# Global registry for multi-gpu support
_STREAMERS = {}

def transfer_ramtensor_to_device(tensor_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Args:
        tensor_id: Used for debugging/logging, but no longer used for memory allocation keys.
    """
    if not getattr(tensor_cpu, 'is_ramtorch', False):
        return tensor_cpu.to(device, non_blocking=True)
    
    if device.type == 'cpu':
        return tensor_cpu

    if device not in _STREAMERS:
        _STREAMERS[device] = AsyncTensorStreamer(device)
    
    return _STREAMERS[device].transfer(tensor_cpu)

class ModuleCustomSD(nn.Module):
    def __init__(self):
        super().__init__()
        self._register_load_state_dict_pre_hook(self.load_weight_prehook)
        self.register_load_state_dict_post_hook(self.load_weight_hook)

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def load_weight_hook(self, module, incompatible_keys):
        pass

    def custom_state_dict(self):
        return None

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        if (custom_sd := self.custom_state_dict()) is not None:
            for k, v in custom_sd.items():
                destination[f"{prefix}{k}"] = v
            return destination
        else:
            return super().state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )


class LycorisBaseModule(ModuleCustomSD):
    name: str
    dtype_tensor: torch.Tensor
    support_module = {}
    weight_list = []
    weight_list_det = []

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        rank_dropout_scale: bool = False,
        bypass_mode: bool = False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.not_supported = False
        self.grad_count = 0
        self.sum_grads = None
        self.sum_squared_grads = None

        self.is_ramtorch_org = isinstance(org_module, CPUBouncingLinear)
        if self.is_ramtorch_org:
            logger.info(f"RamTorch module detected: {lora_name}")

        self.module = type(org_module)
        if isinstance(org_module, (nn.Linear, CPUBouncingLinear)):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
            self.op = F.conv1d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
            self.op = F.conv2d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv3d):
            self.module_type = "conv3d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
            self.op = F.conv3d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.LayerNorm):
            self.module_type = "layernorm"
            self.shape = tuple(org_module.normalized_shape)
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.shape = (org_module.num_channels,)
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            self.not_supported = True
            self.module_type = "unknown"

        self.register_buffer("dtype_tensor", torch.tensor(0.0), persistent=False)

        self.is_quant = False
        if isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            self.is_quant = True
            bypass_mode = True
        if (
            isinstance(org_module, nn.Linear)
            and org_module.__class__.__name__ != "Linear"
        ):
            if bypass_mode is None:
                log_suspect()
                bypass_mode = True
            if bypass_mode == True:
                self.is_quant = True
        self.bypass_mode = bypass_mode
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        ## Dropout things
        # Since LoKr/LoHa/OFT/BOFT are hard to follow the rank_dropout definition from kohya
        # We redefine the dropout procedure here.
        # g(x) = WX + drop(Brank_drop(AX)) for LoCon(lora), bypass
        # g(x) = WX + drop(ΔWX) for any algo except LoCon(lora), bypass
        # g(x) = (W + Brank_drop(A))X for LoCon(lora), rebuid
        # g(x) = (W + rank_drop(ΔW))X for any algo except LoCon(lora), rebuild
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.rank_drop = (
            nn.Identity() if rank_dropout == 0 else nn.Dropout(rank_dropout)
        )

        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = [org_module]

        self.ggpo_sigma = ggpo_sigma
        self.ggpo_beta = ggpo_beta
        self.ggpo_conv = ggpo_conv
        self.ggpo_conv_weight_sample_size = ggpo_conv_weight_sample_size

    def _orthogonalize(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalizes the weight matrix using QR decomposition.
        This is only active during training if `use_orthogonal_weights` is True.
        """
        if not self.use_orthogonal_weights or not self.training:
            return weight_matrix

        # QR decomposition works on 2D matrices.
        # Conv weights can be > 2D, but LoKr factors are 2D.
        shape = weight_matrix.shape
        dimcount = len(shape)
        if dimcount == 0:
            return weight_matrix
        elif dimcount > 2:
            weight_matrix = weight_matrix.reshape(len(weight_matrix), -1) # Make 2D if conv or 1 dim
        elif dimcount < 2:
            weight_matrix = weight_matrix.reshape(1, -1) # Make 2D if conv or 1 dim
        
        # Upcast to fp32 for QR (bf16 CUDA kernel not implemented)
        orig_dtype = weight_matrix.dtype
        weight_matrix_fp32 = weight_matrix.to(torch.float32)

        # For matrices where rows >= cols, QR gives orthonormal columns.
        # For matrices where rows < cols, we transpose to make columns from rows,
        # apply QR, and transpose back. This results in orthonormal rows.
        rows, cols = weight_matrix.shape
        if rows >= cols:
            q, r = torch.linalg.qr(weight_matrix_fp32)
            weight_matrix_fp32 = q * torch.diag(r)
        else:
            q, r = torch.linalg.qr(weight_matrix_fp32.T)
            weight_matrix_fp32 = (q * torch.diag(r)).T
        return weight_matrix_fp32.to(orig_dtype).reshape(shape).contiguous()

    @classmethod
    def parametrize(cls, org_module, attr, *args, **kwargs):
        from .full import FullModule

        if cls is FullModule:
            raise RuntimeError("FullModule cannot be used for parametrize.")
        target_param = getattr(org_module, attr)
        kwargs["bypass_mode"] = False
        if target_param.dim() == 2:
            proxy_module = nn.Linear(
                target_param.shape[0], target_param.shape[1], bias=False
            )
            proxy_module.weight = target_param
        elif target_param.dim() > 2:
            module_type = [
                None,
                None,
                None,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                None,
                None,
            ][target_param.dim()]
            proxy_module = module_type(
                target_param.shape[0],
                target_param.shape[1],
                *target_param.shape[2:],
                bias=False,
            )
            proxy_module.weight = target_param
        module_obj = cls("", proxy_module, *args, **kwargs)
        module_obj.forward = module_obj.parametrize_forward
        module_obj.to(target_param)
        parametrize.register_parametrization(org_module, attr, module_obj)
        return module_obj

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return any(f"{lora_name}.{k}" in state_dict for k in cls.weight_list_det)

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        return [state_dict.get(f"{lora_name}.{k}", None) for k in cls.weight_list]

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, *weights):
        raise NotImplementedError

    @property
    def dtype(self):
        return self.dtype_tensor.dtype

    @property
    def device(self):
        return self.dtype_tensor.device

    @property
    def org_weight(self):
        return self.org_module[0].weight
    
    def get_org_weight_for_compute(self, device: torch.device):
        """Get org_weight on compute device with async transfer if needed"""
        if self.org_module[0].weight is None:
            return None
        weight = self.org_module[0].weight
        return transfer_ramtensor_to_device(weight, device)
    
    def get_org_bias_for_compute(self, device: torch.device):
        """Get org_bias on compute device with async transfer if needed"""
        if self.org_module[0].bias is None:
            return None
        bias = self.org_module[0].bias
        return transfer_ramtensor_to_device(bias, device)

    @org_weight.setter
    def org_weight(self, value):
        self.org_module[0].weight.data.copy_(value)

    def apply_to(self, **kwargs):
        if self.not_supported:
            return
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        if self.not_supported:
            return
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        if self.not_supported:
            return
        self_device = next(self.parameters()).device
        self_dtype = next(self.parameters()).dtype
        self.to(self.org_weight)
        weight, bias = self.get_merged_weight(
            multiplier, self.org_weight.shape, self.org_weight.device
        )
        self.org_weight = weight.to(self.org_weight)
        if bias is not None:
            bias = bias.to(self.org_weight)
            if self.org_module[0].bias is not None:
                self.org_module[0].bias.data.copy_(bias)
            else:
                self.org_module[0].bias = nn.Parameter(bias)
        self.to(self_device, self_dtype)

    def onfly_merge(self, multiplier=1.0):
        if self.not_supported:
            return
        self_device = next(self.parameters()).device
        self_dtype = next(self.parameters()).dtype
        self.to(self.org_weight)
        self.cached_org_weight = self.org_weight.data.cpu()
        self.cached_org_bias = None
        weight, bias = self.get_merged_weight(
            multiplier, self.org_weight.shape, self.org_weight.device
        )
        self.org_weight = weight
        if bias is not None:
            bias = bias.to(self.org_weight)
            if self.org_module[0].bias is not None:
                self.org_module[0].bias.data.copy_(bias)
                self.cached_org_bias = self.org_module[0].bias.data.cpu()
            else:
                self.org_module[0].bias = nn.Parameter(bias)
        if self.org_module[0].bias is not None:
            self.org_module[0].bias = self.org_module[0].bias.to(self.org_weight)
        self.to(self_device, self_dtype)

    def onfly_restore(self):
        if self.not_supported:
            return
        self.org_weight = self.cached_org_weight.to(self.org_weight)
        if self.cached_org_bias is not None:
            self.org_module[0].bias.data.copy_(self.cached_org_bias.to(self.org_weight))
        del self.cached_org_weight
        del self.cached_org_bias

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        return None, None
    
    @torch.no_grad()
    def get_norm(self, device=None):
        return None, None

    def bypass_forward_diff(self, x, scale=1):
        raise NotImplementedError

    def bypass_forward(self, x, scale=1):
        raise NotImplementedError

    def parametrize_forward(self, x: torch.Tensor, *args, **kwargs):
        return self.get_merged_weight(
            multiplier=self.multiplier, shape=x.shape, device=x.device
        )[0].to(x.dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def initialize_norm_cache(self, org_module_weight: torch.Tensor):
        # Choose a reasonable sample size
        n_rows = org_module_weight.shape[0]
        sample_size = min(2000, n_rows)  # Cap at 2000 samples or use all if smaller

        # Sample random indices across all rows
        indices = torch.randperm(n_rows)[:sample_size]

        # Convert to a supported data type first, then index
        # Use float32 for indexing operations
        weights_float32 = org_module_weight.to(dtype=torch.float32)
        sampled_weights = weights_float32[indices].to(device=self.device)

        # Calculate sampled norms
        sampled_norms = torch.norm(sampled_weights, dim=1, keepdim=True)

        # Store the mean norm as our estimate
        self.org_weight_norm_estimate = sampled_norms.mean()

        # Optional: store standard deviation for confidence intervals
        self.org_weight_norm_std = sampled_norms.std()

        # Free memory
        del sampled_weights, weights_float32

    @torch.no_grad()
    def validate_norm_approximation(self, org_module_weight: torch.Tensor, verbose=True):
        # Calculate the true norm (this will be slow but it's just for validation)
        true_norms = []
        chunk_size = 1024  # Process in chunks to avoid OOM

        for i in range(0, org_module_weight.shape[0], chunk_size):
            end_idx = min(i + chunk_size, org_module_weight.shape[0])
            chunk = org_module_weight[i:end_idx].to(device=self.device, dtype=self.dtype)
            chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
            true_norms.append(chunk_norms.cpu())
            del chunk

        true_norms = torch.cat(true_norms, dim=0)
        true_mean_norm = true_norms.mean().item()

        # Compare with our estimate
        estimated_norm = self.org_weight_norm_estimate.item()

        # Calculate error metrics
        absolute_error = abs(true_mean_norm - estimated_norm)
        relative_error = absolute_error / true_mean_norm * 100  # as percentage

        if verbose:
            logger.info(f"True mean norm: {true_mean_norm:.6f}")
            logger.info(f"Estimated norm: {estimated_norm:.6f}")
            logger.info(f"Absolute error: {absolute_error:.6f}")
            logger.info(f"Relative error: {relative_error:.2f}%")

        return {
            'true_mean_norm': true_mean_norm,
            'estimated_norm': estimated_norm,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }

    @torch.no_grad()
    def update_norms(self):
        # Early returns for common cases
        if self.ggpo_beta is None or self.ggpo_sigma is None or not self.training:
            return
        
        if not(self.module_type == "linear" or (self.module_type.startswith("conv") and self.ggpo_conv)):
            return
        
        if not (hasattr(self, 'lora_down') and hasattr(self.lora_down, 'weight') and self.lora_down.weight.grad is not None):
            return
        
        if not (hasattr(self, 'lora_up') and hasattr(self.lora_up, 'weight') and self.lora_down.weight.grad is not None):
            return
        
        # Skip update every other step for convolutions to reduce overhead
        if self.module_type != "linear" and hasattr(self, '_skip_counter'):
            self._skip_counter = not self._skip_counter
            if self._skip_counter:
                return
        else:
            self._skip_counter = False
        
        # Fast path for linear layers
        if self.module_type == "linear":
            # Calculate norms directly without forming the full weight matrix
            up_norm = torch.sum(self.lora_up.weight**2)
            down_norm = torch.sum(self.lora_down.weight**2)
            
            # Frobenius norm of the product can be bounded/approximated 
            effect = torch.sqrt(up_norm * down_norm) * self.scale
            
            # Calculate per-output channel distribution (much faster than full matrix mul)
            up_channel_norms = torch.sum(self.lora_up.weight**2, dim=1, keepdim=True)
            total_norm = up_channel_norms.sum()
            
            # Avoid division by zero and normalize
            self.weight_norms = up_channel_norms * (effect / total_norm)
            self.combined_weight_norms = torch.sqrt(
                (self.org_weight_norm_estimate**2) + self.weight_norms**2
            )
            return
        
        if self.module_type.startswith("conv") and self.ggpo_conv:
            # Handle convolution layers - use sampling for efficiency
            # Sample-based estimation for convolution layers
            out_size = self.lora_up.weight.size(0)
            
            # Use a constant estimation factor based on typical CNN properties
            # This avoids expensive reconstruction while capturing essential scaling
            if not hasattr(self, 'conv_norm_estimate'):
                # Cache this value since it's relatively constant
                up = self.lora_up.weight
                down = self.lora_down.weight
                
                # Sample a small subset of weights to estimate norm
                sample_size = min(self.ggpo_conv_weight_sample_size, up.size(0))
                if sample_size < up.size(0):
                    up_indices = torch.randperm(up.size(0))[:sample_size]
                    up_sample = up[up_indices]
                else:
                    up_sample = up
                    
                sample_size = min(self.ggpo_conv_weight_sample_size, down.size(0))
                if sample_size < down.size(0):
                    down_indices = torch.randperm(down.size(0))[:sample_size]
                    down_sample = down[down_indices]
                else:
                    down_sample = down
                
                # Calculate squared Frobenius norms on samples
                up_norm_sq = torch.sum(up_sample**2) * (up.size(0) / up_sample.size(0))
                down_norm_sq = torch.sum(down_sample**2) * (down.size(0) / down_sample.size(0))
                
                # Cache the estimation factor
                self.conv_norm_estimate = torch.sqrt(up_norm_sq * down_norm_sq) * self.scale
            
            # Calculate per-channel output scaling - much faster than full norm calculation
            up_flat = self.lora_up.weight.view(out_size, -1)
            up_channel_norms = torch.sum(up_flat**2, dim=1, keepdim=True)
            channel_sum = up_channel_norms.sum()
            
            # Distribute the precomputed norm across channels
            self.weight_norms = up_channel_norms * (self.conv_norm_estimate / channel_sum)
            self.combined_weight_norms = torch.sqrt(
                (self.org_weight_norm_estimate**2) + self.weight_norms**2
            )


    @torch.no_grad()
    def update_grad_norms(self):
        if not self.training:
            return
        
        if not(self.module_type == "linear" or (self.module_type.startswith("conv") and self.ggpo_conv)):
            return
        
        if not (hasattr(self, 'lora_down') and hasattr(self.lora_down, 'weight') and self.lora_down.weight.grad is not None):
            return
        
        if not (hasattr(self, 'lora_up') and hasattr(self.lora_up, 'weight') and self.lora_down.weight.grad is not None):
            return
            
        # Skip update every other step for convolutions to reduce overhead
        if self.module_type != "linear" and hasattr(self, '_skip_grad_counter'):
            self._skip_grad_counter = not self._skip_grad_counter
            if self._skip_grad_counter:
                return
        else:
            self._skip_grad_counter = False

        # Use direct parameter access instead of named iteration (faster)
        lora_down_grad = self.lora_down.weight.grad
        lora_up_grad = self.lora_up.weight.grad
        
        # Fast path for linear layers
        if self.module_type == "linear":
            # Calculate gradient norms efficiently using matrix properties
            lora_up_weight = self.lora_up.weight
            lora_down_weight = self.lora_down.weight
            
            # For linear layers, directly calculate gradient approximation
            up_down_grad = self.scale * (lora_up_weight @ lora_down_grad)
            up_grad_down = self.scale * (lora_up_grad @ lora_down_weight)
            
            # Sum the gradient components
            approx_grad = up_down_grad + up_grad_down
            
            # Calculate row-wise norms
            self.grad_norms = torch.norm(approx_grad, dim=1, keepdim=True)
        
        if self.module_type.startswith("conv") and self.ggpo_conv:
            # Use a fast approximation for convolution gradients
            out_size = self.lora_up.weight.size(0)
            
            # Calculate gradient magnitude using norm products (faster than reconstruction)
            up_grad_norm = torch.norm(lora_up_grad.view(-1))
            down_weight_norm = torch.norm(self.lora_down.weight.view(-1))
            up_weight_norm = torch.norm(self.lora_up.weight.view(-1))
            down_grad_norm = torch.norm(lora_down_grad.view(-1))
            
            # Approximation of the combined gradient magnitude
            grad_magnitude = self.scale * (up_grad_norm * down_weight_norm + up_weight_norm * down_grad_norm)
            
            # Distribute gradient magnitude across output channels
            # This avoids expensive per-channel calculations while capturing key behavior
            up_channel_magnitudes = torch.norm(self.lora_up.weight.view(out_size, -1), dim=1, keepdim=True)
            magnitude_sum = up_channel_magnitudes.sum()
            
            # Distribute based on weight magnitudes (channels with larger weights get larger gradients)
            self.grad_norms = up_channel_magnitudes * (grad_magnitude / magnitude_sum)

    @torch.no_grad()
    def init_ggpo(self):
        if self.ggpo_beta is not None and self.ggpo_sigma is not None:
            self.combined_weight_norms = None
            self.grad_norms = None
            self.weight_norms = None
            self.perturbation_norm_factor = 1.0 / math.sqrt(self.org_module[0].weight.shape[0])
            self.initialize_norm_cache(self.org_module[0].weight)
            self.org_module_shape: tuple[int] = self.org_module[0].weight.shape

    @torch.no_grad()
    def accumulate_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                grad = param.grad.detach().flatten()
                self.grad_count += grad.numel()

                # Update running sums
                if self.sum_grads is None:
                    self.sum_grads = grad.sum()
                    self.sum_squared_grads = (grad**2).sum()
                else:
                    self.sum_grads += grad.sum()
                    self.sum_squared_grads += (grad**2).sum()

    def ggpo_pertubation(self, x):
        return None