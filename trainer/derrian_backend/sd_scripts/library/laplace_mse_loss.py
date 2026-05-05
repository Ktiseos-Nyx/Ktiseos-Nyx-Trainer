import torch
import torch.nn.functional as F

def gaussian_kernel_2d(size=5, sigma=1.0, channels=1, device=None):
    coords = torch.arange(size, device=device) - size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

def conv2d_fp64_compat(input_tensor, weight, padding="same", groups=1):
    """
    Executes Conv2d in fp32 (to satisfy backend constraints) but accepts
    and returns fp64 to maintain pipeline consistency.
    """
    # 1. Downcast inputs to float32
    inp_f32 = input_tensor.to(torch.float32)
    w_f32 = weight.to(torch.float32)
    
    # 2. Perform Convolution
    out_f32 = F.conv2d(inp_f32, w_f32, padding=padding, groups=groups)
    
    # 3. Upcast immediately back to float64
    return out_f32.to(torch.float64)

def mse_pyramid_loss_2d_non_reduced(pred, target, levels=5):
    """
    Calculates a multi-scale MSE loss and returns the full spatial loss map.
    Returns: Tensor of shape [B, C, H, W]
    """
    b, c, h, w = pred.shape
    device = pred.device
    
    # Create Universal Kernel
    kernel_size = 5
    kernel = gaussian_kernel_2d(size=kernel_size, sigma=1.0, channels=c, device=device)
    
    working_pred = pred.to(torch.float64)
    working_target = target.to(torch.float64)
    
    # Initialize the accumulator with the shape of the original input
    total_loss_map = torch.zeros_like(pred)
    
    # Helper to calculate normalized error map and accumulate
    def accumulate_level_loss(l_pred, l_target):
        # 1. Calculate Squared Error (Non-reduced MSE)
        # Shape: [B, C, H_level, W_level]
        squared_error = (l_pred - l_target) ** 2
        
        # 2. Normalize by target STD (per sample, across C,H,W)
        # keepdim=True ensures shape [B, 1, 1, 1] for broadcasting
        target_std = l_target.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        
        normalized_error = squared_error / target_std
        
        # 3. Upscale to original resolution using nearest-exact
        # We only upscale if the current level is smaller than original
        if normalized_error.shape[-2:] != (h, w):
            normalized_error = F.interpolate(
                normalized_error, 
                size=(h, w), 
                mode='nearest-exact'
            )
            
        return normalized_error

    actual_levels = 0

    for i in range(levels - 1):
        if working_pred.shape[-1] < kernel_size or working_pred.shape[-2] < kernel_size:
            break
            
        # Blur
        blurred_pred = conv2d_fp64_compat(working_pred, weight=kernel, padding="same", groups=c)
        blurred_target = conv2d_fp64_compat(working_target, weight=kernel, padding="same", groups=c)
        
        # High-freq component
        hf_pred = working_pred - blurred_pred
        hf_target = working_target - blurred_target
        
        # Accumulate loss for this level
        total_loss_map += accumulate_level_loss(hf_pred, hf_target)
        actual_levels += 1
        
        # Downsample
        working_pred = F.interpolate(blurred_pred, scale_factor=0.5, mode="area")
        working_target = F.interpolate(blurred_target, scale_factor=0.5, mode="area")
    
    # Process the final Low-freq component
    total_loss_map += accumulate_level_loss(working_pred, working_target)
    actual_levels += 1
    
    # Returns [B, C, H, W]
    return total_loss_map / actual_levels