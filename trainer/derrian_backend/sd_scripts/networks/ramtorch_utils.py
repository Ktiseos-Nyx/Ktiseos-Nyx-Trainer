# Shared utilities for RamTorch CPU-offloaded parameter streaming.
#
# These are used by LoRA/DyLoRA/OFT modules to efficiently transfer
# CPU-resident RamTorch weights to the GPU during the forward pass,
# WITHOUT going through BouncingLinearFn.apply() which saves the full
# CPU weight tensor into the autograd graph via ctx.save_for_backward.
#
# Bypassing BouncingLinearFn prevents the autograd graph from holding
# references to ALL CPU weight tensors simultaneously during backward,
# which was causing massive system RAM usage.

import torch


class AsyncTensorStreamer:
    """Ring-buffer async streamer for CPU→GPU transfers.

    Maintains a small ring buffer of GPU slots so that the transfer of
    the *next* tensor can be overlapped with computation on the *current*
    tensor.  The ring size of 3 is safe for the typical pattern:
        [Weight_Layer_N, Bias_Layer_N, Weight_Layer_N+1]

    This prevents overwriting a slot that the GPU compute stream is still
    reading when the next layer immediately starts transferring.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device=device)

        # RING BUFFER SETTINGS
        # Size 3 is safe: [Weight_Layer_N, Bias_Layer_N, Weight_Layer_N+1]
        # This prevents overwriting the weight currently being computed if the 
        # next layer starts transferring immediately.
        self.num_buffers = 3
        self.idx = 0

        self.buffers = [None] * self.num_buffers
        # Each event records when the *compute* stream finishes using a slot,
        # so we know it is safe to overwrite it on the next round.
        self.compute_done_events = [torch.cuda.Event() for _ in range(self.num_buffers)]

    def transfer(self, tensor_cpu: torch.Tensor) -> torch.Tensor:
        # Pin memory for async DMA (no-op if already pinned).
        if not tensor_cpu.is_pinned():
            tensor_cpu = tensor_cpu.pin_memory()

        # Select the next slot in the Ring Buffer
        slot_idx = self.idx
        self.idx = (self.idx + 1) % self.num_buffers

        ready_event = self.compute_done_events[slot_idx]

        # Wait until the compute stream is done with this slot before we
        # overwrite it.  (First iteration: event is unrecorded → no-op.)
        self.transfer_stream.wait_event(ready_event)

        with torch.cuda.stream(self.transfer_stream):
            with torch.no_grad():

                # We use .to() which uses PyTorch's Caching Allocator. 
                # If self.buffers[slot_idx] existed, it goes back to the pool.
                # We don't manually hold .new_empty() anymore to allow dynamic resizing 
                # if layers have different shapes.
                gpu_tensor = tensor_cpu.to(self.device, non_blocking=True)

                # Keep a reference so Python's GC doesn't free the tensor
                # before the async copy completes.
                self.buffers[slot_idx] = gpu_tensor

            # Record that transfer is finished
            transfer_done = torch.cuda.Event()
            transfer_done.record()

        # Tell the compute stream to wait until the transfer has finished.
        torch.cuda.current_stream().wait_event(transfer_done)

        # Record on the compute stream so the *next* use of this slot waits
        # until we're done computing with it.
        ready_event.record()

        return self.buffers[slot_idx]


# Per-device registry — one streamer per CUDA device for multi-GPU support.
_STREAMERS: dict = {}


def transfer_ramtensor_to_device(
    tensor_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transfer a (possibly RamTorch) CPU tensor to *device* efficiently.

    For ordinary tensors (``is_ramtorch`` not set) this falls back to a
    plain ``.to(device)`` call so it is safe to call unconditionally.

    For RamTorch tensors the async ring-buffer streamer is used, which
    overlaps the DMA transfer with GPU computation and avoids holding the
    full CPU weight in the autograd graph.
    """
    if not getattr(tensor_cpu, "is_ramtorch", False):
        return tensor_cpu.to(device, non_blocking=True)

    if device.type == "cpu":
        return tensor_cpu

    if device not in _STREAMERS:
        _STREAMERS[device] = AsyncTensorStreamer(device)

    return _STREAMERS[device].transfer(tensor_cpu)
