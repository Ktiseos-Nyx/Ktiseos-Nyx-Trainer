# Training Settings Reference

These are starting point suggestions for different scenarios. Training is experimental - adjust based on your results and hardware. Please note that training parameters have not been factored into this document guide as of yet, and training is an absolute art and always depends on what you're doing. If your system doesn't work with a specific setting always experiment.

## Suggested Starting Points by Dataset Size

**Important**: These are suggestions, not rules. Adjust based on your specific model, content, and results.

### Small Datasets (50-100 Images)

**Settings:**
- Batch Size: 1-3
- Epochs: 7-10
- Learning Rate: 5e-4 (UNet), 1e-4 (Text Encoder)
- Memory Usage: 6-8GB VRAM
- Training Time: 2-4 hours

**Use for:**
- Character LoRAs
- Simple style concepts
- Testing new ideas

### Medium Datasets (100-300 Images)

**Settings:**
- Batch Size: 2-3
- Epochs: 5-8
- Learning Rate: 4e-4 (UNet), 8e-5 (Text Encoder)
- Memory Usage: 8-12GB VRAM
- Training Time: 4-8 hours

**Use for:**
- Complex characters
- Style LoRAs with variety
- Multi-concept training

### Large Datasets (300-500 Images)

**Settings:**
- Batch Size: 4
- Epochs: 5
- Learning Rate: 3e-4 (UNet), 6e-5 (Text Encoder)
- Memory Usage: 10-16GB VRAM
- Training Time: 6-12 hours

**Use for:**
- Professional style LoRAs
- Complex scene training
- Multi-character datasets

### Very Large Datasets (500+ Images)

**Settings:**
- Batch Size: 6-8
- Epochs: 3-5
- Learning Rate: 2e-4 (UNet), 5e-5 (Text Encoder)
- Memory Usage: 12-20GB VRAM
- Training Time: 8-16 hours

**Use for:**
- Professional-grade style LoRAs
- Environment/scene training
- Complex multi-concept LoRAs

## Recognizing Training Problems

### Overfitting Signs
- Generated images look identical
- Same poses and backgrounds repeatedly
- Poor response to new prompts
- Only recreates training images exactly

### Underfitting Signs
- Blurry or soft outputs
- Weak response to trigger words
- Generic results that look like base model
- Inconsistent quality

### Good Training Signs
- Consistent style but with variety
- Responds well to different prompts
- Clear but not overwhelming effect
- Works across different scenarios

## Memory Optimization

### Low VRAM (8GB or less)
- Use batch size 1
- Enable gradient checkpointing
- Use CAME optimizer
- Train at 512x512 resolution
- Use mixed precision (fp16)

### Medium VRAM (12-16GB)
- Batch size 2-4
- Standard settings work
- Full resolution training possible
- Can use memory-intensive optimizers

### High VRAM (20GB+)
- Larger batch sizes (4-8)
- Can train multiple LoRAs
- Higher resolution training
- Advanced optimization techniques

## Quality Control

### During Training
- Monitor loss curves for steady decrease
- Test intermediate checkpoints
- Use consistent test prompts
- Stop at optimal checkpoint

### After Training
- Test with new prompts not in training data
- Try different styles and contexts
- Check compatibility with different base models
- Test trigger word effectiveness

### Troubleshooting Common Issues

**CUDA Out of Memory:**
- Reduce batch size to 1
- Lower training resolution
- Use gradient checkpointing
- Try CAME optimizer

**Training Too Slow:**
- Check batch size settings
- Monitor GPU usage
- Close other applications
- Verify CUDA installation

**Poor Quality Results:**
- Review dataset quality
- Check caption accuracy
- Adjust learning rates
- Try different optimizers

**Loss Becomes NaN:**
- Lower learning rate significantly
- Check for corrupted images
- Try adaptive optimizers (Prodigy, CAME)
- Reduce network dimension
