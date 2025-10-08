# Training Parameters Guide

This guide explains the training parameters available in the current three-notebook widget system using the Derrian backend with Kohya integration.

**Note**: While other training systems like OneTrainer or standalone Kohya_ss may have different parameter sets, this documentation focuses specifically on what's available in our widget-based notebook system.

## Core Training Parameters

### Learning Rates

**UNet Learning Rate**
- **Purpose**: Controls how quickly the image generation network learns
- **Typical values**: 1e-4 to 1e-3 (0.0001 to 0.001)
- **Default**: 5e-4 (0.0005)
- **Effect**: Higher values learn faster but may cause instability

**Text Encoder Learning Rate**
- **Purpose**: Controls how the text understanding component learns
- **Typical values**: 1e-5 to 1e-4 (usually 5-10x lower than UNet)
- **Default**: 1e-4 (0.0001)
- **Why lower**: Text encoder is more sensitive to changes

### Network Architecture

**Network Dimension (Rank)**
- **Purpose**: The "size" or capacity of the LoRA adaptation
- **Range**: 4-128, commonly 8-32
- **Default**: 16
- **Trade-offs**: Higher = more detail capture, larger files, slower training
- **Recommendation**: Start with 8-16 for most use cases

**Network Alpha**
- **Purpose**: Scaling factor for LoRA weights
- **Common practice**: Often set to half of network dimension
- **Default**: 8
- **Effect**: Higher alpha = stronger LoRA influence during training

**Network Dropout**
- **Purpose**: Regularization to prevent overfitting
- **Range**: 0.0 (no dropout) to 0.5
- **Default**: 0.0
- **When to use**: Large datasets or signs of overfitting

### Training Schedule

**Epochs**
- **Definition**: How many times to go through entire dataset
- **Balance**: Too few = undertrained, too many = overtrained
- **Monitoring**: Watch loss curves rather than targeting specific numbers

**Batch Size**
- **Purpose**: How many images processed simultaneously
- **Memory impact**: Larger batches use more VRAM
- **Training impact**: Larger batches = more stable gradients
- **Default**: 4
- **Adjustment**: Reduce if running out of VRAM

**VAE Batch Size**
- **Purpose**: Batch size for VAE processing (separate from training batch)
- **Range**: 1-8
- **Default**: 1
- **Effect**: Higher values process images faster but use more VRAM

## Optimizer Options

These optimizers are available in the Training Configuration widget:

**AdamW (Default)**
- **Description**: Standard adaptive optimizer with weight decay
- **Memory usage**: Higher memory requirements
- **Stability**: Reliable and well-tested
- **Best for**: General purpose training

**AdamW8bit**
- **Description**: Memory-optimized version of AdamW
- **Memory savings**: Significant VRAM reduction
- **Requirement**: Needs Triton compiler (may fail in some environments)
- **Use case**: When memory is constrained

**LoraEasyCustomOptimizer.came.CAME**
- **Description**: Confidence-guided Adaptive Memory Efficient optimizer
- **Memory efficiency**: Very low memory overhead
- **Special features**: Confidence-guided weight updates, good for unstable models
- **Best for**: Memory-constrained systems, v-prediction models
- **Widget name**: Shows as "LoraEasyCustomOptimizer.came.CAME" in dropdown

**Prodigy**
- **Description**: Learning rate-free optimization
- **Feature**: Automatically finds optimal learning rates
- **Usage**: Set learning rate to 1.0, let Prodigy handle the rest
- **Best for**: When you don't want to tune learning rates manually

**AdaFactor**
- **Description**: Memory-efficient adaptive optimizer
- **Memory usage**: Scales well with model size
- **Stability**: More stable than AdamW for large models
- **Best for**: Large models, SDXL training

**DAdaptation/DadaptAdam/DadaptLion**
- **Description**: Adaptive learning rate optimizers
- **Feature**: Automatically adjust learning rates during training
- **Usage**: Can use higher learning rates initially

**Lion**
- **Description**: Evolved sign momentum optimizer
- **Memory**: Lower memory usage than AdamW
- **Performance**: Can be faster than AdamW in some cases

**SGDNesterov/SGDNesterov8bit**
- **Description**: Stochastic gradient descent with Nesterov momentum
- **Usage**: Classic optimizer, less commonly used for LoRA
- **Memory**: Lower memory requirements

## Advanced Options

### Precision Settings

**Mixed Precision**
- **fp16**: Half precision, saves memory
- **bf16**: Brain float 16, better stability than fp16
- **fp32**: Full precision, highest quality but most memory

**FP8 Base**
- **Purpose**: Use FP8 precision for base model
- **Benefit**: Memory savings for large models
- **Compatibility**: Requires compatible hardware

### Memory Optimization

**Gradient Checkpointing**
- **Purpose**: Save memory by recomputing gradients
- **Trade-off**: Slower training but uses less VRAM
- **When to use**: When running out of memory

**No Half VAE**
- **Purpose**: Use full precision for VAE
- **When to use**: If getting VAE-related artifacts
- **Trade-off**: Uses more VRAM but better quality

### LoRA Types

These LoRA types are available in the LoRA Structure section:

**LoRA (Default)**
- **Description**: Basic Low-Rank Adaptation
- **Compatibility**: Works with all models
- **File size**: Smaller files
- **Best for**: General purpose training

**LoCon**
- **Description**: LoRA for Convolution layers
- **Feature**: Can adapt convolutional layers too
- **File size**: Larger than standard LoRA
- **Best for**: Detailed textures and fine features

**LoKR**
- **Description**: LoRA with Kronecker Product
- **Feature**: Alternative decomposition method
- **Usage**: Experimental approach

**DyLoRA**
- **Description**: Dynamic LoRA
- **Feature**: Adaptive rank during training
- **Usage**: Advanced technique

**DoRA (Weight Decomposition)**
- **Description**: Weight-decomposed LoRA
- **Feature**: Separates magnitude and direction
- **Usage**: Newer experimental method

**LoHa (Hadamard Product)**
- **Description**: LoRA with Hadamard Product
- **Feature**: Alternative decomposition method
- **Usage**: Experimental approach

**(IA)³ (Few Parameters)**
- **Description**: Extremely parameter-efficient adaptation
- **Feature**: Very small file sizes
- **Usage**: Minimal parameter approach

**GLoRA (Generalized LoRA)**
- **Description**: Generalized LoRA approach
- **Feature**: Extended LoRA capabilities
- **Usage**: Advanced technique

**GLoKr (Generalized LoKR)**
- **Description**: Generalized LoKR approach
- **Feature**: Extended LoKR capabilities
- **Usage**: Advanced technique

**Native Fine-Tuning (Full)**
- **Description**: Full model fine-tuning
- **Feature**: Adapts all parameters
- **Usage**: Complete model training

**Diag-OFT (Orthogonal Fine-Tuning)**
- **Description**: Orthogonal fine-tuning approach
- **Feature**: Maintains orthogonality constraints
- **Usage**: Specialized technique

**BOFT (Butterfly Transform)**
- **Description**: Butterfly transformation method
- **Feature**: Structured transformation
- **Usage**: Advanced structured approach

## Dataset Configuration

**Resolution**
- **Standard**: 512x512 for SD1.5, 1024x1024 for SDXL
- **Effect**: Higher resolution needs more VRAM
- **Quality**: Match your target generation resolution

**Repeats**
- **Purpose**: How many times each image is seen per epoch
- **Effect**: Higher repeats = more training on each image
- **Balance**: Too high can cause overfitting

**Bucket Resolution**
- **Purpose**: Allow training on different aspect ratios
- **Benefit**: Don't need to crop all images to squares
- **Options**: Enable/disable aspect ratio bucketing

## Learning Rate Schedulers

These schedulers are available in the Training Configuration widget:

**Cosine (Default/Recommended)**
- **Behavior**: Smooth cosine curve decrease
- **Use case**: Most LoRA training (recommended)
- **Widget setting**: "cosine"

**Cosine with Restarts**
- **Behavior**: Cosine decay with periodic restarts
- **Use case**: Long training runs, escaping local minima
- **Widget setting**: "cosine_with_restarts"
- **Extra parameter**: Scheduler Num controls restart frequency

**Constant**
- **Behavior**: Learning rate stays the same throughout
- **Use case**: Simple training scenarios
- **Widget setting**: "constant"

**Linear**
- **Behavior**: Linear decrease from start to end
- **Use case**: Gradual slowdown
- **Widget setting**: "linear"

**Polynomial**
- **Behavior**: Polynomial decay curve
- **Use case**: Custom decay patterns
- **Widget setting**: "polynomial"
- **Extra parameter**: Scheduler Num controls polynomial power

**REX**
- **Behavior**: REX scheduler
- **Use case**: Advanced scheduling
- **Widget setting**: "rex"

## Parameter Interaction

### Learning Rate vs Batch Size
- Larger batch sizes may need higher learning rates
- Smaller batch sizes may need lower learning rates
- Monitor loss curves to find the right balance

### Network Dimension vs Training Data
- More complex content may need higher dimensions
- Simple styles can work with lower dimensions
- Balance file size vs detail capture

### Optimizer vs Memory
- AdamW8bit/LoraEasyCustomOptimizer.came.CAME for memory-constrained systems
- Standard AdamW for maximum stability
- Prodigy for automatic tuning

## Troubleshooting Parameters

**Training Too Slow**
- Increase batch size if VRAM allows
- Use fp16 instead of fp32
- Enable gradient checkpointing for memory vs speed trade-off

**Out of Memory Errors**
- Reduce batch size
- Use fp16 precision
- Try memory-efficient optimizers (LoraEasyCustomOptimizer.came.CAME, AdaFactor)
- Enable gradient checkpointing
- Use cache latents to disk
- Reduce VAE batch size

**Poor Quality Results**
- Check learning rates (may be too high/low)
- Adjust network dimension
- Review dataset quality and size
- Monitor for overfitting/underfitting

**Training Unstable (NaN losses)**
- Lower learning rates
- Try different optimizer (LoraEasyCustomOptimizer.came.CAME for stability)
- Check for corrupted images in dataset
- Use gradient clipping if available
- Enable Min SNR Gamma for noise stability