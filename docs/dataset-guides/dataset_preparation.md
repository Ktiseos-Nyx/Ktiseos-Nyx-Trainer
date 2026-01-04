# 📖 Dataset Preparation Guide

This comprehensive guide covers all dataset preparation tasks for LoRA training.

## 📁 Dataset Upload & Management

### Local Upload Options
- **Drag & Drop**: Simply drag ZIP files into the upload area
- **File Browser**: Click to select ZIP files or folders
- **Direct Folders**: Point to existing image folders on your system

### HuggingFace Integration
- **Direct Download**: Paste HuggingFace dataset URLs
- **Token Support**: Use HF tokens for private datasets
- **Automatic Extraction**: Handles HF dataset formats

### Supported Formats
- **Archives**: ZIP, RAR, 7z archives
- **Images**: JPG, PNG, WebP formats
- **Folder Structure**: Flat folders or nested directories

## 🏷️ Auto-Tagging Systems

### WD14 v3 Taggers (Recommended for Anime/Art)

#### Available Models:
- **wd14-vit-v2**: General purpose, balanced accuracy and speed
- **wd14-convnext-v2**: Higher accuracy, slower processing
- **wd14-swinv2-v2**: Best for complex scenes and compositions
- **wd14-convnext-v3**: Latest model with improved accuracy

#### Features:
- **ONNX Optimization**: 2-3x faster inference when available
- **Threshold Control**: Adjust tag sensitivity (0.1-0.9)
- **Batch Processing**: Efficient GPU utilization
- **Blacklist Support**: Filter unwanted tags during tagging

#### Best Settings:
- **Characters**: 0.35 threshold, wd14-vit-v2 model
- **Complex Art**: 0.4 threshold, wd14-swinv2-v2 model
- **General Anime**: 0.35 threshold, wd14-convnext-v2 model

### BLIP Captioning (Best for Photos/Realistic)

#### Features:
- **Natural Language**: Generates descriptive sentences
- **Scene Understanding**: Captures context and relationships
- **Length Control**: 10-75 tokens per caption
- **Beam Search**: Higher quality generation

#### Best For:
- Real photos and portraits
- Realistic landscapes and scenes
- Complex real-world compositions
- When you want natural descriptions

## ✏️ Caption Management

### Bulk Editing Tools
- **Global Find & Replace**: Update terms across all captions
- **Tag Addition/Removal**: Add or remove specific tags
- **Trigger Word Injection**: Add unique identifiers
- **Format Standardization**: Consistent comma separation

### Advanced Features
- **Tag Frequency Analysis**: See most/least common tags
- **Caption Length Distribution**: Optimize caption lengths
- **Duplicate Tag Detection**: Find and merge similar tags
- **Quality Metrics**: Caption completeness scoring

### Manual Editing
- **Individual Caption Editing**: Fine-tune specific images
- **Preview Changes**: See before/after comparisons
- **Undo/Redo Support**: Safe editing with rollback
- **Real-time Validation**: Check tag format and content

## 🎯 Trigger Word System

### Purpose
- **Unique Identification**: Help model learn your specific concept
- **Generation Control**: Enable targeted image generation
- **Style Consistency**: Maintain coherent visual style

### Best Practices
- **Unique Terms**: Use uncommon words or character names
- **Consistent Placement**: Beginning or end of captions
- **Memorable**: Easy to remember for generation
- **Avoid Conflicts**: Don't use existing art terms

### Examples
- **Characters**: "saria_zelda", "miku_vocaloid", "elsa_frozen"
- **Styles**: "cyberpunk_neon", "watercolor_soft", "pixel_art_style"
- **Concepts**: "golden_hour_photo", "macro_close_up", "vintage_film"

## 🚫 Tag Filtering & Blacklists

### Quality Filters
- **Confidence Thresholds**: Remove low-confidence predictions
- **Tag Length Limits**: Filter overly long or short tags
- **Special Character Removal**: Clean up formatting issues

### Content Filters
- **NSFW Content**: Remove inappropriate tags
- **Bias Reduction**: Filter demographic assumptions
- **Style Conflicts**: Remove contradictory style tags
- **Irrelevant Details**: Focus on important features

### Custom Blacklists
Create your own filters for:
- **Unwanted Aesthetics**: "ugly", "blurry", "low quality"
- **Technical Artifacts**: "watermark", "signature", "border"
- **Conflicting Styles**: Mix of realistic and cartoon tags
- **Personal Preferences**: Any tags you don't want

## 📈 Quality Analysis

### Image Statistics
- **Resolution Analysis**: Check size distribution
- **Format Consistency**: Ensure compatible formats
- **File Size Distribution**: Identify outliers
- **Aspect Ratio Analysis**: Portrait vs landscape balance

### Tag Distribution
- **Most Common Tags**: Identify dataset themes
- **Rare Tags**: Find unique characteristics
- **Tag Co-occurrence**: Which tags appear together
- **Vocabulary Size**: Total unique tags

### Caption Quality Metrics
- **Average Length**: Optimal 50-200 tokens
- **Completeness Score**: How well images are described
- **Consistency Check**: Similar tags for similar images
- **Readability**: Natural language flow

## 💡 Best Practices by Use Case

### Character LoRAs
**Dataset Size**: 15-50 high-quality images
**Variety Requirements**:
- Different poses and angles
- Various expressions and emotions
- Multiple outfits or variations
- Consistent character design

**Tagging Tips**:
- Focus on character-specific features
- Include consistent trigger word
- Tag clothing and accessories
- Note distinctive features

### Style LoRAs
**Dataset Size**: 50-200 images minimum
**Consistency Requirements**:
- Similar art style and technique
- Consistent color palette
- Similar rendering approach
- Unified aesthetic vision

**Tagging Tips**:
- Emphasize style descriptors
- Include medium/technique tags
- Note color and lighting style
- Tag composition elements

### Concept LoRAs
**Dataset Size**: 20-100 images depending on complexity
**Focus Areas**:
- Clear representation of concept
- Various contexts and situations
- Good quality examples
- Avoid conflicting interpretations

**Tagging Tips**:
- Precise concept description
- Include context tags
- Note relevant attributes
- Avoid ambiguous terms

## 🔧 Technical Recommendations

### Image Preparation
- **Resolution**: 768x768 minimum, 1024x1024 recommended
- **Format**: JPG for photos, PNG for art with transparency
- **Quality**: High quality, avoid compression artifacts
- **Consistency**: Similar quality across dataset

### Caption Optimization
- **Length**: 50-200 tokens for balanced training
- **Accuracy**: Verify auto-generated tags
- **Completeness**: Describe all important features
- **Consistency**: Use same terms for same concepts

### Storage Considerations
- **Backup**: Keep original images separate
- **Organization**: Clear folder structure
- **Version Control**: Track caption changes
- **Documentation**: Note preparation decisions

## 🚨 Common Issues & Solutions

### "No Images Found"
- **Check ZIP Structure**: Images should be in root or single folder
- **Verify Formats**: Only JPG, PNG, WebP supported
- **File Corruption**: Test individual image files
- **Path Issues**: Avoid special characters in names

### "Tagging Failed"
- **Internet Connection**: Required for model downloads
- **Disk Space**: Need 2-3GB for tagger models
- **Memory Issues**: Reduce batch size or image resolution
- **Model Access**: Some models require HuggingFace login

### "Captions Too Long/Short"
- **Adjust Thresholds**: Lower for fewer tags, higher for more
- **Tag Filtering**: Remove irrelevant tags
- **Manual Editing**: Fine-tune important images
- **Model Selection**: Different taggers have different verbosity

### "Missing Trigger Words"
- **Check Injection Settings**: Verify trigger word configuration
- **Review Filters**: Ensure trigger word isn't blacklisted
- **Manual Verification**: Check a few files manually
- **Batch Processing**: Use bulk edit to add missing triggers

### "Inconsistent Quality"
- **Image Review**: Remove low-quality images
- **Tag Verification**: Check auto-generated tags
- **Caption Editing**: Improve important descriptions
- **Threshold Tuning**: Adjust confidence levels

## 📚 Advanced Techniques

### Multi-Character Datasets
- Use character-specific trigger words
- Separate tagging for each character
- Careful caption management
- Consider individual LoRAs vs combined

### Style Mixing Prevention
- Consistent style vocabulary
- Remove conflicting style tags
- Manual curation of examples
- Clear style boundaries

### Quality Enhancement
- Iterative caption improvement
- A/B testing different tag sets
- Community feedback integration
- Professional annotation review

---

*"Good datasets make good LoRAs!" - Take time to prepare quality training data.*