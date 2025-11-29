'use client';

import { useState } from 'react';
import { Home, BookOpen, ChevronRight, ExternalLink } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { GradientCard } from '@/components/effects';

interface DocSection {
  id: string;
  title: string;
  icon: string;
  content: string;
  links?: { label: string; url: string }[];
}

const docSections: DocSection[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: '🚀',
    content: `Welcome to Ktiseos-Nyx Trainer! This guide will help you get started with training your first LoRA model.

**Prerequisites:**
- A dataset of images (20-100+ images recommended)
- A base model downloaded (SDXL, SD 1.5, Flux, or SD 3.5)
- (Optional) A VAE file for improved quality

**Quick Start Workflow:**
1. Download a base model from the Models & VAEs page
2. Upload your dataset (images in a folder)
3. Auto-tag your images using WD14 tagger
4. Configure training parameters
5. Start training and monitor progress
6. Download your trained LoRA`,
    links: [
      { label: 'VastAI Setup Guide', url: 'https://vast.ai/docs' },
      { label: 'Kohya Documentation', url: 'https://github.com/kohya-ss/sd-scripts' },
    ],
  },
  {
    id: 'datasets',
    title: 'Dataset Preparation',
    icon: '📁',
    content: `Proper dataset preparation is crucial for good LoRA training results.

**Dataset Structure:**
- Use Kohya format: \`{repeats}_{folder_name}\`
- Example: \`10_my_character\` (10 repeats of images in folder)
- Higher repeats = more training on those images

**Image Requirements:**
- Format: JPG, PNG, or WebP
- Resolution: 512x512 to 1024x1024 (SDXL prefers 1024x1024)
- Quality: High-quality, well-lit images work best
- Variety: Different poses, angles, and settings
- Quantity: 20-100 images is a good range

**Tagging:**
- WD14 tagger auto-generates tags for each image
- Tags are saved as .txt files next to images
- You can manually edit tags in the Tag Editor
- Common format: \`1girl, blue eyes, long hair, smile\``,
    links: [
      { label: 'WD14 Tagger Models', url: 'https://huggingface.co/SmilingWolf' },
    ],
  },
  {
    id: 'training-config',
    title: 'Training Configuration',
    icon: '⚙️',
    content: `Understanding training parameters helps you achieve better results.

**Essential Parameters:**

**Learning Rate (LR):**
- Default: 1e-4 (0.0001)
- Lower = slower, more stable training
- Higher = faster, but risk of overtraining
- SDXL: 1e-4 to 5e-5
- SD 1.5: 1e-4 to 1e-3

**Network Rank (Dim):**
- Default: 32
- Higher = more detail, larger file size
- Common values: 8, 16, 32, 64, 128
- Recommendation: 32 for characters, 64+ for styles

**Epochs:**
- Number of times to train on full dataset
- Start with 5-10 epochs
- Monitor loss to avoid overtraining

**Batch Size:**
- Higher = faster training, more VRAM
- Lower = slower, less VRAM
- Recommendation: 1-4 depending on GPU`,
  },
  {
    id: 'lora-types',
    title: 'LoRA Types (LyCORIS)',
    icon: '🔧',
    content: `Different LoRA architectures for different use cases.

**Standard LoRA:**
- Most common and widely supported
- Good for characters and concepts
- Efficient and fast to train

**LoCon (LoRA for Convolution):**
- Better for style transfer
- Captures texture and patterns well
- Slightly larger file size

**LoHa (LoRA with Hadamard Product):**
- Advanced architecture
- Better detail retention
- More complex training

**LoKr (LoRA with Kronecker Product):**
- Experimental but powerful
- Can capture fine details
- Requires more experimentation

**Recommendation:** Start with standard LoRA, experiment with LoCon for styles.`,
  },
  {
    id: 'monitoring',
    title: 'Training Monitoring',
    icon: '📊',
    content: `Understanding training progress and when to stop.

**Loss Values:**
- Loss should generally decrease over time
- Too low (<0.05): Possible overtraining
- Fluctuating wildly: Learning rate too high
- Not decreasing: Learning rate too low

**Sample Images:**
- Generate sample images during training
- Check for quality and adherence to concept
- Stop if images degrade (overtraining)

**Signs of Good Training:**
- Steady loss decrease
- Sample images improve quality
- Concept is recognizable
- No artifacts or degradation

**Signs of Overtraining:**
- Loss becomes very low (<0.05)
- Images look "burned" or have artifacts
- Loss of diversity in outputs
- Stop training and use earlier checkpoint`,
  },
  {
    id: 'post-training',
    title: 'Post-Training',
    icon: '🎨',
    content: `What to do after training completes.

**LoRA Resizing:**
- Reduce LoRA file size while maintaining quality
- Useful for sharing or reducing VRAM usage
- Common targets: dim 32 → 16 or 8
- Use the Resize tool in Utilities

**Testing Your LoRA:**
- Test in multiple inference UIs (ComfyUI, A1111, Forge)
- Try different strengths (0.5 to 1.0)
- Combine with different base models
- Test with various prompts

**Sharing:**
- Upload to HuggingFace Hub via Utilities page
- Include sample images and trigger words
- Document recommended settings
- Specify compatible base models`,
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    icon: '🔍',
    content: `Common issues and solutions.

**Training Won't Start:**
- Check that base model is downloaded
- Verify dataset path is correct
- Ensure images are valid format
- Check backend logs for errors

**Out of Memory (OOM):**
- Reduce batch size to 1
- Lower network dimension (rank)
- Enable gradient checkpointing
- Use a smaller base model

**Poor Results:**
- Increase number of epochs
- Adjust learning rate
- Add more varied training images
- Check image quality and tags
- Try different LoRA type

**Loss Not Decreasing:**
- Increase learning rate
- Check that images are properly tagged
- Verify dataset structure
- Increase batch size if possible`,
  },
];

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState<string>('getting-started');

  const currentSection = docSections.find((s) => s.id === activeSection) || docSections[0];

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Documentation', icon: <BookOpen className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-emerald-400 to-green-400 bg-clip-text text-transparent">
            Documentation
          </h1>
          <p className="text-xl text-muted-foreground">
            Guides and references for LoRA training
          </p>
        </div>

        {/* Two-column layout */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-card backdrop-blur-sm border border-border rounded-lg p-4 sticky top-4 overflow-hidden">
              <h2 className="text-lg font-bold text-foreground mb-4">Sections</h2>
              <nav className="space-y-1">
                {docSections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors truncate ${
                      activeSection === section.id
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                    }`}
                  >
                    <span className="text-xl flex-shrink-0">{section.icon}</span>
                    <span className="text-sm font-medium truncate">{section.title}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <GradientCard variant="ocean" intensity="subtle">
              <div className="p-8">
                <div className="flex items-center gap-3 mb-6">
                  <span className="text-4xl">{currentSection.icon}</span>
                  <h2 className="text-3xl font-bold text-foreground">{currentSection.title}</h2>
                </div>

                <div className="prose prose-invert max-w-none break-words overflow-hidden">
                  {currentSection.content.split('\n\n').map((paragraph, idx) => {
                    if (paragraph.startsWith('**') && paragraph.endsWith(':**')) {
                      // Section header
                      return (
                        <h3 key={idx} className="text-xl font-bold text-cyan-400 mt-6 mb-3">
                          {paragraph.replace(/\*\*/g, '').replace(':', '')}
                        </h3>
                      );
                    } else if (paragraph.startsWith('**') && paragraph.includes('**\n')) {
                      // Bold header with content
                      const [header, ...rest] = paragraph.split('\n');
                      return (
                        <div key={idx} className="mb-4">
                          <h4 className="text-lg font-semibold text-foreground mb-2">
                            {header.replace(/\*\*/g, '')}
                          </h4>
                          <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
                            {rest.join('\n')}
                          </p>
                        </div>
                      );
                    } else if (paragraph.startsWith('- ') || paragraph.includes('\n- ')) {
                      // List items
                      const items = paragraph.split('\n').filter((line) => line.trim());
                      return (
                        <ul key={idx} className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
                          {items.map((item, i) => (
                            <li
                              key={i}
                              className="ml-4"
                              dangerouslySetInnerHTML={{
                                __html: item.replace(/^- /, '').replace(/\*\*(.*?)\*\*/g, '<strong class="text-foreground">$1</strong>')
                              }}
                            />
                          ))}
                        </ul>
                      );
                    } else {
                      // Regular paragraph
                      return (
                        <p
                          key={idx}
                          className="text-muted-foreground leading-relaxed mb-4"
                          dangerouslySetInnerHTML={{
                            __html: paragraph
                              .replace(/\*\*(.*?)\*\*/g, '<strong class="text-foreground">$1</strong>')
                              .replace(/`(.*?)`/g, '<code class="bg-slate-800 px-2 py-1 rounded text-cyan-400 break-all">$1</code>'),
                          }}
                        />
                      );
                    }
                  })}
                </div>

                {/* External Links */}
                {currentSection.links && currentSection.links.length > 0 && (
                  <div className="mt-8 pt-6 border-t border-slate-700">
                    <h4 className="text-lg font-semibold text-foreground mb-3">External Resources</h4>
                    <div className="space-y-2">
                      {currentSection.links.map((link, idx) => (
                        <a
                          key={idx}
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors"
                        >
                          <ExternalLink className="w-4 h-4" />
                          <span>{link.label}</span>
                          <ChevronRight className="w-4 h-4" />
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </GradientCard>
          </div>
        </div>
      </div>
    </div>
  );
}
