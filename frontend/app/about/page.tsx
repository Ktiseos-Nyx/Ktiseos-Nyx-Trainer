'use client';

import { Home, Info, Github, Heart, Zap, Sparkles, ExternalLink, Twitter, Youtube, Twitch } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { GradientCard } from '@/components/effects';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'About', icon: <Info className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
            About the Trainer
          </h1>
          <p className="text-xl text-muted-foreground">
            Part of the Ktiseos-Nyx ecosystem
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-6">
          {/* Ktiseos-Nyx Brand */}
          <GradientCard variant="dusk" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Sparkles className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                <h2 className="text-2xl font-bold text-foreground">About Ktiseos-Nyx</h2>
              </div>
              <div className="text-foreground space-y-3">
                <p>
                  <strong className="text-purple-600 dark:text-purple-400">Ktiseos Nyx</strong> is a space cultivated by{' '}
                  <strong className="text-foreground">Earth & Dusk Media</strong>, a name whispered from the echoes of{' '}
                  <em>Ktisis Hyperboreia</em>, the level 87 dungeon in Final Fantasy XIV: Endwalker. Like the Ktiseos
                  gear found within that ancient place, we see power and potential in what's often overlooked – a place
                  where we are allowed to grow, experiment, and expand.
                </p>
                <p>
                  We are a collective of coders, gamers, artists, and thinkers whose diverse experiences and perspectives
                  shape the world around us. We believe that the most transformative ideas are born when we dare to stray
                  from the well-worn paths.
                </p>
                <p>
                  Ktiseos Nyx is a refuge for those who value community, where the foundations we build are as important
                  as the tools themselves. We don't aim to simply conform; we strive to create connections, push limits,
                  and foster a sanctuary for the bold and brilliant.
                </p>
                <p className="text-purple-600 dark:text-purple-300 font-semibold italic">
                  In the darkness, we find our strength; together, we craft a new dawn.
                </p>
              </div>
            </div>
          </GradientCard>

          {/* This Trainer */}
          <GradientCard variant="ocean" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Zap className="w-8 h-8 text-cyan-600 dark:text-cyan-400" />
                <h2 className="text-2xl font-bold text-foreground">Ktiseos-Nyx Trainer</h2>
              </div>
              <div className="text-foreground space-y-3">
                <p>
                  The Trainer is a modern, web-based interface for training LoRA (Low-Rank Adaptation) models
                  for Stable Diffusion, SDXL, Flux, and SD 3.5. Built with accessibility and power-user features in mind.
                </p>
                <p>
                  Originally a Jupyter notebook workflow, it has evolved into a full-stack application with a
                  beautiful Next.js frontend and FastAPI backend, making professional-grade LoRA training
                  accessible to everyone.
                </p>
              </div>
            </div>
          </GradientCard>

          {/* Features */}
          <GradientCard variant="ocean" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Zap className="w-8 h-8 text-cyan-600 dark:text-cyan-400" />
                <h2 className="text-2xl font-bold text-foreground">Key Features</h2>
              </div>
              <ul className="space-y-3 text-foreground">
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Comprehensive Training Config:</strong> 100+ parameters organized across 7 intuitive tabs</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Auto-Tagging:</strong> WD14 tagger support with 6 model variants (v2/v3 for ViT, ConvNeXT, SwinV2)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Real-time Monitoring:</strong> WebSocket-based live progress tracking during training</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Model Downloads:</strong> Built-in downloader for HuggingFace and Civitai models/VAEs</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Step Calculator:</strong> Interactive calculator with Kohya-compatible step counting</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Post-Training Utils:</strong> LoRA resizing and HuggingFace Hub upload integration</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-600 dark:text-cyan-400 mt-1">•</span>
                  <span><strong className="text-foreground">Multi-Architecture:</strong> Supports SDXL, SD 1.5, Flux, and SD 3.5</span>
                </li>
              </ul>
            </div>
          </GradientCard>

          {/* Technology Stack */}
          <GradientCard variant="cotton-candy" intensity="subtle">
            <div className="p-6">
              <h2 className="text-2xl font-bold text-foreground mb-4">Built With</h2>
              <div className="grid md:grid-cols-2 gap-4 text-foreground">
                <div>
                  <h3 className="text-lg font-semibold text-pink-600 dark:text-pink-400 mb-2">Frontend</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• Next.js 15.5 (App Router)</li>
                    <li>• React 19</li>
                    <li>• TypeScript</li>
                    <li>• Tailwind CSS</li>
                    <li>• Radix UI Components</li>
                    <li>• Lucide Icons</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-pink-600 dark:text-pink-400 mb-2">Backend</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• FastAPI (Python)</li>
                    <li>• Kohya SD-Scripts</li>
                    <li>• LyCORIS Extensions</li>
                    <li>• WD14 Tagger (ONNX)</li>
                    <li>• HuggingFace Hub</li>
                    <li>• WebSockets (Live Updates)</li>
                  </ul>
                </div>
              </div>
            </div>
          </GradientCard>

          {/* Credits & Love */}
          <GradientCard variant="watermelon" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Heart className="w-8 h-8 text-red-600 dark:text-red-400" />
                <h2 className="text-2xl font-bold text-foreground">Credits & Acknowledgments</h2>
              </div>
              <div className="text-foreground space-y-3">
                <p>
                  This project stands on the shoulders of giants. Special thanks to:
                </p>
                <ul className="space-y-2">
                  <li><strong className="text-foreground">Kohya SS</strong> - For the incredible sd-scripts training framework</li>
                  <li><strong className="text-foreground">LyCORIS Team</strong> - For advanced LoRA architectures (LoCon, LoHa, LoKr, etc.)</li>
                  <li><strong className="text-foreground">SmilingWolf</strong> - For the WD14 tagger models</li>
                  <li><strong className="text-foreground">Derrian Distro</strong> - For the original training backend architecture</li>
                  <li><strong className="text-foreground">Stability AI</strong> - For Stable Diffusion</li>
                  <li><strong className="text-foreground">Black Forest Labs</strong> - For Flux</li>
                </ul>
                <p className="pt-3 border-t border-border">
                  Built with love by a neurodivergent developer who wanted LoRA training to be
                  more accessible and less overwhelming. <span className="text-purple-600 dark:text-purple-400">♥</span>
                </p>
              </div>
            </div>
          </GradientCard>

          {/* Version & License */}
          <div className="bg-card/50 border border-border rounded-lg p-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Version</p>
                <p className="text-lg font-semibold text-foreground">0.1.0 - Alpha</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">License</p>
                <p className="text-lg font-semibold text-foreground">MIT</p>
              </div>
              <div>
                <a
                  href="https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
                >
                  <Github className="w-5 h-5" />
                  <span className="font-semibold">GitHub</span>
                </a>
              </div>
            </div>
          </div>

          {/* Connect With Us */}
          <GradientCard variant="watermelon" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Heart className="w-8 h-8 text-red-600 dark:text-red-400" />
                <h2 className="text-2xl font-bold text-foreground">Connect With Us</h2>
              </div>
              <p className="text-muted-foreground mb-6">
                Join our community, share your creations, and help shape the future of Ktiseos-Nyx.
              </p>

              <div className="space-y-6">
                {/* Community */}
                <div>
                  <h3 className="text-lg font-semibold text-purple-600 dark:text-purple-400 mb-3">Community</h3>
                  <div className="space-y-2">
                    <a
                      href="https://discord.gg/HhBSvM9gBY"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Ktiseos-Nyx Discord</span>
                    </a>
                    <a
                      href="https://discord.gg/5t2kYxt7An"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Earth & Dusk Media Discord</span>
                    </a>
                    <p className="text-xs text-muted-foreground ml-6">
                      Backup invite: <a href="https://discord.gg/MASBKnNFWh" className="text-cyan-600 dark:text-cyan-400 hover:underline">discord.gg/MASBKnNFWh</a>
                    </p>
                  </div>
                </div>

                {/* Support */}
                <div>
                  <h3 className="text-lg font-semibold text-purple-600 dark:text-purple-400 mb-3">Support Our Work</h3>
                  <div className="grid md:grid-cols-2 gap-2">
                    <a
                      href="https://www.patreon.com/earthndusk"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Patreon</span>
                    </a>
                    <a
                      href="https://ko-fi.com/earthnicity"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Ko-fi (Earthnicity)</span>
                    </a>
                    <a
                      href="https://ko-fi.com/duskfallcrew/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Ko-fi (Duskfall Crew)</span>
                    </a>
                    <a
                      href="https://ko-fi.com/OTNAngel/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Ko-fi (OTNAngel)</span>
                    </a>
                    <a
                      href="https://www.buymeacoffee.com/duskfallxcrew"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Buy Me A Pizza</span>
                    </a>
                    <a
                      href="https://duskfallcrew-shop.fourthwall.com/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Merchandise Shop</span>
                    </a>
                  </div>
                </div>

                {/* Social Media */}
                <div>
                  <h3 className="text-lg font-semibold text-purple-600 dark:text-purple-400 mb-3">Follow Us</h3>
                  <div className="grid md:grid-cols-2 gap-2">
                    <a
                      href="https://x.com/KtiseosNyx_AI"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <Twitter className="w-4 h-4" />
                      <span>X (Twitter)</span>
                    </a>
                    <a
                      href="https://bsky.app/profile/duskfallcrew.bsky.social"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Bluesky</span>
                    </a>
                    <a
                      href="https://www.threads.com/@duskfallcrew"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Threads</span>
                    </a>
                    <a
                      href="https://www.youtube.com/@duskfallmusic"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <Youtube className="w-4 h-4" />
                      <span>YouTube</span>
                    </a>
                    <a
                      href="https://twitch.tv/duskfallcrew"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <Twitch className="w-4 h-4" />
                      <span>Twitch</span>
                    </a>
                    <a
                      href="https://www.reddit.com/r/earthndusk/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Reddit</span>
                    </a>
                  </div>
                </div>

                {/* Referrals */}
                <div>
                  <h3 className="text-lg font-semibold text-purple-600 dark:text-purple-400 mb-3">Deployment Partners</h3>
                  <div className="space-y-2">
                    <a
                      href="https://cloud.vast.ai/?ref_id=70354"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>VastAI - Cloud GPU Instances</span>
                    </a>
                    <a
                      href="https://railway.com/?referralCode=EQxw4P"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Railway - Easy Deployment</span>
                    </a>
                    <a
                      href="https://pixai.art/referral?refCode=BRU9AUNA"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>PixAI Referral</span>
                    </a>
                  </div>
                </div>
              </div>

              <p className="mt-6 text-center text-purple-600 dark:text-purple-300 font-semibold italic">
                Join us on this journey. Welcome to Ktiseos Nyx.
              </p>
            </div>
          </GradientCard>
        </div>
      </div>
    </div>
  );
}
