'use client';

import Link from 'next/link';
import { Github, Twitter, Heart, Zap } from 'lucide-react';

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border bg-card">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 md:gap-8">
          {/* Left: Logo, Copyright & License */}
          <div className="text-sm text-muted-foreground text-center md:text-left flex-shrink-0">
            <Link href="/" className="flex items-center gap-2 font-bold text-lg mb-2 justify-center md:justify-start">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 shadow-lg">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent whitespace-nowrap">
                KNX TRAINER
              </span>
            </Link>
            <p className="whitespace-nowrap">© {currentYear} Ktiseos-Nyx Trainer</p>
            <p className="text-xs mt-1 whitespace-nowrap">
              Licensed under{' '}
              <a
                href="https://opensource.org/licenses/MIT"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
              >
                MIT License
              </a>
            </p>
          </div>

          {/* Center: Made with love */}
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground whitespace-nowrap">
            <span>Made with</span>
            <Heart className="w-4 h-4 text-red-500 dark:text-red-400 fill-red-500 dark:fill-red-400" />
            <span>by the community</span>
          </div>

          {/* Right: Social Icons */}
          <div className="flex items-center justify-center md:justify-end gap-4 flex-shrink-0">
            <a
              href="https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://x.com/KtiseosNyx_AI"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              aria-label="X (Twitter)"
            >
              <Twitter className="w-5 h-5" />
            </a>
          </div>
        </div>

        {/* Bottom: Tech stack */}
        <div className="mt-4 pt-4 border-t border-border text-center text-xs text-muted-foreground">
          Powered by Kohya SD-Scripts • Built with Next.js & FastAPI
        </div>
      </div>
    </footer>
  );
}
