'use client';

import Link from 'next/link';
import { Github, Twitter, Heart } from 'lucide-react';

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border bg-card">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Left: Copyright & License */}
          <div className="text-sm text-muted-foreground text-center md:text-left md:w-1/3">
            <p>© {currentYear} Ktiseos-Nyx Trainer</p>
            <p className="text-xs mt-1">
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
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground md:w-1/3">
            <span>Made with</span>
            <Heart className="w-4 h-4 text-red-500 dark:text-red-400 fill-red-500 dark:fill-red-400" />
            <span>by the community</span>
          </div>

          {/* Right: Social Icons */}
          <div className="flex items-center justify-end gap-4 md:w-1/3">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://twitter.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              aria-label="Twitter"
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
