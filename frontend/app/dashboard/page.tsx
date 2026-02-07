// frontend/app/dashboard/page.tsx

'use client';

import Link from 'next/link';
import { FileText, FolderOpen, Settings, Upload, Calculator, Wrench, Download, Zap, BookOpen, Info } from 'lucide-react';

// --- DASHBOARD COMPONENT ---
export default function Dashboard() {
  const cardClasses = "bg-card text-card-foreground backdrop-blur-sm border rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-accent transition-all cursor-pointer group";

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16">

        <div className="text-center mb-16">
            <h1 className="text-5xl font-bold text-foreground mb-4">
                Training Dashboard
            </h1>
            <p className="text-xl text-muted-foreground">
                Quick access to all your training tools
            </p>
        </div>


        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">

          {/* Example of a refactored card */}
          <Link href="/models">
            {/* CHANGED: All the hardcoded classes are replaced with our clean helper variable. */}
            <div className={cardClasses}>
              <Download className="w-12 h-12 text-cyan-400 mb-4 group-hover:scale-110 transition-transform" />
              {/* CHANGED: Titles now use the theme's main text color. */}
              <h2 className="text-xl font-semibold mb-2 text-foreground">Models & VAEs</h2>
              {/* CHANGED: Paragraphs use the theme's muted text color. */}
              <p className="text-muted-foreground text-sm">
                Download base models and VAEs
              </p>
            </div>
          </Link>

          {/* --- Repeat for all other cards --- */}

          <Link href="/files">
            <div className={cardClasses}>
              <FolderOpen className="w-12 h-12 text-blue-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">File Manager</h2>
              <p className="text-muted-foreground text-sm">
                Browse, upload, and manage your files
              </p>
            </div>
          </Link>

          <Link href="/dataset">
            <div className={cardClasses}>
              <Upload className="w-12 h-12 text-green-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Dataset</h2>
              <p className="text-muted-foreground text-sm">
                Upload and prepare training datasets
              </p>
            </div>
          </Link>

          <Link href="/training">
            <div className={cardClasses}>
              <Zap className="w-12 h-12 text-purple-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Training</h2>
              <p className="text-muted-foreground text-sm">
                Configure and monitor LoRA training
              </p>
            </div>
          </Link>

          <Link href="/calculator">
            <div className={cardClasses}>
              <Calculator className="w-12 h-12 text-indigo-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Calculator</h2>
              <p className="text-muted-foreground text-sm">
                Calculate optimal training steps
              </p>
            </div>
          </Link>

          <Link href="/utilities">
            <div className={cardClasses}>
              <Wrench className="w-12 h-12 text-orange-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Utilities</h2>
              <p className="text-muted-foreground text-sm">
                Manage and upload trained LoRAs
              </p>
            </div>
          </Link>

          <Link href="/settings">
            <div className={cardClasses}>
              <Settings className="w-12 h-12 text-pink-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Settings</h2>
              <p className="text-muted-foreground text-sm">
                Configure app preferences and defaults
              </p>
            </div>
          </Link>

          <Link href="/docs">
            <div className={cardClasses}>
              <BookOpen className="w-12 h-12 text-emerald-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">Documentation</h2>
              <p className="text-muted-foreground text-sm">
                Guides, tutorials, and best practices
              </p>
            </div>
          </Link>

          <Link href="/about">
            <div className={cardClasses}>
              <Info className="w-12 h-12 text-rose-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-foreground">About</h2>
              <p className="text-muted-foreground text-sm">
                Learn about the project and credits
              </p>
            </div>
          </Link>
        </div>

        {/* CHANGED: Quick Start section now also uses card styling for consistency. */}
        <div className={`mt-16 max-w-4xl mx-auto p-8 ${cardClasses}`}>
          <h3 className="text-2xl font-bold mb-4 text-foreground">Quick Start</h3>
          <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
            <li>Download a base model from <strong className="text-cyan-400">Models & VAEs</strong></li>
            <li>Upload your training images via <strong className="text-green-400">Dataset</strong></li>
            <li>Auto-tag images with WD14 tagger</li>
            <li>Configure training parameters in <strong className="text-purple-400">Training</strong></li>
            <li>Monitor progress in real-time (WebSockets!)</li>
            <li>Download your trained LoRA model</li>
          </ol>
        </div>

        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Powered by Kohya SD-Scripts • Built with Next.js & FastAPI</p>
        </div>
      </div>
    </div>
  );
}
