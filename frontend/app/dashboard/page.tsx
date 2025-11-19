'use client';

import Link from 'next/link';
import { FileText, FolderOpen, Settings, Upload, Calculator, Wrench, Download, Zap, BookOpen, Info } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-white mb-4">
            Training Dashboard
          </h1>
          <p className="text-xl text-gray-300">
            Quick access to all your training tools
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <Link href="/models">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Download className="w-12 h-12 text-cyan-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Models & VAEs</h2>
              <p className="text-gray-400 text-sm">
                Download base models and VAEs
              </p>
            </div>
          </Link>

          <Link href="/files">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <FolderOpen className="w-12 h-12 text-blue-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">File Manager</h2>
              <p className="text-gray-400 text-sm">
                Browse, upload, and manage your files
              </p>
            </div>
          </Link>

          <Link href="/dataset">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Upload className="w-12 h-12 text-green-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Dataset</h2>
              <p className="text-gray-400 text-sm">
                Upload and prepare training datasets
              </p>
            </div>
          </Link>

          <Link href="/training">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Zap className="w-12 h-12 text-purple-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Training</h2>
              <p className="text-gray-400 text-sm">
                Configure and monitor LoRA training
              </p>
            </div>
          </Link>

          <Link href="/calculator">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Calculator className="w-12 h-12 text-indigo-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Calculator</h2>
              <p className="text-gray-400 text-sm">
                Calculate optimal training steps
              </p>
            </div>
          </Link>

          <Link href="/utilities">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Wrench className="w-12 h-12 text-orange-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Utilities</h2>
              <p className="text-gray-400 text-sm">
                Manage and upload trained LoRAs
              </p>
            </div>
          </Link>

          <Link href="/settings">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Settings className="w-12 h-12 text-pink-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Settings</h2>
              <p className="text-gray-400 text-sm">
                Configure app preferences and defaults
              </p>
            </div>
          </Link>

          <Link href="/docs">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <BookOpen className="w-12 h-12 text-emerald-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">Documentation</h2>
              <p className="text-gray-400 text-sm">
                Guides, tutorials, and best practices
              </p>
            </div>
          </Link>

          <Link href="/about">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6 hover:shadow-xl hover:bg-slate-800/70 transition-all cursor-pointer group">
              <Info className="w-12 h-12 text-rose-400 mb-4 group-hover:scale-110 transition-transform" />
              <h2 className="text-xl font-semibold mb-2 text-white">About</h2>
              <p className="text-gray-400 text-sm">
                Learn about the project and credits
              </p>
            </div>
          </Link>
        </div>

        <div className="mt-16 max-w-4xl mx-auto bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-8">
          <h3 className="text-2xl font-bold mb-4 text-white">Quick Start</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-300">
            <li>Download a base model from <strong className="text-cyan-400">Models & VAEs</strong></li>
            <li>Upload your training images via <strong className="text-green-400">Dataset</strong></li>
            <li>Auto-tag images with WD14 tagger</li>
            <li>Configure training parameters in <strong className="text-purple-400">Training</strong></li>
            <li>Monitor progress in real-time (WebSockets!)</li>
            <li>Download your trained LoRA model</li>
          </ol>
        </div>

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>Powered by Kohya SD-Scripts • Built with Next.js & FastAPI</p>
        </div>
      </div>
    </div>
  );
}
