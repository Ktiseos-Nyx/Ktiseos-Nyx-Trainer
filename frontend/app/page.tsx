'use client';

import Link from 'next/link';
import { FileText, FolderOpen, Settings, Upload } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Ktiseos-Nyx LoRA Trainer
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Professional LoRA training with modern web interface
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            No more Jupyter race conditions! 🎉
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          <Link href="/files">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer">
              <FolderOpen className="w-12 h-12 text-blue-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">File Manager</h2>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Browse, upload, and manage your files
              </p>
            </div>
          </Link>

          <Link href="/dataset">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer">
              <Upload className="w-12 h-12 text-green-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Dataset</h2>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Upload and prepare training datasets
              </p>
            </div>
          </Link>

          <Link href="/training">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer">
              <Settings className="w-12 h-12 text-purple-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Training</h2>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Configure and monitor LoRA training
              </p>
            </div>
          </Link>

          <Link href="/configs">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer">
              <FileText className="w-12 h-12 text-orange-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Configs</h2>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Load and save training configurations
              </p>
            </div>
          </Link>
        </div>

        <div className="mt-16 max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
          <h3 className="text-2xl font-bold mb-4">Quick Start</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
            <li>Upload your training images via <strong>Dataset</strong></li>
            <li>Auto-tag images with WD14 tagger</li>
            <li>Configure training parameters in <strong>Training</strong></li>
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
