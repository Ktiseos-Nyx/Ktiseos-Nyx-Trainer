'use client';

import { Home, ArrowRight, GitMerge, Layers, Puzzle, Shrink } from 'lucide-react';
import Link from 'next/link';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';

const mergeTools = [
  {
    title: 'Merge Checkpoints',
    description: 'Merge two or three checkpoints (Basic weighted or Advanced 24-mode)',
    href: '/merge/checkpoints',
    icon: Layers,
    color: 'from-purple-500 to-blue-600',
  },
  {
    title: 'Merge LoRAs',
    description: 'Combine multiple LoRAs into a single LoRA file',
    href: '/merge/loras',
    icon: Puzzle,
    color: 'from-emerald-500 to-teal-600',
  },
  {
    title: 'LoRA → Checkpoint',
    description: 'Bake one or more LoRAs into a base checkpoint',
    href: '/merge/bake',
    icon: GitMerge,
    color: 'from-orange-500 to-red-600',
  },
  {
    title: 'Resize LoRA',
    description: 'Change the rank of a LoRA via SVD',
    href: '/merge/resize',
    icon: Shrink,
    color: 'from-cyan-500 to-blue-600',
  },
];

export default function MergeHubPage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Merge Tools', icon: <GitMerge className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-violet-400 via-fuchsia-400 to-cyan-400 bg-clip-text text-transparent">
            Merge Tools
          </h1>
          <p className="text-xl text-muted-foreground">
            Pick the type of merge you want to perform
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {mergeTools.map(tool => (
            <Link key={tool.href} href={tool.href} className="group">
              <div className="relative h-full rounded-xl border border-border bg-card p-6 transition-all duration-200 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5">
                <div className="flex items-start gap-4">
                  <div className={`shrink-0 rounded-lg bg-gradient-to-br ${tool.color} p-3`}>
                    <tool.icon className="h-6 w-6 text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold mb-1 group-hover:text-primary transition-colors">
                      {tool.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {tool.description}
                    </p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground shrink-0 mt-1 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* Recent merges placeholder */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold mb-2">Recent Merges</h2>
          <p className="text-sm text-muted-foreground">
            Your recent merge history will appear here once you run your first merge.
          </p>
        </div>
      </div>
    </div>
  );
}
