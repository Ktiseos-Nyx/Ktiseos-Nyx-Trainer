'use client';

import { Wrench, Home } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MergeLoRAsTab } from '@/components/merge/MergeLoRAs';
import { MergeCheckpointsTab } from '@/components/merge/MergeCheckpoints';
import { LoRAToCheckpointTab } from '@/components/merge/LoRAToCheckpoint';
import { ResizeLoRATab } from '@/components/merge/ResizeLoRA';

export default function UtilitiesPage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Utilities', icon: <Wrench className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-red-400 to-orange-400 bg-clip-text text-transparent">
            Model Utilities
          </h1>
          <p className="text-xl text-muted-foreground">
            Merge LoRAs, merge checkpoints, resize and bake models
          </p>
        </div>

        {/* Transition banner */}
        <div className="mb-6 rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-700 dark:text-blue-400">
          <strong>🆕 New Merge Hub!</strong>{' '}
          Dedicated merge pages are now at{' '}
          <a href="/merge" className="underline font-medium">/merge</a> — same tools, more room, better layout.
        </div>

        <Tabs defaultValue="merge-lora">
          <TabsList className="mb-6">
            <TabsTrigger value="merge-lora">Merge LoRAs</TabsTrigger>
            <TabsTrigger value="merge-checkpoint">Merge Checkpoints</TabsTrigger>
            <TabsTrigger value="lora-to-checkpoint">LoRA → Checkpoint</TabsTrigger>
            <TabsTrigger value="resize">Resize LoRA</TabsTrigger>
          </TabsList>
          <TabsContent value="merge-lora"><MergeLoRAsTab /></TabsContent>
          <TabsContent value="merge-checkpoint"><MergeCheckpointsTab /></TabsContent>
          <TabsContent value="lora-to-checkpoint"><LoRAToCheckpointTab /></TabsContent>
          <TabsContent value="resize"><ResizeLoRATab /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
