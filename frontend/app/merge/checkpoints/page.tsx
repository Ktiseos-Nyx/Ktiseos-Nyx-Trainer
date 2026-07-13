'use client';

import { useState } from 'react';
import { Home, Layers } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MergeCheckpointsTab } from '@/components/merge/MergeCheckpoints';
import { MergeAdvancedTab } from '@/components/merge/MergeAdvanced';

export default function MergeCheckpointsPage() {
  const [tab, setTab] = useState('basic');

  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs items={[
          { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
          { label: 'Merge Tools', href: '/merge', icon: <Layers className="w-4 h-4" /> },
          { label: 'Merge Checkpoints' },
        ]} />
        <div className="mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Merge Checkpoints
          </h1>
          <p className="text-muted-foreground">Merge two or three checkpoints — Basic weighted average or Advanced 24-mode merge</p>
        </div>

        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>
          <TabsContent value="basic"><MergeCheckpointsTab /></TabsContent>
          <TabsContent value="advanced"><MergeAdvancedTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
