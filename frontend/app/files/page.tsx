'use client';

import dynamic from 'next/dynamic';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, FolderOpen, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

// 🚀 OPTIMIZATION: Lazy load FileManager (tree view is heavy)
const FileManager = dynamic(() => import('@/components/FileManager'), {
  loading: () => (
    <Card>
      <CardContent className="p-12 flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading file manager...</p>
        </div>
      </CardContent>
    </Card>
  ),
  ssr: false,
});

export default function FilesPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'File Manager', icon: <FolderOpen className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 text-foreground">
            File Manager
          </h1>
          <p className="text-xl text-muted-foreground">
            Browse, upload, and manage your files
          </p>
        </div>

        <FileManager />
      </div>
    </div>
  );
}
