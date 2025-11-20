import FileManager from '@/components/FileManager';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, FolderOpen } from 'lucide-react';

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
