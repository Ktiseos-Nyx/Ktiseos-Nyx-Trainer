import FileManager from '@/components/FileManager';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, FolderOpen } from 'lucide-react';

export default function FilesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-16">
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
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
            File Manager
          </h1>
          <p className="text-xl text-gray-300">
            Browse, upload, and manage your files
          </p>
        </div>

        <FileManager />
      </div>
    </div>
  );
}
