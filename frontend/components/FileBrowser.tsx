'use client';

import { useState, useEffect } from 'react';
import { fileAPI, DirectoryListing, FileInfo } from '@/lib/api';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Folder, File, ArrowLeft, Home, Check } from 'lucide-react';

interface FileBrowserProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (path: string) => void;
  mode: 'file' | 'directory';
  title?: string;
  description?: string;
  startPath?: string;
  fileFilter?: (file: FileInfo) => boolean; // Optional filter for files
}

export default function FileBrowser({
  open,
  onOpenChange,
  onSelect,
  mode,
  title = 'Select Path',
  description = 'Browse and select a path',
  startPath,
  fileFilter,
}: FileBrowserProps) {
  const [defaultWorkspace, setDefaultWorkspace] = useState<string>('');
  const [currentPath, setCurrentPath] = useState(startPath || '');
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  // Fetch default workspace on mount
  useEffect(() => {
    const fetchDefaultWorkspace = async () => {
      try {
        const { path } = await fileAPI.getDefaultWorkspace();
        setDefaultWorkspace(path);
        if (!startPath) {
          setCurrentPath(path);
        }
      } catch (err) {
        console.error('Failed to fetch default workspace:', err);
      }
    };
    fetchDefaultWorkspace();
  }, [startPath]);

  // Load directory contents
  const loadDirectory = async (path: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fileAPI.list(path);
      setListing(data);
      setCurrentPath(data.path);
      setSelectedItem(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load directory');
      console.error('Failed to load directory:', err);
    } finally {
      setLoading(false);
    }
  };

  // Load initial directory when opened
  useEffect(() => {
    if (open && startPath) {
      loadDirectory(startPath);
    }
  }, [open, startPath]);

  // Navigate to parent directory
  const goUp = () => {
    if (listing?.parent) {
      loadDirectory(listing.parent);
    }
  };

  // Navigate to home
  const goHome = () => {
    loadDirectory(defaultWorkspace || startPath || '/');
  };

  // Handle item click
  const handleItemClick = (item: FileInfo) => {
    if (item.type === 'dir') {
      // Navigate into directory
      loadDirectory(item.path);
    } else {
      // Select file (if in file mode)
      if (mode === 'file') {
        setSelectedItem(item.path);
      }
    }
  };

  // Handle item double-click
  const handleItemDoubleClick = (item: FileInfo) => {
    if (item.type === 'dir' && mode === 'directory') {
      // Select directory
      onSelect(item.path);
      onOpenChange(false);
    } else if (item.type === 'file' && mode === 'file') {
      // Select file
      onSelect(item.path);
      onOpenChange(false);
    }
  };

  // Handle select button
  const handleSelect = () => {
    if (mode === 'directory') {
      // Select current directory
      onSelect(currentPath);
    } else if (selectedItem) {
      // Select selected file
      onSelect(selectedItem);
    }
    onOpenChange(false);
  };

  // Filter files if filter provided
  const filteredFiles = listing?.files.filter((file) => {
    if (file.type === 'dir') return true; // Always show directories
    if (!fileFilter) return true; // No filter, show all
    return fileFilter(file);
  }) || [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Navigation Bar */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={goHome}
              disabled={loading}
            >
              <Home className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={goUp}
              disabled={loading || !listing?.parent}
            >
              <ArrowLeft className="w-4 h-4" />
            </Button>
            <Input
              value={currentPath}
              onChange={(e) => setCurrentPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  loadDirectory(currentPath);
                }
              }}
              className="flex-1"
              placeholder={defaultWorkspace ? `${defaultWorkspace}/path` : "/path/to/directory"}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadDirectory(currentPath)}
              disabled={loading}
            >
              Go
            </Button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="text-sm text-red-500 bg-red-50 border border-red-200 rounded p-3">
              {error}
            </div>
          )}

          {/* File List */}
          <ScrollArea className="h-[400px] border rounded">
            {loading ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                Loading...
              </div>
            ) : filteredFiles.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                No items found
              </div>
            ) : (
              <div className="p-2 space-y-1">
                {filteredFiles.map((item) => (
                  <div
                    key={item.path}
                    onClick={() => handleItemClick(item)}
                    onDoubleClick={() => handleItemDoubleClick(item)}
                    className={`
                      flex items-center gap-3 p-2 rounded cursor-pointer
                      hover:bg-gray-100 transition-colors
                      ${selectedItem === item.path ? 'bg-blue-50 border border-blue-200' : ''}
                    `}
                  >
                    {item.type === 'dir' ? (
                      <Folder className="w-5 h-5 text-blue-500 flex-shrink-0" />
                    ) : (
                      <File className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    )}
                    <span className="flex-1 truncate">{item.name}</span>
                    {item.type === 'file' && (
                      <span className="text-xs text-gray-400">
                        {(item.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>

          {/* Selection Info */}
          {mode === 'directory' && (
            <div className="text-sm text-gray-600">
              Current directory: <code className="bg-gray-100 px-2 py-1 rounded">{currentPath}</code>
            </div>
          )}
          {mode === 'file' && selectedItem && (
            <div className="text-sm text-gray-600">
              Selected: <code className="bg-gray-100 px-2 py-1 rounded">{selectedItem}</code>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSelect}
            disabled={mode === 'file' && !selectedItem}
          >
            <Check className="w-4 h-4 mr-2" />
            Select {mode === 'directory' ? 'Directory' : 'File'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
