'use client';

import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Folder,
  File,
  Image,
  Upload,
  Trash2,
  Plus,
  ArrowLeft,
  FileText,
  Download,
  FolderOpen,
} from 'lucide-react';
import { fileAPI, FileInfo, DirectoryListing } from '@/lib/api';
import { Tree, Folder as TreeFolder, File as TreeFile, type TreeViewElement } from '@/components/ui/file-tree';

export default function FileManager() {
  // Start at default workspace (will be set after fetching from backend)
  const [currentPath, setCurrentPath] = useState<string>('');
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [treeElements, setTreeElements] = useState<TreeViewElement[]>([]);
  const [showTree, setShowTree] = useState(true);

  // Build tree elements from directory listing
  const buildTreeElements = useCallback((listing: DirectoryListing | null): TreeViewElement[] => {
    if (!listing) return [];

    const folders = listing.files.filter(f => f.type === 'dir');
    const files = listing.files.filter(f => f.type === 'file');

    const elements: TreeViewElement[] = [];

    // Add folders first
    folders.forEach(folder => {
      elements.push({
        id: folder.path,
        name: folder.name,
        isSelectable: true,
        children: [], // Will be populated when expanded
      });
    });

    // Add files (no children)
    files.forEach(file => {
      elements.push({
        id: file.path,
        name: file.name,
        isSelectable: true,
      });
    });

    return elements;
  }, []);

  // Load directory listing
  const loadDirectory = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fileAPI.list(path);
      setListing(data);
      setCurrentPath(data.path);
      setTreeElements(buildTreeElements(data));
    } catch (err) {
      // Backend not available - show helpful message instead of error
      const errorMsg = err instanceof Error ? err.message : 'Failed to load directory';
      if (errorMsg.includes('fetch') || errorMsg.includes('Failed to fetch')) {
        setError('Backend API not running. Start the backend server to browse files.');
      } else {
        setError(errorMsg);
      }
      // Set empty listing to show UI
      setListing({ path: path, files: [], parent: null });
      setTreeElements([]);
    } finally {
      setLoading(false);
    }
  }, [buildTreeElements]);

  // Initialize with default workspace path
  useEffect(() => {
    const initializeWorkspace = async () => {
      try {
        const { path } = await fileAPI.getDefaultWorkspace();
        setCurrentPath(path);
        loadDirectory(path);
      } catch (err) {
        console.error('Failed to get default workspace:', err);
        setError('Unable to connect to backend');
      }
    };
    initializeWorkspace();
  }, [loadDirectory]);

  // File drop zone for uploads
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      for (const file of acceptedFiles) {
        try {
          await fileAPI.upload(file, currentPath);
        } catch (err) {
          console.error(`Failed to upload ${file.name}:`, err);
        }
      }
      // Reload directory after uploads
      loadDirectory(currentPath);
    },
    [currentPath, loadDirectory]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    noClick: true, // Only activate on drop, not click
  });

  // Navigate to directory
  const navigateTo = (path: string) => {
    loadDirectory(path);
  };

  // Delete file/directory
  const handleDelete = async (file: FileInfo) => {
    if (!confirm(`Delete ${file.name}?`)) return;

    try {
      await fileAPI.delete(file.path);
      loadDirectory(currentPath);
    } catch (err) {
      alert(`Failed to delete: ${err}`);
    }
  };

  // Create new directory
  const handleCreateDir = async () => {
    const name = prompt('Directory name:');
    if (!name) return;

    try {
      await fileAPI.mkdir(currentPath, name);
      loadDirectory(currentPath);
    } catch (err) {
      alert(`Failed to create directory: ${err}`);
    }
  };

  // Get icon for file type
  const getFileIcon = (file: FileInfo) => {
    if (file.type === 'dir') {
      return <Folder className="w-5 h-5 text-primary" />;
    }
    if (file.is_image) {
      return <Image className="w-5 h-5 text-green-600 dark:text-green-400" />;
    }
    if (file.name.endsWith('.toml') || file.name.endsWith('.json')) {
      return <FileText className="w-5 h-5 text-orange-600 dark:text-orange-400" />;
    }
    return <File className="w-5 h-5 text-muted-foreground" />;
  };

  // Format file size
  const formatSize = (bytes: number) => {
    if (bytes === 0) return '-';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold text-foreground">File Manager</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setShowTree(!showTree)}
              className="flex items-center gap-2 px-4 py-2 bg-muted text-foreground rounded hover:bg-muted/80 border border-border transition-colors"
            >
              {showTree ? <FolderOpen className="w-4 h-4" /> : <Folder className="w-4 h-4" />}
              {showTree ? 'Hide Tree' : 'Show Tree'}
            </button>
            <button
              onClick={handleCreateDir}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Folder
            </button>
          </div>
        </div>

        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm">
          {listing?.parent && (
            <button
              onClick={() => navigateTo(listing.parent!)}
              className="text-primary hover:text-primary/80 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
          )}
          <span className="text-muted-foreground font-mono">
            {currentPath}
          </span>
        </div>
      </div>

      {/* Drop zone overlay */}
      <div {...getRootProps()} className="flex-1 relative flex overflow-hidden">
        <input {...getInputProps()} />

        {isDragActive && (
          <div className="absolute inset-0 bg-primary/20 border-4 border-dashed border-primary flex items-center justify-center z-50">
            <div className="text-center">
              <Upload className="w-16 h-16 text-primary mx-auto mb-4" />
              <p className="text-xl font-semibold text-foreground">
                Drop files here to upload
              </p>
            </div>
          </div>
        )}

        {/* Tree View Sidebar */}
        {showTree && (
          <div className="w-64 border-r border-border bg-card overflow-y-auto">
            <div className="p-2">
              <Tree
                initialSelectedId={currentPath}
                elements={treeElements}
                className="h-full"
              >
                {treeElements.map((element) => {
                  const hasChildren = element.children && element.children.length > 0;

                  if (hasChildren) {
                    return (
                      <TreeFolder
                        key={element.id}
                        element={element.name}
                        value={element.id}
                        isSelectable={element.isSelectable}
                      >
                        {element.children!.map((child) => {
                          const childHasChildren = child.children && child.children.length > 0;

                          if (childHasChildren) {
                            return (
                              <TreeFolder
                                key={child.id}
                                element={child.name}
                                value={child.id}
                                isSelectable={child.isSelectable}
                              />
                            );
                          } else {
                            return (
                              <TreeFile
                                key={child.id}
                                value={child.id}
                                onClick={() => navigateTo(child.id)}
                              >
                                <span className="ml-1">{child.name}</span>
                              </TreeFile>
                            );
                          }
                        })}
                      </TreeFolder>
                    );
                  } else {
                    return (
                      <TreeFile
                        key={element.id}
                        value={element.id}
                        onClick={() => navigateTo(element.id)}
                        isSelectable={element.isSelectable}
                      >
                        <span className="ml-1">{element.name}</span>
                      </TreeFile>
                    );
                  }
                })}
              </Tree>
            </div>
          </div>
        )}

        {/* File listing */}
        <div className="flex-1 p-4 overflow-auto">
          {loading && (
            <div className="text-center py-8">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          )}

          {error && (
            <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded">
              {error}
            </div>
          )}

          {listing && !loading && (
            <div className="space-y-1">
              {listing.files.length === 0 && (
                <div className="text-center py-12 text-muted-foreground">
                  <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Empty directory</p>
                  <p className="text-sm mt-2">Drag & drop files here to upload</p>
                </div>
              )}

              {listing.files.map((file) => (
                <div
                  key={file.path}
                  className={`
                    flex items-center justify-between p-3 rounded hover:bg-muted cursor-pointer transition-colors
                    ${selectedFile?.path === file.path ? 'bg-muted border border-border' : ''}
                  `}
                  onClick={() => {
                    if (file.type === 'dir') {
                      navigateTo(file.path);
                    } else {
                      setSelectedFile(file);
                    }
                  }}
                >
                  <div className="flex items-center gap-3 flex-1">
                    {getFileIcon(file)}
                    <div>
                      <div className="font-medium text-foreground">{file.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {formatSize(file.size)} •{' '}
                        {new Date(file.modified * 1000).toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {file.type === 'file' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(`/api/files/download${file.path}`, '_blank');
                        }}
                        className="p-2 hover:bg-muted rounded transition-colors"
                        title="Download"
                      >
                        <Download className="w-4 h-4 text-foreground" />
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(file);
                      }}
                      className="p-2 hover:bg-destructive/10 rounded text-destructive transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Upload hint */}
      <div className="border-t border-border p-4 bg-muted">
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <Upload className="w-4 h-4" />
          Drag & drop files anywhere to upload to current directory
        </p>
      </div>
    </div>
  );
}
