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
} from 'lucide-react';
import { fileAPI, FileInfo, DirectoryListing } from '@/lib/api';

export default function FileManager() {
  // Start at home directory for local dev, /workspace for cloud/VastAI
  const [currentPath, setCurrentPath] = useState('~');
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);

  // Load directory listing
  const loadDirectory = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fileAPI.list(path);
      setListing(data);
      setCurrentPath(data.path);
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
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDirectory(currentPath);
  }, []);

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
      return <Folder className="w-5 h-5 text-blue-500" />;
    }
    if (file.is_image) {
      return <Image className="w-5 h-5 text-green-500" />;
    }
    if (file.name.endsWith('.toml') || file.name.endsWith('.json')) {
      return <FileText className="w-5 h-5 text-orange-500" />;
    }
    return <File className="w-5 h-5 text-gray-500" />;
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
    <div className="h-full flex flex-col bg-white dark:bg-gray-900">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold">File Manager</h2>
          <div className="flex gap-2">
            <button
              onClick={handleCreateDir}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
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
              className="text-blue-500 hover:text-blue-700"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
          )}
          <span className="text-gray-600 dark:text-gray-400 font-mono">
            {currentPath}
          </span>
        </div>
      </div>

      {/* Drop zone overlay */}
      <div {...getRootProps()} className="flex-1 relative">
        <input {...getInputProps()} />

        {isDragActive && (
          <div className="absolute inset-0 bg-blue-500 bg-opacity-20 border-4 border-dashed border-blue-500 flex items-center justify-center z-50">
            <div className="text-center">
              <Upload className="w-16 h-16 text-blue-500 mx-auto mb-4" />
              <p className="text-xl font-semibold text-blue-700">
                Drop files here to upload
              </p>
            </div>
          </div>
        )}

        {/* File listing */}
        <div className="p-4 overflow-auto h-full">
          {loading && (
            <div className="text-center py-8">
              <p className="text-gray-500">Loading...</p>
            </div>
          )}

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              {error}
            </div>
          )}

          {listing && !loading && (
            <div className="space-y-1">
              {listing.files.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Empty directory</p>
                  <p className="text-sm mt-2">Drag & drop files here to upload</p>
                </div>
              )}

              {listing.files.map((file) => (
                <div
                  key={file.path}
                  className={`
                    flex items-center justify-between p-3 rounded hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer
                    ${selectedFile?.path === file.path ? 'bg-blue-50 dark:bg-blue-900' : ''}
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
                      <div className="font-medium">{file.name}</div>
                      <div className="text-xs text-gray-500">
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
                        className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                        title="Download"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(file);
                      }}
                      className="p-2 hover:bg-red-100 dark:hover:bg-red-900 rounded text-red-500"
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
      <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800">
        <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
          <Upload className="w-4 h-4" />
          Drag & drop files anywhere to upload to current directory
        </p>
      </div>
    </div>
  );
}
