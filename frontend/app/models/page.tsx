'use client';

import { useState, useEffect } from 'react';
import { modelsAPI, ModelFile, PopularModel } from '@/lib/api';
import { Download, Trash2, HardDrive, Loader2, ExternalLink, Home, Sparkles, Search } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import Link from 'next/link';

export default function ModelsPage() {
  const [activeTab, setActiveTab] = useState<'download' | 'manage'>('download');

  // Download state
  const [downloadUrl, setDownloadUrl] = useState('');
  const [downloadType, setDownloadType] = useState<'model' | 'vae'>('model');
  const [modelType, setModelType] = useState('sdxl');
  const [downloading, setDownloading] = useState(false);
  const [downloadResult, setDownloadResult] = useState<any>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // File lists
  const [models, setModels] = useState<ModelFile[]>([]);
  const [vaes, setVaes] = useState<ModelFile[]>([]);
  const [loading, setLoading] = useState(false);

  // Popular models
  const [popularModels, setPopularModels] = useState<any>(null);

  // Load files and popular models
  useEffect(() => {
    loadFiles();
    loadPopular();
  }, []);

  const loadFiles = async () => {
    try {
      setLoading(true);
      const data = await modelsAPI.list();
      setModels(data.models || []);
      setVaes(data.vaes || []);
    } catch (err) {
      console.error('Failed to load files:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadPopular = async () => {
    try {
      const data = await modelsAPI.popular();
      setPopularModels(data);
    } catch (err) {
      console.error('Failed to load popular models:', err);
    }
  };

  const handleDownload = async () => {
    if (!downloadUrl.trim()) {
      setDownloadError('Please enter a URL');
      return;
    }

    try {
      setDownloading(true);
      setDownloadError(null);
      setDownloadResult(null);

      // Extract filename from URL for better UX
      const filename = downloadUrl.split('/').pop() || 'file';

      const result = await modelsAPI.download(downloadUrl, downloadType, modelType);

      setDownloadResult(result);
      setDownloadUrl('');

      // Reload file list
      setTimeout(() => loadFiles(), 1000);
    } catch (err: any) {
      setDownloadError(err.message || 'Download failed');
    } finally {
      setDownloading(false);
    }
  };

  const handleDelete = async (file: ModelFile) => {
    if (!confirm(`Delete ${file.name}?`)) return;

    try {
      await modelsAPI.delete(file.type, file.name);
      loadFiles();
    } catch (err: any) {
      alert(`Failed to delete: ${err.message}`);
    }
  };

  const usePopularUrl = (url: string, type: 'model' | 'vae' = 'model') => {
    setDownloadUrl(url);
    setDownloadType(type);
    setActiveTab('download');
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Models & VAEs', icon: <Download className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Models & VAEs
              </h1>
              <p className="text-xl text-gray-300">
                Download base models and VAEs from HuggingFace or Civitai
              </p>
            </div>
            <Link href="/models/browse">
              <button className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-lg transition-all shadow-lg hover:shadow-xl hover:scale-105 whitespace-nowrap">
                <Search className="w-5 h-5" />
                Browse Civitai
              </button>
            </Link>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="border-b border-slate-700">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('download')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'download'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                <Download className="w-5 h-5 inline mr-2" />
                Download
              </button>
              <button
                onClick={() => setActiveTab('manage')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'manage'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                <HardDrive className="w-5 h-5 inline mr-2" />
                Manage ({models.length + vaes.length})
              </button>
            </nav>
          </div>
        </div>

        {/* Download Tab */}
        {activeTab === 'download' && (
          <div className="space-y-6">
            {/* Active Download Status - Prominent */}
            {downloading && (
              <div className="bg-gradient-to-r from-cyan-900/40 to-blue-900/40 backdrop-blur-sm border-2 border-cyan-500 rounded-lg p-6 shadow-lg animate-pulse">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <Loader2 className="w-12 h-12 text-cyan-400 animate-spin" />
                    <div className="absolute inset-0 bg-cyan-400/20 blur-xl rounded-full"></div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-white mb-1">Download In Progress</h3>
                    <p className="text-cyan-200 text-sm">
                      Downloading {downloadType === 'model' ? 'model' : 'VAE'} from {downloadUrl.includes('huggingface') ? 'HuggingFace' : downloadUrl.includes('civitai') ? 'Civitai' : 'source'}...
                    </p>
                    <p className="text-cyan-300/70 text-xs mt-2">
                      This may take several minutes depending on file size and network speed. Please do not close this page.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-cyan-900/20 backdrop-blur-sm border border-cyan-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-cyan-100">
                  <p className="font-semibold mb-1">Quick Start:</p>
                  <p>
                    Want a popular model? Scroll down to the <span className="text-cyan-400 font-semibold">Popular Models</span> section below and click "Use" on any model to auto-fill the download form.
                  </p>
                </div>
              </div>
            </div>

            {/* Download Form */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-white mb-4">Download Model or VAE</h2>

              <div className="space-y-4">
                {/* Type Selection */}
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Download Type
                    </label>
                    <select
                      value={downloadType}
                      onChange={(e) => setDownloadType(e.target.value as 'model' | 'vae')}
                      className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                    >
                      <option value="model">Base Model</option>
                      <option value="vae">VAE</option>
                    </select>
                  </div>

                  {downloadType === 'model' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Model Type
                      </label>
                      <select
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                      >
                        <option value="sdxl">SDXL</option>
                        <option value="sd15">SD 1.5</option>
                        <option value="flux">Flux</option>
                        <option value="sd3.5">SD 3.5</option>
                      </select>
                    </div>
                  )}
                </div>

                {/* URL Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Download URL (HuggingFace or Civitai)
                  </label>
                  <input
                    type="text"
                    value={downloadUrl}
                    onChange={(e) => setDownloadUrl(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                    placeholder="https://huggingface.co/.../model.safetensors"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Supports HuggingFace (resolve links) and Civitai (direct download links).{' '}
                    <Link href="/models/browse" className="text-cyan-400 hover:text-cyan-300 underline">
                      Browse Civitai Models →
                    </Link>
                  </p>
                </div>
              </div>

              {/* Download Button */}
              <button
                onClick={handleDownload}
                disabled={downloading || !downloadUrl.trim()}
                className="mt-6 w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white px-6 py-3 rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {downloading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-5 h-5" />
                    Download {downloadType === 'model' ? 'Model' : 'VAE'}
                  </>
                )}
              </button>

              {/* Download Error */}
              {downloadError && (
                <div className="mt-4 bg-red-900/50 border-2 border-red-500 text-red-200 px-6 py-4 rounded-lg shadow-lg">
                  <p className="font-bold text-lg mb-1">Download Failed</p>
                  <p className="text-sm">{downloadError}</p>
                </div>
              )}

              {/* Download Success */}
              {downloadResult && downloadResult.success && (
                <div className="mt-4 bg-gradient-to-r from-green-900/50 to-emerald-900/50 border-2 border-green-500 rounded-lg shadow-lg p-6 animate-in fade-in duration-500">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
                      <Download className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-xl text-green-100">Download Complete!</p>
                      <p className="text-green-200 text-sm mt-1">
                        {downloadResult.file_name} ({downloadResult.size_mb} MB) saved successfully
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Popular Models */}
            {popularModels && (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
                <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                  <Sparkles className="w-6 h-6 text-yellow-400" />
                  Popular Models
                </h2>

                <div className="space-y-4">
                  {Object.entries(popularModels.models || {}).map(([type, modelsList]: [string, any]) => (
                    <div key={type}>
                      <h3 className="text-lg font-semibold text-cyan-400 mb-2 uppercase">{type}</h3>
                      <div className="space-y-2">
                        {modelsList.map((model: PopularModel, idx: number) => (
                          <div
                            key={idx}
                            className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 hover:border-cyan-500/50 transition-colors"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <h4 className="font-semibold text-white">{model.name}</h4>
                                <p className="text-sm text-gray-400 mt-1">{model.description}</p>
                              </div>
                              <button
                                onClick={() => usePopularUrl(model.url, 'model')}
                                className="ml-4 flex items-center gap-1 text-cyan-400 hover:text-cyan-300 text-sm"
                              >
                                <ExternalLink className="w-4 h-4" />
                                Use
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}

                  {/* Popular VAEs */}
                  <div>
                    <h3 className="text-lg font-semibold text-cyan-400 mb-2">VAES</h3>
                    <div className="space-y-2">
                      {(popularModels.vaes || []).map((vae: PopularModel, idx: number) => (
                        <div
                          key={idx}
                          className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 hover:border-cyan-500/50 transition-colors"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="font-semibold text-white">{vae.name}</h4>
                              <p className="text-sm text-gray-400 mt-1">{vae.description}</p>
                            </div>
                            <button
                              onClick={() => usePopularUrl(vae.url, 'vae')}
                              className="ml-4 flex items-center gap-1 text-cyan-400 hover:text-cyan-300 text-sm"
                            >
                              <ExternalLink className="w-4 h-4" />
                              Use
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Manage Tab */}
        {activeTab === 'manage' && (
          <div className="space-y-6">
            {/* Models */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-white mb-4">
                Base Models ({models.length})
              </h2>

              {models.length === 0 ? (
                <div className="text-center py-12">
                  <Download className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                  <p className="text-gray-400">No models downloaded yet</p>
                  <button
                    onClick={() => setActiveTab('download')}
                    className="mt-4 text-cyan-400 hover:text-cyan-300"
                  >
                    Download your first model →
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {models.map((model) => (
                    <div
                      key={model.path}
                      className="flex items-center justify-between bg-slate-900/50 border border-slate-700 rounded-lg p-4 hover:border-cyan-500/50 transition-colors"
                    >
                      <div className="flex-1">
                        <h3 className="font-semibold text-white">{model.name}</h3>
                        <p className="text-sm text-gray-400 mt-1">
                          {model.size_mb.toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={() => handleDelete(model)}
                        className="ml-4 text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* VAEs */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-white mb-4">
                VAE Files ({vaes.length})
              </h2>

              {vaes.length === 0 ? (
                <div className="text-center py-12">
                  <Download className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                  <p className="text-gray-400">No VAEs downloaded yet</p>
                  <button
                    onClick={() => setActiveTab('download')}
                    className="mt-4 text-cyan-400 hover:text-cyan-300"
                  >
                    Download a VAE →
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {vaes.map((vae) => (
                    <div
                      key={vae.path}
                      className="flex items-center justify-between bg-slate-900/50 border border-slate-700 rounded-lg p-4 hover:border-cyan-500/50 transition-colors"
                    >
                      <div className="flex-1">
                        <h3 className="font-semibold text-white">{vae.name}</h3>
                        <p className="text-sm text-gray-400 mt-1">
                          {vae.size_mb.toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={() => handleDelete(vae)}
                        className="ml-4 text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
