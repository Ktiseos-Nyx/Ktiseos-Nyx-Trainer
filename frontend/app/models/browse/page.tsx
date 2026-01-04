'use client';

/**
 * Civitai Model Browser
 *
 * Attribution: Inspired by sd-webui-civbrowser extension
 * https://github.com/SignalFlagZ/sd-webui-civbrowser
 */

// Force dynamic rendering - don't pre-render this page at build time
export const dynamic = 'force-dynamic';

import { useState, useEffect, useRef, useCallback } from 'react';
import { civitaiAPI, CivitaiModel, API_BASE } from '@/lib/api';
import {
  Download,
  Search,
  Filter,
  X,
  Heart,
  Eye,
  Star,
  ChevronDown,
  Loader2,
  Home,
  AlertTriangle,
  Key,
  Lock,
} from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import Image from 'next/image';
import Link from 'next/link';

export default function CivitaiBrowsePage() {
  // API Key check
  const [hasApiKey, setHasApiKey] = useState<boolean | null>(null);
  const [checkingApiKey, setCheckingApiKey] = useState(true);
  const [showApiKeyWarning, setShowApiKeyWarning] = useState(true);

  // State
  const [models, setModels] = useState<CivitaiModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [cursor, setCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [totalResults, setTotalResults] = useState(0);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [searchMode, setSearchMode] = useState<'keyword' | 'tag' | 'creator'>('keyword');
  const [selectedType, setSelectedType] = useState<string>('All');
  const [selectedBaseModel, setSelectedBaseModel] = useState<string>('All');
  const [selectedSort, setSelectedSort] = useState('Highest Rated');
  const [selectedPeriod, setSelectedPeriod] = useState('AllTime');
  const [allowNSFW, setAllowNSFW] = useState(false);
  const [showFilters, setShowFilters] = useState(true);

  // Browsing Level (bitmask system like civbrowser)
  const nsfwLevels = {
    'PG': 1,
    'PG-13': 2,
    'R': 4,
    'X': 8,
    'XXX': 16,
  };
  const [browsingLevel, setBrowsingLevel] = useState<number>(1); // Default: PG only

  // Download state
  const [downloading, setDownloading] = useState<Set<number>>(new Set());

  // Browsing level helpers
  const toggleBrowsingLevel = (level: number) => {
    setBrowsingLevel((prev) => prev ^ level); // XOR to toggle bit
  };

  const isBrowsingLevelEnabled = (level: number) => {
    return (browsingLevel & level) !== 0;
  };

  const matchBrowsingLevel = (modelLevel: number) => {
    // If browsing level is 0, show everything
    if (browsingLevel === 0) return true;
    // Check if model's level matches any enabled browsing levels
    return (modelLevel & browsingLevel) !== 0;
  };

  // Infinite scroll
  const observer = useRef<IntersectionObserver | null>(null);
  const lastModelRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (loading) return;
      if (observer.current) observer.current.disconnect();

      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore) {
          setPage((prev) => prev + 1);
        }
      });

      if (node) observer.current.observe(node);
    },
    [loading, hasMore]
  );

  // Check for API key on mount
  useEffect(() => {
    const checkApiKey = async () => {
      try {
        const response = await fetch(`${API_BASE}/settings/user`);
        if (response.ok) {
          const data = await response.json();
          setHasApiKey(data.settings?.has_civitai_api_key || false);
        } else {
          setHasApiKey(false);
        }
      } catch (err) {
        console.error('Failed to check API key:', err);
        setHasApiKey(false);
      } finally {
        setCheckingApiKey(false);
      }
    };

    checkApiKey();
  }, []);

  // Load models
  const loadModels = async (pageNum: number, append: boolean = false) => {
    try {
      setLoading(true);

      const params: any = {
        limit: 20,
        sort: selectedSort,
        period: selectedPeriod,
        nsfw: allowNSFW,
      };

      // Use cursor-based pagination for search queries, page-based otherwise
      if (searchQuery) {
        // Map search mode to correct API parameter
        if (searchMode === 'keyword') {
          params.query = searchQuery;
        } else if (searchMode === 'tag') {
          params.tag = searchQuery;
        } else if (searchMode === 'creator') {
          params.username = searchQuery;
        }

        if (cursor && append) {
          params.cursor = cursor;
        }
      } else {
        params.page = pageNum;
      }

      if (selectedType !== 'All') params.types = selectedType;
      if (selectedBaseModel !== 'All') params.baseModel = selectedBaseModel;

      const response = await civitaiAPI.browse(params);

      if (response.success) {
        const newModels = response.data.items || [];
        const metadata = response.data.metadata || {};

        // Update cursor for next request
        setCursor(metadata.nextCursor || null);

        if (append) {
          // Deduplicate models by ID to prevent React key warnings
          setModels((prev) => {
            const existingIds = new Set(prev.map(m => m.id));
            const uniqueNewModels = newModels.filter((m: CivitaiModel) => !existingIds.has(m.id));
            return [...prev, ...uniqueNewModels];
          });
        } else {
          setModels(newModels);
        }

        setTotalResults(metadata.totalItems || 0);
        setHasMore(newModels.length === 20 && (metadata.nextCursor || pageNum < 100));
      }
    } catch (err: any) {
      console.error('Failed to load models:', err);
      alert(`Failed to load models: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Initial load (now works with or without API key)
  useEffect(() => {
    // Only load if we've finished checking for API key
    if (hasApiKey !== null) {
      setPage(1);
      setCursor(null); // Reset cursor when filters change
      setModels([]);
      setHasMore(true);
      loadModels(1, false);
    }
  }, [searchQuery, selectedType, selectedBaseModel, selectedSort, selectedPeriod, allowNSFW, browsingLevel, hasApiKey]);

  // Load more on page change
  useEffect(() => {
    if (page > 1) {
      loadModels(page, true);
    }
  }, [page]);

  // Handle search
  const handleSearch = () => {
    setSearchQuery(searchInput);
  };

  // Handle download
  const handleDownload = async (model: CivitaiModel) => {
    const latestVersion = model.modelVersions[0];
    if (!latestVersion || !latestVersion.files || latestVersion.files.length === 0) {
      alert('No downloadable files found for this model');
      return;
    }

    const primaryFile = latestVersion.files[0];
    const downloadUrl = primaryFile.downloadUrl || latestVersion.downloadUrl;

    if (!downloadUrl) {
      alert('Download URL not available');
      return;
    }

    try {
      setDownloading((prev) => new Set(prev).add(model.id));

      const modelType = model.type === 'VAE' ? 'vae' : 'model';

      await civitaiAPI.download(
        model.id,
        latestVersion.id,
        downloadUrl,
        primaryFile.name,
        modelType
      );

      alert(`Download started: ${primaryFile.name}`);
    } catch (err: any) {
      alert(`Download failed: ${err.message}`);
    } finally {
      setDownloading((prev) => {
        const newSet = new Set(prev);
        newSet.delete(model.id);
        return newSet;
      });
    }
  };

  // Model types
  const modelTypes = ['All', 'Checkpoint', 'LORA', 'LoCon', 'VAE', 'TextualInversion'];

  // Base models (comprehensive list from Civitai)
  const baseModels = [
    { value: 'All', label: 'All Base Models' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'SD 1.4', label: 'SD 1.4' },
    { value: 'SD 1.5', label: 'SD 1.5' },
    { value: 'SD 1.5 LCM', label: 'SD 1.5 LCM' },
    { value: 'SD 1.5 Hyper', label: 'SD 1.5 Hyper' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'SD 2.0', label: 'SD 2.0' },
    { value: 'SD 2.0 768', label: 'SD 2.0 768' },
    { value: 'SD 2.1', label: 'SD 2.1' },
    { value: 'SD 2.1 768', label: 'SD 2.1 768' },
    { value: 'SD 2.1 Unclip', label: 'SD 2.1 Unclip' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'SDXL 0.9', label: 'SDXL 0.9' },
    { value: 'SDXL 1.0', label: 'SDXL 1.0' },
    { value: 'SDXL 1.0 LCM', label: 'SDXL 1.0 LCM' },
    { value: 'SDXL Turbo', label: 'SDXL Turbo' },
    { value: 'SDXL Lightning', label: 'SDXL Lightning' },
    { value: 'SDXL Distilled', label: 'SDXL Distilled' },
    { value: 'SDXL Hyper', label: 'SDXL Hyper' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'Pony', label: 'Pony (SDXL)' },
    { value: 'Illustrious', label: 'Illustrious (SDXL)' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'SD 3', label: 'SD 3' },
    { value: 'SD 3 Medium', label: 'SD 3 Medium' },
    { value: 'SD 3.5 Large', label: 'SD 3.5 Large' },
    { value: 'SD 3.5 Large Turbo', label: 'SD 3.5 Large Turbo' },
    { value: 'SD 3.5 Medium', label: 'SD 3.5 Medium' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'Flux.1 D', label: 'Flux.1 Dev' },
    { value: 'Flux.1 S', label: 'Flux.1 Schnell' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'AuraFlow', label: 'AuraFlow' },
    { value: 'Hunyuan 1', label: 'HunyuanDiT' },
    { value: 'Kolors', label: 'Kolors' },
    { value: 'Lumina', label: 'Lumina' },
    { value: 'PixArt a', label: 'PixArt Alpha' },
    { value: 'PixArt E', label: 'PixArt Sigma' },
    { value: '', label: '─────────────────', disabled: true },
    { value: 'Playground v2', label: 'Playground v2' },
    { value: 'Playground v2.5', label: 'Playground v2.5' },
    { value: 'SVD', label: 'Stable Video Diffusion' },
    { value: 'SVD XT', label: 'SVD XT' },
    { value: 'Stable Cascade', label: 'Stable Cascade' },
  ];

  // Sort options
  const sortOptions = [
    'Highest Rated',
    'Most Downloaded',
    'Newest',
    'Most Liked',
    'Most Discussed',
  ];

  // Period options
  const periodOptions = ['AllTime', 'Year', 'Month', 'Week', 'Day'];

  // Show loading while checking API key
  if (checkingApiKey) {
    return (
      <div className="min-h-screen bg-background py-16 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-cyan-400 mx-auto mb-4" />
          <p className="text-muted-foreground">Checking API key...</p>
        </div>
      </div>
    );
  }


  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Models', href: '/models', icon: <Download className="w-4 h-4" /> },
            { label: 'Browse Civitai', icon: <Search className="w-4 h-4" /> },
          ]}
        />

        {/* API Key Info Banner (Optional - Dismissible) */}
        {!hasApiKey && showApiKeyWarning && (
          <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 relative">
            <button
              onClick={() => setShowApiKeyWarning(false)}
              className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
              aria-label="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-start gap-3">
              <Key className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-foreground mb-1">Civitai API Key (Optional)</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  <strong>Browsing works without an API key!</strong> Adding one is optional but gives you:
                  higher rate limits, faster browsing, and access to NSFW-rated content.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Link href="/settings">
                    <button className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-md text-sm font-medium transition-colors">
                      <Key className="w-4 h-4" />
                      Add API Key (Optional)
                    </button>
                  </Link>
                  <a
                    href="https://civitai.com/user/account"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 border border-border hover:bg-accent text-foreground rounded-md text-sm font-medium transition-colors"
                  >
                    Get Free API Key →
                  </a>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Browse Civitai
          </h1>
          <p className="text-xl text-foreground">
            Discover and download models from the Civitai community
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Inspired by{' '}
            <a
              href="https://github.com/SignalFlagZ/sd-webui-civbrowser"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:text-cyan-300 underline"
            >
              sd-webui-civbrowser
            </a>
          </p>
        </div>

        {/* Active Downloads Status */}
        {downloading.size > 0 && (
          <div className="mb-6 bg-gradient-to-r from-cyan-900/40 to-blue-900/40 backdrop-blur-sm border-2 border-cyan-500 rounded-lg p-6 shadow-lg">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
                <div className="absolute inset-0 bg-cyan-400/20 blur-xl rounded-full"></div>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-1">
                  {downloading.size} Download{downloading.size > 1 ? 's' : ''} In Progress
                </h3>
                <p className="text-cyan-200 text-sm">
                  Downloading from Civitai... This may take several minutes depending on file size.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Search Bar */}
        <div className="mb-6 space-y-3">
          {/* Search Mode Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Search by:</span>
            <div className="flex gap-1 bg-card/50 border border-border rounded-lg p-1">
              <button
                onClick={() => setSearchMode('keyword')}
                className={`px-4 py-1.5 text-sm font-medium rounded transition-all ${
                  searchMode === 'keyword'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Keyword
              </button>
              <button
                onClick={() => setSearchMode('tag')}
                className={`px-4 py-1.5 text-sm font-medium rounded transition-all ${
                  searchMode === 'tag'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Tag
              </button>
              <button
                onClick={() => setSearchMode('creator')}
                className={`px-4 py-1.5 text-sm font-medium rounded transition-all ${
                  searchMode === 'creator'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Creator
              </button>
            </div>
          </div>

          {/* Search Input */}
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="w-full pl-10 pr-4 py-3 bg-card border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                placeholder={
                  searchMode === 'keyword'
                    ? 'Search model names...'
                    : searchMode === 'tag'
                    ? 'Search by tag (e.g., "anime", "realistic")...'
                    : 'Search by creator username...'
                }
              />
            </div>
            <button
              onClick={handleSearch}
              className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-foreground font-semibold rounded-lg transition-all"
            >
              Search
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-6 py-3 bg-card border border-border text-foreground rounded-lg hover:bg-accent transition-all flex items-center gap-2"
            >
              <Filter className="w-5 h-5" />
              Filters
            </button>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="mb-6 bg-card/50 backdrop-blur-sm border border-border rounded-lg p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {/* Model Type */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Model Type
                </label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-border text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500"
                >
                  {modelTypes.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>

              {/* Base Model */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Base Model
                </label>
                <select
                  value={selectedBaseModel}
                  onChange={(e) => setSelectedBaseModel(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-border text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500"
                >
                  {baseModels.map((model, index) => (
                    <option
                      key={`${model.value}-${index}`}
                      value={model.value}
                      disabled={model.disabled}
                      className={model.disabled ? 'text-muted-foreground' : ''}
                    >
                      {model.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sort */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Sort By
                </label>
                <select
                  value={selectedSort}
                  onChange={(e) => setSelectedSort(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-border text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500"
                >
                  {sortOptions.map((sort) => (
                    <option key={sort} value={sort}>
                      {sort}
                    </option>
                  ))}
                </select>
              </div>

              {/* Period */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Time Period
                </label>
                <select
                  value={selectedPeriod}
                  onChange={(e) => setSelectedPeriod(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-border text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500"
                >
                  {periodOptions.map((period) => (
                    <option key={period} value={period}>
                      {period}
                    </option>
                  ))}
                </select>
              </div>

              {/* NSFW API Toggle */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Include NSFW in API
                </label>
                <button
                  onClick={() => setAllowNSFW(!allowNSFW)}
                  className={`w-full px-3 py-2 border rounded-lg transition-all flex items-center justify-center gap-2 ${
                    allowNSFW
                      ? 'bg-orange-900/50 border-orange-500/50 text-orange-200'
                      : 'bg-input border-border text-foreground'
                  }`}
                >
                  {allowNSFW ? (
                    <>
                      <AlertTriangle className="w-4 h-4" />
                      NSFW Enabled
                    </>
                  ) : (
                    <>
                      <Lock className="w-4 h-4" />
                      SFW Only
                    </>
                  )}
                </button>
                <p className="text-xs text-muted-foreground mt-1">
                  Adds mature content to API results
                </p>
              </div>
            </div>

            {/* Browsing Level Checkboxes */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-foreground">
                  Browsing Level (Content Ratings)
                </label>
                <button
                  onClick={() => setBrowsingLevel(browsingLevel === 0 ? 1 : 0)}
                  className="text-xs text-cyan-400 hover:text-cyan-300"
                >
                  {browsingLevel === 0 ? 'Reset' : 'Show All'}
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
                {Object.entries(nsfwLevels).map(([label, value]) => (
                  <label
                    key={label}
                    className={`flex items-center gap-2 px-3 py-2 border rounded cursor-pointer transition-colors ${
                      isBrowsingLevelEnabled(value)
                        ? 'bg-cyan-900/50 border-cyan-500 text-cyan-100'
                        : 'bg-input border-border text-foreground hover:border-border'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isBrowsingLevelEnabled(value)}
                      onChange={() => toggleBrowsingLevel(value)}
                      className="w-4 h-4 rounded border-border bg-card text-cyan-500 focus:ring-2 focus:ring-cyan-500"
                    />
                    <span className="text-sm font-medium">{label}</span>
                  </label>
                ))}
              </div>

              <p className="text-xs text-muted-foreground mt-2">
                Select which content ratings to show. Combine multiple for broader results.
              </p>
            </div>

            {/* Content Rating Info */}
            <div className="mt-4 p-4 bg-blue-900/20 dark:bg-blue-900/20 bg-blue-100/50 border border-blue-500/30 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-foreground">
                  <p className="font-semibold mb-1">How Filtering Works:</p>
                  <div className="text-xs space-y-2">
                    <p><strong>1. NSFW Toggle:</strong> Tells Civitai API to include mature content in results (SFW + NSFW). When OFF, only safe content returns.</p>
                    <p><strong>2. Browsing Level:</strong> Filters which ratings YOU see from those results. Select multiple to combine (e.g., PG + R = show both).</p>
                    <p className="mt-2"><strong>Rating Levels:</strong></p>
                    <ul className="ml-4 space-y-0.5">
                      <li><strong>PG:</strong> Safe/General</li>
                      <li><strong>PG-13:</strong> Mild themes</li>
                      <li><strong>R:</strong> Mature content</li>
                      <li><strong>X:</strong> Explicit</li>
                      <li><strong>XXX:</strong> Maximum adult</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Count */}
        {totalResults > 0 && (
          <div className="mb-4 text-sm text-muted-foreground">
            Found {totalResults.toLocaleString()} models
          </div>
        )}

        {/* Models Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {models.map((model, index) => {
            const latestVersion = model.modelVersions[0];
            const previewImage = latestVersion?.images?.find((img) => !img.nsfw) || latestVersion?.images?.[0];
            const isDownloading = downloading.has(model.id);

            return (
              <div
                key={model.id}
                ref={index === models.length - 1 ? lastModelRef : null}
                className="bg-card/50 backdrop-blur-sm border border-border rounded-lg overflow-hidden hover:border-cyan-500/50 transition-all group"
              >
                {/* Preview Image */}
                <div className="relative aspect-[3/4] bg-input overflow-hidden">
                  {previewImage ? (
                    <Image
                      src={previewImage.url}
                      alt={model.name}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform duration-300"
                      unoptimized
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                      No Preview
                    </div>
                  )}

                  {/* NSFW Badge */}
                  {model.nsfw && (
                    <div className="absolute top-2 right-2 px-2 py-1 bg-red-500/90 text-foreground text-xs font-bold rounded">
                      NSFW
                    </div>
                  )}

                  {/* Model Type Badge */}
                  <div className="absolute top-2 left-2 px-2 py-1 bg-cyan-500/90 text-foreground text-xs font-bold rounded">
                    {model.type}
                  </div>
                </div>

                {/* Model Info */}
                <div className="p-4">
                  <h3 className="font-bold text-foreground text-lg mb-2 line-clamp-2">
                    {model.name}
                  </h3>

                  <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                    by {model.creator.username}
                  </p>

                  {/* Stats */}
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-4">
                    <div className="flex items-center gap-1">
                      <Download className="w-3 h-3" />
                      {((model.stats?.downloadCount || 0) / 1000).toFixed(1)}k
                    </div>
                    <div className="flex items-center gap-1">
                      <Heart className="w-3 h-3" />
                      {((model.stats?.favoriteCount || 0) / 1000).toFixed(1)}k
                    </div>
                    <div className="flex items-center gap-1">
                      <Star className="w-3 h-3" />
                      {(model.stats?.rating || 0).toFixed(1)}
                    </div>
                  </div>

                  {/* Download Button */}
                  <button
                    onClick={() => handleDownload(model)}
                    disabled={isDownloading}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-foreground font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDownloading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Downloading...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Download
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
          </div>
        )}

        {/* No Results */}
        {!loading && models.length === 0 && (
          <div className="text-center py-12">
            <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <p className="text-xl text-muted-foreground">No models found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Try adjusting your search or filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
