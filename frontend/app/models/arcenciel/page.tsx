'use client';

export const dynamic = 'force-dynamic';

import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'sonner';
import { sourcesAPI, SourceModelSummary, SourceModelVersion, SourceModelDetail } from '@/lib/api';
import {
  Download,
  Search,
  Filter,
  Loader2,
  Home,
  Globe,
  Layers,
} from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import BorderGlow from '@/components/BorderGlow';
import { AspectRatio } from '@/components/ui/aspect-ratio';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from '@/components/ui/card';
import Image from 'next/image';

const SOURCE_NAMES: Record<string, string> = {
  arcenciel: 'Arc En Ciel',
};

function sourceDisplayName(name: string): string {
  return SOURCE_NAMES[name] || name.charAt(0).toUpperCase() + name.slice(1);
}

function formatSize(kb: number | null | undefined): string {
  if (!kb) return '';
  const mb = kb / 1024;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

const SOURCE = 'arcenciel';
const KNOWN_TYPES = ['CHECKPOINT', 'LORA', 'VAE', 'OTHER'];

export default function SourcesBrowsePage() {
  const [selectedSource] = useState<string>(SOURCE);

  const [models, setModels] = useState<SourceModelSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [totalResults, setTotalResults] = useState(0);

  const [searchQuery, setSearchQuery] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedBaseModel, setSelectedBaseModel] = useState<string>('all');
  const [selectedSort, setSelectedSort] = useState('newest');
  const [showFilters, setShowFilters] = useState(true);

  const [showNsfw, setShowNsfw] = useState(true);
  const [downloadDestination, setDownloadDestination] = useState<'training' | 'comfyui'>('training');
  const [downloading, setDownloading] = useState<Set<string>>(new Set());

  const [baseModelOptions, setBaseModelOptions] = useState<string[]>([]);
  const [modelTypeOptions, setModelTypeOptions] = useState<string[]>([...KNOWN_TYPES]);

  // Version picker dialog
  const [selectedModel, setSelectedModel] = useState<SourceModelDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    sourcesAPI.list().then((srcs) => {
      if (srcs.length > 0) {
        setBaseModelOptions(srcs[0].base_model_classes || []);
      }
    }).catch((err) => console.error('Failed to load source info:', err));
    return () => {
      abortControllerRef.current?.abort();
      observer.current?.disconnect();
    };
  }, []);

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
    [loading, hasMore],
  );

  const loadModels = useCallback(async (pageNum: number, append: boolean = false) => {
    if (!selectedSource) return;
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    try {
      setLoading(true);
      const result = await sourcesAPI.search(selectedSource, {
        term: searchQuery,
        sort: selectedSort,
        page: pageNum,
        limit: 20,
        base_model: selectedBaseModel !== 'all' ? selectedBaseModel : undefined,
        model_type: selectedType !== 'all' ? selectedType : undefined,
        nsfw: showNsfw ? undefined : false,
      });

      if (result) {
        const items = result.items || [];
        if (append) {
          setModels((prev) => {
            const existingIds = new Set(prev.map((m) => m.model_id));
            const unique = items.filter((m: SourceModelSummary) => !existingIds.has(m.model_id));
            return [...prev, ...unique];
          });
        } else {
          setModels(items);
          const types = [...new Set([...KNOWN_TYPES, ...items.map((m: SourceModelSummary) => m.type).filter(Boolean)])] as string[];
          setModelTypeOptions(types.sort());
        }
        setTotalResults(items.length);
        setHasMore(result.has_more);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      console.error('Failed to load models:', err);
      toast.error(`Failed to load models: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedSource, searchQuery, selectedSort, selectedBaseModel, selectedType, showNsfw]);

  useEffect(() => {
    if (selectedSource) {
      setPage(1);
      setModels([]);
      setHasMore(true);
      loadModels(1, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSource, searchQuery, selectedSort, selectedBaseModel, selectedType, showNsfw]);

  useEffect(() => {
    if (page > 1) {
      loadModels(page, true);
    }
  }, [page, loadModels]);

  const handleSearch = () => {
    setSearchQuery(searchInput);
  };

  const handleModelClick = async (model: SourceModelSummary) => {
    try {
      setDetailLoading(true);
      const detail = await sourcesAPI.getModel(selectedSource, model.model_id);
      setSelectedModel(detail);
    } catch (err: any) {
      toast.error(`Failed to load model details: ${err.message}`);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDownload = async (version: SourceModelVersion) => {
    try {
      setDownloading((prev) => new Set(prev).add(`${selectedModel?.model_id}-${version.version_id}`));
      const comfyuiFolder =
        version.dest_type === 'vae' ? 'vae'
        : version.dest_type === 'lora' ? 'loras'
        : version.dest_type === 'embedding' ? 'embeddings'
        : downloadDestination === 'comfyui' ? 'checkpoints'
        : undefined;

      await sourcesAPI.download(
        selectedSource,
        version.model_id,
        version.version_id,
        downloadDestination,
        downloadDestination === 'comfyui' ? comfyuiFolder : undefined,
      );

      toast.success(`Download started: ${version.original_name || version.name || version.version_id}`);
      setSelectedModel(null);
    } catch (err: any) {
      toast.error(`Download failed: ${err.message}`);
    } finally {
      setDownloading((prev) => {
        const next = new Set(prev);
        next.delete(`${selectedModel?.model_id}-${version.version_id}`);
        return next;
      });
    }
  };



  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Models', href: '/models', icon: <Download className="w-4 h-4" /> },
            { label: 'Browse Models', icon: <Search className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
            Browse Models
          </h1>
          <p className="text-xl text-foreground">
            Discover and download models from Arc En Ciel creators
          </p>
        </div>

        {/* Source Badge */}
        <div className="mb-6 flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-sm text-purple-300">
            <Globe className="w-4 h-4" />
            {sourceDisplayName(selectedSource || '')}
          </div>
        </div>

        {/* Downloading Indicator */}
        {downloading.size > 0 && (
          <div className="mb-6 bg-gradient-to-r from-purple-900/40 to-pink-900/40 backdrop-blur-sm border-2 border-purple-500 rounded-lg p-6 shadow-lg">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Loader2 className="w-10 h-10 text-purple-400 animate-spin" />
                <div className="absolute inset-0 bg-purple-400/20 blur-xl rounded-full" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-1">
                  {downloading.size} Download{downloading.size > 1 ? 's' : ''} In Progress
                </h3>
                <p className="text-purple-200 text-sm">
                  Downloading from {sourceDisplayName(selectedSource)}… This may take several minutes.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Search Bar */}
        <div className="mb-6 space-y-3">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="w-full pl-10 pr-4 py-3 bg-card border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-purple-500"
                placeholder="Search models…"
              />
            </div>
            <Button onClick={handleSearch} className="px-6">
              Search
            </Button>
            <Button
              variant="outline"
              onClick={() => setShowFilters(!showFilters)}
              className="px-6"
            >
              <Filter className="w-5 h-5" />
              <span className="ml-2 hidden sm:inline">Filters</span>
            </Button>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="mb-6 bg-card/50 backdrop-blur-sm border border-border rounded-lg p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {/* Model Type */}
              <div>
                <label id="filter-type-label" className="block text-sm font-medium text-foreground mb-2">
                  Model Type
                </label>
                <Select value={selectedType} onValueChange={setSelectedType}>
                  <SelectTrigger aria-labelledby="filter-type-label" className="w-full">
                    <SelectValue placeholder="All Types" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    {modelTypeOptions.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Base Model */}
              <div>
                <label id="filter-base-label" className="block text-sm font-medium text-foreground mb-2">
                  Base Model
                </label>
                <Select value={selectedBaseModel} onValueChange={setSelectedBaseModel}>
                  <SelectTrigger aria-labelledby="filter-base-label" className="w-full">
                    <SelectValue placeholder="All Base Models" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Base Models</SelectItem>
                    {baseModelOptions.map((bm) => (
                      <SelectItem key={bm} value={bm}>
                        {bm}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Sort */}
              <div>
                <label id="filter-sort-label" className="block text-sm font-medium text-foreground mb-2">
                  Sort By
                </label>
                <Select value={selectedSort} onValueChange={setSelectedSort}>
                  <SelectTrigger aria-labelledby="filter-sort-label" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="newest">Newest</SelectItem>
                    <SelectItem value="popular">Most Popular</SelectItem>
                    <SelectItem value="downloads">Most Downloaded</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Download Destination */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Download To
                </label>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant={downloadDestination === 'training' ? 'secondary' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => setDownloadDestination('training')}
                  >
                    Trainer
                  </Button>
                  <Button
                    type="button"
                    variant={downloadDestination === 'comfyui' ? 'secondary' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => setDownloadDestination('comfyui')}
                  >
                    ComfyUI
                  </Button>
                </div>
              </div>

              {/* NSFW Toggle */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  NSFW Content
                </label>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant={showNsfw ? 'secondary' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => setShowNsfw(true)}
                  >
                    Show
                  </Button>
                  <Button
                    type="button"
                    variant={!showNsfw ? 'secondary' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => setShowNsfw(false)}
                  >
                    Hide
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {totalResults > 0 && (
          <div className="mb-4 text-sm text-muted-foreground">
            Showing {models.length} model{models.length !== 1 ? 's' : ''}
			<span className="text-xs text-muted-foreground/60 leading-tight line-clamp-2">
                  Click for version information & Downloads.
                  </span>
          </div>
        )}

        {/* Model Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {models.map((model, index) => (
            <BorderGlow key={model.model_id}>
              <Card
                ref={index === models.length - 1 ? lastModelRef : null}
                className="bg-card/50 backdrop-blur-sm border-1 overflow-hidden border-purple-500/50 transition-all cursor-pointer flex flex-col py-0 gap-0"
                onClick={() => handleModelClick(model)}
              >
                {/* Cover */}
                <div className="bg-input overflow-hidden relative">
                  <AspectRatio ratio={3 / 4}>
                    {model.cover_url ? (
                      <Image
                        src={model.cover_url}
                        alt={model.title}
                        fill
                        className="object-cover hover:scale-105 transition-transform duration-300"
                        unoptimized
                        loading="eager"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                        <Layers className="w-12 h-12" />
                      </div>
                    )}
                  </AspectRatio>
                  {model.nsfw && (
                    <div className="absolute top-2 right-2 px-2 py-1 bg-red-500/90 text-foreground text-xs font-bold rounded">
                      NSFW
                    </div>
                  )}
                  {model.type && (
                    <div className="absolute top-2 left-2 px-2 py-1 bg-purple-500/90 text-foreground text-xs font-bold rounded">
                      {model.type}
                    </div>
                  )}
                </div>

                <CardHeader className="px-3 pt-3 pb-2  ">
                  
                  <p className="text-xs text-foreground leading-tight truncate">
                    {[model.base_model, model.type].filter(Boolean).join(" — ")}
                  </p>
                </CardHeader>

                <CardContent className="px-3 pt-3 pb-2 leading-tight line-clamp-2 min-h-[68px]">
                  <span className="font-semibold text-foreground text-xs leading-tight line-clamp-2">
                    {model.title}
                  </span>
				  
				  {model.uploader && (
                    <p className="text-xs text-foreground text-xs leading-tight line-clamp-2">
                     {model.uploader}
                    </p>
                  )}
                </CardContent>

                <CardFooter className="px-3 pb-2 ">
            
                </CardFooter>
              </Card>
            </BorderGlow>
          ))}
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
          </div>
        )}

        {/* No Results */}
        {!loading && models.length === 0 && selectedSource && (
          <div className="text-center py-12">
            <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <p className="text-xl text-muted-foreground">No models found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Try adjusting your search or filters
            </p>
          </div>
        )}
      </div>

      {/* Version Picker Dialog */}
      <Dialog open={selectedModel !== null} onOpenChange={(open) => { if (!open) setSelectedModel(null); }}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
          {detailLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
            </div>
          ) : selectedModel ? (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl">{selectedModel.title}</DialogTitle>
                {selectedModel.uploader && (
                  <p className="text-sm text-muted-foreground/70">by {selectedModel.uploader}</p>
                )}
                {selectedModel.description && (selectedModel.description.includes('<') ? (
                  <div
                    className="line-clamp-5 text-sm text-muted-foreground [&_a]:text-purple-400 [&_a]:underline prose prose-sm max-w-none mt-2"
                    dangerouslySetInnerHTML={{ __html: selectedModel.description }}
                  />
                ) : (
                  <DialogDescription className="line-clamp-3 mt-2">
                    {selectedModel.description}
                  </DialogDescription>
                ))}
              </DialogHeader>

              {/* Model Info */}
              <div className="flex flex-wrap gap-2 text-sm text-muted-foreground mb-4">
                {selectedModel.type && (
                  <span className="px-2 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-purple-300">
                    {selectedModel.type}
                  </span>
                )}
                {selectedModel.base_model && (
                  <span className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 rounded text-blue-300">
                    {selectedModel.base_model}
                  </span>
                )}
                {selectedModel.nsfw && (
                  <span className="px-2 py-1 bg-red-500/20 border border-red-500/30 rounded text-red-300">
                    NSFW
                  </span>
                )}
              </div>

              {/* Versions */}
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground">Versions</h4>
                {selectedModel.versions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No versions available</p>
                ) : (
                  selectedModel.versions.map((version) => {
                    const downloadKey = `${selectedModel.model_id}-${version.version_id}`;
                    const isDownloading = downloading.has(downloadKey);

                    return (
                      <Card key={version.version_id} className="bg-card/50 border-border">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1 min-w-0">
                              <h5 className="font-medium text-foreground truncate">
                                {version.name || version.version_id}
                              </h5>
                              <div className="flex flex-wrap gap-2 mt-1 text-xs text-muted-foreground">
                                {version.base_model && (
                                  <span>{version.base_model}</span>
                                )}
                                {version.file_size_kb && (
                                  <span>{formatSize(version.file_size_kb)}</span>
                                )}
                                {version.status && (
                                  <span className={version.status === 'PUBLISHED' ? 'text-green-400' : 'text-yellow-400'}>
                                    {version.status}
                                  </span>
                                )}
                                {version.original_name && (
                                  <span className="truncate max-w-[200px]">{version.original_name}</span>
                                )}
                              </div>
                            </div>
                            <Button
                              size="sm"
                              disabled={isDownloading || (version.status && version.status !== 'PUBLISHED')}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDownload(version);
                              }}
                              className="flex-shrink-0"
                            >
                              {isDownloading ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <Download className="w-4 h-4" />
                              )}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })
                )}
              </div>

              <DialogFooter className="text-xs text-muted-foreground">
                Source: {sourceDisplayName(selectedModel.source)}
              </DialogFooter>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
