'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Loader2, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  SectionCard,
  ErrorBanner,
  SuccessBanner,
  loadModelFiles,
  ListedFile,
  LoRAFile,
  utilitiesAPI,
} from '@/components/merge/shared';

export function MergeLoRAsTab() {
  const [availableFiles, setAvailableFiles] = useState<ListedFile[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [modelType, setModelType] = useState<'sd' | 'sdxl' | 'flux' | 'svd'>('sdxl');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        setAvailableFiles(await loadModelFiles(
          [{ label: 'output/', dir: dirs.output }, { label: 'ComfyUI', dir: dirs.comfyui_loras }],
          'safetensors',
        ));
      } catch (e) {
        console.error('Failed to load LoRA files:', e);
        setLoadError('Could not list LoRAs. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (selectedLoras.length >= 2) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`merged_${modelType}_${date}.safetensors`);
    }
  }, [selectedLoras.length, modelType]);

  const add = (file: LoRAFile) => {
    if (!selectedLoras.find(l => l.path === file.path))
      setSelectedLoras(prev => [...prev, { path: file.path, name: file.name, ratio: 1.0 }]);
  };
  const remove = (path: string) => setSelectedLoras(prev => prev.filter(l => l.path !== path));
  const setRatio = (path: string, ratio: number) =>
    setSelectedLoras(prev => prev.map(l => l.path === path ? { ...l, ratio: Math.max(0, Math.min(2, ratio)) } : l));

  const handleMerge = async () => {
    if (selectedLoras.length < 2) { setError('Select at least 2 LoRAs to merge'); return; }
    if (!outputPath) { setError('Provide an output filename'); return; }
    try {
      setMerging(true); setError(null); setResult(null);
      const res = await utilitiesAPI.mergeLora(
        selectedLoras.map(l => ({ path: l.path, ratio: l.ratio })),
        outputPath, modelType, 'cpu', 'fp16', 'float',
      );
      if (res.success) { setResult(res); setSelectedLoras([]); }
      else setError(res.message ?? 'Merge failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Merge failed');
    } finally { setMerging(false); }
  };

  return (
    <div className="space-y-6">
      <SectionCard title="Model Architecture">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(['sd', 'sdxl', 'flux', 'svd'] as const).map(t => (
            <Button
              key={t}
              variant={modelType === t ? 'default' : 'outline'}
              onClick={() => setModelType(t)}
            >
              {t.toUpperCase()}
            </Button>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Available LoRAs">
        <p className="text-xs text-muted-foreground">From your trainer output/ and ComfyUI loras folder.</p>
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        {availableFiles.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No LoRA files found in output/ or ComfyUI.</p>
        ) : (
          <div className="grid md:grid-cols-2 gap-2 max-h-80 overflow-y-auto">
            {availableFiles.map(f => (
              <Button
                key={f.path}
                variant="outline"
                onClick={() => add(f)}
                disabled={selectedLoras.some(l => l.path === f.path)}
                className="justify-between h-auto py-2 px-3"
              >
                <div className="text-left min-w-0">
                  <div className="truncate text-xs font-medium">{f.name}</div>
                  <div className="text-xs text-muted-foreground">{f.source} · {f.size_formatted}</div>
                </div>
                {selectedLoras.some(l => l.path === f.path) && (
                  <CheckCircle className="h-4 w-4 text-green-500 shrink-0 ml-2" />
                )}
              </Button>
            ))}
          </div>
        )}
      </SectionCard>

      {selectedLoras.length > 0 && (
        <SectionCard title={`Selected LoRAs (${selectedLoras.length})`}>
          <div className="space-y-2">
            {selectedLoras.map(l => (
              <div key={l.path} className="flex items-center gap-3 rounded-md border border-border px-3 py-2">
                <span className="flex-1 truncate text-xs font-medium">{l.name}</span>
                <Label className="text-xs text-muted-foreground shrink-0">Ratio</Label>
                <Input
                  type="number"
                  value={l.ratio}
                  onChange={e => setRatio(l.path, parseFloat(e.target.value) || 0)}
                  step="0.1" min={0} max={2}
                  className="w-20 h-7 text-xs text-center"
                />
                <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => remove(l.path)}>
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            ))}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Output filename</Label>
            <Input
              value={outputPath}
              onChange={e => setOutputPath(e.target.value)}
              placeholder="merged.safetensors"
              className="text-xs font-mono"
            />
            <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
          </div>
        </SectionCard>
      )}

      <Button
        onClick={handleMerge}
        disabled={merging || selectedLoras.length < 2 || !outputPath}
        className="w-full"
      >
        {merging ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Merging…</> : `Merge ${selectedLoras.length} LoRAs`}
      </Button>

      {error && <ErrorBanner message={error} />}
      {result && (result as { success: boolean }).success && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Merge successful</div>
          <div>Output: {String((result as Record<string, unknown>).output_path)}</div>
          <div>Merged {String((result as Record<string, unknown>).merged_count)} LoRAs · {String((result as Record<string, unknown>).file_size_mb)} MB</div>
        </SuccessBanner>
      )}
    </div>
  );
}
