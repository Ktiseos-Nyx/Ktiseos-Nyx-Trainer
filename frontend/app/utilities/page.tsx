'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, LoRAFile } from '@/lib/api';
import { Wrench, CheckCircle, Loader2, Minimize2, Home, Trash2 } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export default function UtilitiesPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Utilities', icon: <Wrench className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-red-400 to-orange-400 bg-clip-text text-transparent">
            Model Utilities
          </h1>
          <p className="text-xl text-muted-foreground">
            Merge LoRAs, merge checkpoints, resize and bake models
          </p>
        </div>

        <Tabs defaultValue="merge-lora">
          <TabsList className="mb-6">
            <TabsTrigger value="merge-lora">Merge LoRAs</TabsTrigger>
            <TabsTrigger value="merge-checkpoint">Merge Checkpoints</TabsTrigger>
            <TabsTrigger value="lora-to-checkpoint">LoRA → Checkpoint</TabsTrigger>
            <TabsTrigger value="resize">Resize LoRA</TabsTrigger>
          </TabsList>
          <TabsContent value="merge-lora"><MergeLoRATab /></TabsContent>
          <TabsContent value="merge-checkpoint"><MergeCheckpointTab /></TabsContent>
          <TabsContent value="lora-to-checkpoint"><LoRAToCheckpointTab /></TabsContent>
          <TabsContent value="resize"><ResizeLoRATab /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// ─── Shared helpers ────────────────────────────────────────────────────────────

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-lg p-6 space-y-4">
      <h2 className="text-lg font-semibold text-foreground">{title}</h2>
      {children}
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      {message}
    </div>
  );
}

function SuccessBanner({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-green-500/50 bg-green-500/10 px-4 py-3 text-sm text-green-700 dark:text-green-400 space-y-1">
      {children}
    </div>
  );
}

// A listed model file, tagged with which source directory it came from
// (e.g. trainer "output/" vs "ComfyUI") so the merge lists can label it.
type ListedFile = LoRAFile & { source?: string };

/**
 * List model files across multiple source directories and tag each with its
 * source label. Sources whose dir is undefined (e.g. ComfyUI not configured)
 * are skipped, so the merge tools degrade gracefully. Returns a flat list.
 */
async function loadModelFiles(
  sources: Array<{ label: string; dir?: string }>,
  ext: string,
): Promise<ListedFile[]> {
  const present = sources.filter((s): s is { label: string; dir: string } => Boolean(s.dir));
  const lists = await Promise.all(
    present.map(async ({ label, dir }) => {
      const res = await utilitiesAPI.listLoraFiles(dir, ext, 'date');
      return res.success ? res.files.map((f: LoRAFile) => ({ ...f, source: label })) : [];
    }),
  );
  return lists.flat();
}

// ─── Merge LoRAs ──────────────────────────────────────────────────────────────

function MergeLoRATab() {
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

// ─── Merge Checkpoints ────────────────────────────────────────────────────────

function MergeCheckpointTab() {
  const [mergeMode, setMergeMode] = useState<'basic' | 'weighted'>('basic');
  const [availableFiles, setAvailableFiles] = useState<ListedFile[]>([]);
  const [selected, setSelected] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [unetOnly, setUnetOnly] = useState(false);
  const [precision, setPrecision] = useState('float');
  const [savePrecision, setSavePrecision] = useState('fp16');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Block-weighted mode state
  const [modelA, setModelA] = useState('');
  const [modelB, setModelB] = useState('');
  const [modelC, setModelC] = useState('');
  const [bwMode, setBwMode] = useState<'weight' | 'add' | 'triple' | 'twice'>('weight');
  const [bwPreset, setBwPreset] = useState('');
  const [bwPresetC, setBwPresetC] = useState('');
  const [bwPresets, setBwPresets] = useState<Record<string, number[]>>({});
  const [bwPresetsSdxl, setBwPresetsSdxl] = useState<Record<string, number[]>>({});
  const [baseAlpha, setBaseAlpha] = useState(0.5);
  const [baseAlphaC, setBaseAlphaC] = useState(0.5);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const [files, presets] = await Promise.all([
          loadModelFiles(
            [{ label: 'pretrained_model/', dir: dirs.pretrained_model }, { label: 'ComfyUI', dir: dirs.comfyui_checkpoints }],
            'safetensors,ckpt',
          ),
          utilitiesAPI.getBlockWeightPresets().catch(() => ({ sd: {}, sdxl: {} })),
        ]);
        setAvailableFiles(files);
        setBwPresets(presets.sd || {});
        setBwPresetsSdxl(presets.sdxl || {});
        if (files.length > 0) { setModelA(files[0].path); if (files[1]) setModelB(files[1].path); }
      } catch (e) {
        console.error('Failed to load checkpoint files:', e);
        setLoadError('Could not list checkpoints. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (mergeMode === 'basic' && selected.length >= 2) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`merged_checkpoint${unetOnly ? '_unet' : ''}_${date}.safetensors`);
    } else if (mergeMode === 'weighted' && modelA && modelB) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`mbw_${bwMode}_${date}.safetensors`);
    }
  }, [mergeMode, selected.length, unetOnly, modelA, modelB, bwMode]);

  // ── Basic mode handlers ────────────────────────────────────
  const add = (f: LoRAFile) => {
    if (!selected.find(c => c.path === f.path))
      setSelected(prev => [...prev, { path: f.path, name: f.name, ratio: 1.0 }]);
  };
  const remove = (path: string) => setSelected(prev => prev.filter(c => c.path !== path));
  const setRatio = (path: string, ratio: number) =>
    setSelected(prev => prev.map(c => c.path === path ? { ...c, ratio: Math.max(0, Math.min(2, ratio)) } : c));

  const handleBasicMerge = async () => {
    if (selected.length < 2) { setError('Select at least 2 checkpoints to merge'); return; }
    if (!outputPath) { setError('Provide an output filename'); return; }
    try {
      setMerging(true); setError(null); setResult(null);
      const res = await utilitiesAPI.mergeCheckpoint(
        selected.map(c => ({ path: c.path, ratio: c.ratio })),
        outputPath, unetOnly, 'cpu', savePrecision, precision
      );
      if (res.success) { setResult(res); setSelected([]); }
      else setError(res.message ?? 'Merge failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Merge failed');
    } finally { setMerging(false); }
  };

  // ── Weighted mode handlers ─────────────────────────────────
  const getResolvedBlockWeights = (presetName: string, presets: Record<string, number[]>) =>
    presetName && presets[presetName] ? presets[presetName] : null;

  const handleWeightedMerge = async () => {
    if (!modelA || !modelB) { setError('Select at least Model A and Model B'); return; }
    if ((bwMode === 'add' || bwMode === 'triple' || bwMode === 'twice') && !modelC) {
      setError(`Model C is required for ${bwMode} mode`); return;
    }
    if (!bwPreset) { setError('Select a block weight preset'); return; }
    if ((bwMode === 'triple' || bwMode === 'twice') && !bwPresetC) {
      setError(`Second block weight preset required for ${bwMode} mode`); return;
    }
    if (!outputPath) { setError('Provide an output filename'); return; }

    const weights = getResolvedBlockWeights(bwPreset, bwPresets);
    const weightsC = bwPresetC ? getResolvedBlockWeights(bwPresetC, bwPresetsSdxl) ?? getResolvedBlockWeights(bwPresetC, bwPresets) : undefined;

    try {
      setMerging(true); setError(null); setResult(null);
      const res = await utilitiesAPI.mergeCheckpointWeighted(
        modelA, modelB, outputPath, bwMode, weights!, baseAlpha,
        bwMode !== 'weight' ? modelC : undefined,
        weightsC, bwMode === 'twice' ? baseAlphaC : undefined,
      );
      if (res.success) setResult(res);
      else setError(res.message ?? 'Merge failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Merge failed');
    } finally { setMerging(false); }
  };

  const handleFileSelect = (
    path: string,
    setter: (v: string) => void,
    others: string[],
  ) => {
    if (!others.includes(path)) setter(path);
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
        <strong>Large file warning:</strong> Checkpoint merging involves 2–7 GB files. Ensure you have sufficient disk space and RAM — this may take several minutes.
      </div>

      <div className="flex gap-2">
        <Button variant={mergeMode === 'basic' ? 'default' : 'outline'} onClick={() => setMergeMode('basic')} size="sm">
          Basic
        </Button>
        <Button variant={mergeMode === 'weighted' ? 'default' : 'outline'} onClick={() => setMergeMode('weighted')} size="sm">
          Block-Weighted
        </Button>
      </div>

      {mergeMode === 'basic' ? (
        <>
          <SectionCard title="Merge Options">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <Label className="text-xs">Computation precision</Label>
                <Select value={precision} onValueChange={setPrecision}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="float">Float (recommended)</SelectItem>
                    <SelectItem value="fp16">FP16</SelectItem>
                    <SelectItem value="bf16">BF16</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Save precision</Label>
                <Select value={savePrecision} onValueChange={setSavePrecision}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fp16">FP16</SelectItem>
                    <SelectItem value="bf16">BF16</SelectItem>
                    <SelectItem value="float">Float</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-end pb-1">
                <div className="flex items-center gap-2">
                  <Checkbox id="unet-only" checked={unetOnly} onCheckedChange={v => setUnetOnly(Boolean(v))} />
                  <Label htmlFor="unet-only" className="text-xs cursor-pointer">
                    UNet only (keep VAE &amp; TE from first model)
                  </Label>
                </div>
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Available Checkpoints">
            <p className="text-xs text-muted-foreground">From your trainer pretrained_model/ and ComfyUI checkpoints folder.</p>
            {loadError && <p className="text-xs text-destructive">{loadError}</p>}
            {availableFiles.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-6">No checkpoint files found in pretrained_model/ or ComfyUI.</p>
            ) : (
              <div className="grid md:grid-cols-2 gap-2 max-h-80 overflow-y-auto">
                {availableFiles.map(f => (
                  <Button
                    key={f.path}
                    variant="outline"
                    onClick={() => add(f)}
                    disabled={selected.some(c => c.path === f.path)}
                    className="justify-between h-auto py-2 px-3"
                  >
                    <div className="text-left min-w-0">
                      <div className="truncate text-xs font-medium">{f.name}</div>
                      <div className="text-xs text-muted-foreground">{f.source} · {f.size_formatted}</div>
                    </div>
                    {selected.some(c => c.path === f.path) && (
                      <CheckCircle className="h-4 w-4 text-green-500 shrink-0 ml-2" />
                    )}
                  </Button>
                ))}
              </div>
            )}
          </SectionCard>

          {selected.length > 0 && (
            <SectionCard title={`Selected Checkpoints (${selected.length})`}>
              <div className="space-y-2">
                {selected.map(c => (
                  <div key={c.path} className="flex items-center gap-3 rounded-md border border-border px-3 py-2">
                    <span className="flex-1 truncate text-xs font-medium">{c.name}</span>
                    <Label className="text-xs text-muted-foreground shrink-0">Ratio</Label>
                    <Input
                      type="number" value={c.ratio}
                      onChange={e => setRatio(c.path, parseFloat(e.target.value) || 0)}
                      step="0.1" min={0} max={2}
                      className="w-20 h-7 text-xs text-center"
                    />
                    <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => remove(c.path)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Output filename</Label>
                <Input value={outputPath} onChange={e => setOutputPath(e.target.value)}
                  placeholder="merged_checkpoint.safetensors" className="text-xs font-mono" />
                <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
              </div>
            </SectionCard>
          )}

          <Button onClick={handleBasicMerge} disabled={merging || selected.length < 2 || !outputPath} className="w-full">
            {merging ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Merging…</> : `Merge ${selected.length} Checkpoints`}
          </Button>
        </>
      ) : (
        <>
          <SectionCard title="Merge Mode">
            <div className="grid grid-cols-4 gap-2">
              {(['weight', 'add', 'triple', 'twice'] as const).map(m => (
                <Button key={m} variant={bwMode === m ? 'default' : 'outline'} onClick={() => setBwMode(m)} size="sm">
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {bwMode === 'weight' && 'A·(1−α) + B·α — standard weighted blend with per-block control'}
              {bwMode === 'add' && 'A + (B − C)·α — additive difference merge'}
              {bwMode === 'triple' && 'A·(1−α−β) + B·α + C·β — three-model blend'}
              {bwMode === 'twice' && 'merge(A, B, α) then merge(result, C, β) — sequential blend'}
            </p>
          </SectionCard>

          <SectionCard title="Model Selection">
            <div className="space-y-3">
              <div>
                <Label className="text-xs">Model A (base)</Label>
                <Select value={modelA} onValueChange={v => handleFileSelect(v, setModelA, [modelB, modelC])}>
                  <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {availableFiles.map(f => (
                      <SelectItem key={f.path} value={f.path} className="text-xs font-mono">{f.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Model B</Label>
                <Select value={modelB} onValueChange={v => handleFileSelect(v, setModelB, [modelA, modelC])}>
                  <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {availableFiles.map(f => (
                      <SelectItem key={f.path} value={f.path} className="text-xs font-mono">{f.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {(bwMode === 'add' || bwMode === 'triple' || bwMode === 'twice') && (
                <div>
                  <Label className="text-xs">Model C</Label>
                  <Select value={modelC} onValueChange={v => handleFileSelect(v, setModelC, [modelA, modelB])}>
                    <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
                    <SelectContent>
                      {availableFiles.map(f => (
                        <SelectItem key={f.path} value={f.path} className="text-xs font-mono">{f.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          </SectionCard>

          <SectionCard title="Block Weights">
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Block weight preset (α curve)</Label>
                <Select value={bwPreset} onValueChange={setBwPreset}>
                  <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select a preset…" /></SelectTrigger>
                  <SelectContent>
                    {Object.keys(bwPresets).length === 0 && Object.keys(bwPresetsSdxl).length === 0 && (
                      <SelectItem value="__loading" disabled className="text-xs text-muted-foreground">No presets loaded</SelectItem>
                    )}
                    {Object.keys(bwPresets).length > 0 && (
                      <>
                        <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">SD1.5</div>
                        {Object.keys(bwPresets).map(name => (
                          <SelectItem key={name} value={name} className="text-xs font-mono">{name}</SelectItem>
                        ))}
                      </>
                    )}
                    {Object.keys(bwPresetsSdxl).length > 0 && (
                      <>
                        <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">SDXL</div>
                        {Object.keys(bwPresetsSdxl).map(name => (
                          <SelectItem key={name} value={name} className="text-xs font-mono">{name}</SelectItem>
                        ))}
                      </>
                    )}
                  </SelectContent>
                </Select>
              </div>

              {(bwMode === 'triple' || bwMode === 'twice') && (
                <div className="space-y-1.5">
                  <Label className="text-xs">Block weight preset (β curve) — second curve</Label>
                  <Select value={bwPresetC} onValueChange={setBwPresetC}>
                    <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select a preset…" /></SelectTrigger>
                    <SelectContent>
                      {[...Object.keys(bwPresets), ...Object.keys(bwPresetsSdxl)].map(name => (
                        <SelectItem key={name} value={name} className="text-xs font-mono">{name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="space-y-1.5">
                <Label className="text-xs">Base alpha (non-UNet: TE / VAE) — {baseAlpha.toFixed(2)}</Label>
                <Input type="range" min={0} max={1} step={0.01} value={baseAlpha}
                  onChange={e => setBaseAlpha(parseFloat(e.target.value))} />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Use A</span><span>Equal</span><span>Use B</span>
                </div>
              </div>

              {bwMode === 'twice' && (
                <div className="space-y-1.5">
                  <Label className="text-xs">Base alpha (second stage) — {baseAlphaC.toFixed(2)}</Label>
                  <Input type="range" min={0} max={1} step={0.01} value={baseAlphaC}
                    onChange={e => setBaseAlphaC(parseFloat(e.target.value))} />
                </div>
              )}
            </div>
          </SectionCard>

          <div className="space-y-1.5">
            <Label className="text-xs">Output filename</Label>
            <Input value={outputPath} onChange={e => setOutputPath(e.target.value)}
              placeholder="mbw_merged.safetensors" className="text-xs font-mono" />
            <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
          </div>

          <Button onClick={handleWeightedMerge}
            disabled={merging || !modelA || !modelB || !bwPreset || !outputPath}
            className="w-full">
            {merging ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Merging…</> : `Merge (${bwMode})`}
          </Button>
        </>
      )}

      {error && <ErrorBanner message={error} />}
      {result && (result as { success: boolean }).success && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Merge successful</div>
          <div>Output: {String((result as Record<string, unknown>).output_path)}</div>
          <div>{String((result as Record<string, unknown>).file_size_mb)} MB</div>
        </SuccessBanner>
      )}
    </div>
  );
}

// ─── Resize LoRA ──────────────────────────────────────────────────────────────

function ResizeLoRATab() {
  const [inputFile, setInputFile] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [newDim, setNewDim] = useState(32);
  const [availableFiles, setAvailableFiles] = useState<ListedFile[]>([]);
  const [availableDims, setAvailableDims] = useState<number[]>([4, 8, 16, 32, 64, 128, 256]);
  const [resizing, setResizing] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const [files, dimsRes] = await Promise.all([
          loadModelFiles(
            [{ label: 'output/', dir: dirs.output }, { label: 'ComfyUI', dir: dirs.comfyui_loras }],
            'safetensors',
          ),
          utilitiesAPI.getResizeDimensions(),
        ]);
        setAvailableFiles(files);
        if (files.length > 0) setInputFile(files[0].path);
        if (dimsRes.dimensions) setAvailableDims(dimsRes.dimensions);
      } catch (e) {
        console.error('Failed to load data:', e);
        setLoadError('Could not list LoRAs. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (inputFile) {
      const base = inputFile.replace(/\.safetensors$/, '');
      setOutputPath(`${base}_dim${newDim}.safetensors`);
    }
  }, [inputFile, newDim]);

  const handleResize = async () => {
    if (!inputFile || !outputPath) { setError('Select an input file and provide an output path'); return; }
    try {
      setResizing(true); setError(null); setResult(null);
      const res = await utilitiesAPI.resizeLora(inputFile, outputPath, newDim);
      if (res.success) setResult(res);
      else setError(res.error ?? 'Resize failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Resize failed');
    } finally { setResizing(false); }
  };

  return (
    <div className="space-y-6">
      <SectionCard title="Resize Configuration">
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Input LoRA file</Label>
            {availableFiles.length > 0 ? (
              <Select value={inputFile} onValueChange={setInputFile}>
                <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select a file…" /></SelectTrigger>
                <SelectContent>
                  {availableFiles.map(f => (
                    <SelectItem key={f.path} value={f.path} className="text-xs font-mono">
                      {f.name} ({f.source} · {f.size_formatted})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                value={inputFile}
                onChange={e => setInputFile(e.target.value)}
                placeholder="output/my_lora.safetensors"
                className="text-xs font-mono"
              />
            )}
            {loadError && <p className="text-xs text-destructive">{loadError}</p>}
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">New dimension (rank)</Label>
            <Select value={String(newDim)} onValueChange={v => setNewDim(parseInt(v))}>
              <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {availableDims.map(d => (
                  <SelectItem key={d} value={String(d)} className="text-xs">{d}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">Lower = smaller file, less detail</p>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">Output path</Label>
            <Input
              value={outputPath}
              onChange={e => setOutputPath(e.target.value)}
              placeholder="output/my_lora_dim32.safetensors"
              className="text-xs font-mono"
            />
            <p className="text-xs text-muted-foreground">Auto-generated from input file and dimension</p>
          </div>
        </div>

        <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs text-blue-700 dark:text-blue-400 space-y-1">
          <p className="font-medium">How resizing works</p>
          <ul className="list-disc list-inside space-y-0.5">
            <li>Reduces LoRA rank via SVD — smaller file, coarser detail</li>
            <li>Original file is not modified</li>
          </ul>
        </div>
      </SectionCard>

      <Button onClick={handleResize} disabled={resizing || !inputFile || !outputPath} className="w-full">
        {resizing ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Resizing…</> : <><Minimize2 className="mr-2 h-4 w-4" /> Resize LoRA</>}
      </Button>

      {error && <ErrorBanner message={error} />}
      {result && (result as { success: boolean }).success && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Resize successful</div>
          <div>Output: {String((result as Record<string, unknown>).output_path)}</div>
          <div>New dimension: {String((result as Record<string, unknown>).new_dim)} (alpha auto-calculated from SVD)</div>
        </SuccessBanner>
      )}
    </div>
  );
}

// ─── LoRA → Checkpoint (bake LoRA into base model) ─────────────────────────────

function LoRAToCheckpointTab() {
  const [baseModels, setBaseModels] = useState<ListedFile[]>([]);
  const [availableLoras, setAvailableLoras] = useState<ListedFile[]>([]);
  const [baseModel, setBaseModel] = useState('');
  const [textEncoderPath, setTextEncoderPath] = useState('');
  const [selectedLoras, setSelectedLoras] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [modelType, setModelType] = useState<'sd' | 'sdxl' | 'anima'>('sdxl');
  const [outputPath, setOutputPath] = useState('');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const [ckpts, loras] = await Promise.all([
          loadModelFiles(
            [{ label: 'pretrained_model/', dir: dirs.pretrained_model }, { label: 'ComfyUI', dir: dirs.comfyui_checkpoints }],
            'safetensors,ckpt',
          ),
          loadModelFiles(
            [{ label: 'output/', dir: dirs.output }, { label: 'ComfyUI', dir: dirs.comfyui_loras }],
            'safetensors',
          ),
        ]);
        setBaseModels(ckpts);
        if (ckpts.length > 0) setBaseModel(ckpts[0].path);
        setAvailableLoras(loras);
      } catch (e) {
        console.error('Failed to load models:', e);
        setLoadError('Could not list models. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (baseModel && selectedLoras.length > 0) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`baked_${modelType}_${date}.safetensors`);
    }
  }, [baseModel, selectedLoras.length, modelType]);

  const add = (file: ListedFile) => {
    if (!selectedLoras.find(l => l.path === file.path))
      setSelectedLoras(prev => [...prev, { path: file.path, name: file.name, ratio: 1.0 }]);
  };
  const remove = (path: string) => setSelectedLoras(prev => prev.filter(l => l.path !== path));
  const setRatio = (path: string, ratio: number) =>
    setSelectedLoras(prev => prev.map(l => l.path === path ? { ...l, ratio: Math.max(0, Math.min(2, ratio)) } : l));

  const handleMerge = async () => {
    if (!baseModel) { setError('Select a base checkpoint'); return; }
    if (selectedLoras.length < 1) { setError('Select at least 1 LoRA to bake in'); return; }
    if (modelType === 'anima' && !textEncoderPath) { setError('Provide a text encoder path for Anima'); return; }
    if (!outputPath) { setError('Provide an output filename'); return; }
    try {
      setMerging(true); setError(null); setResult(null);
      const res = await utilitiesAPI.mergeLoraToCheckpoint(
        baseModel,
        selectedLoras.map(l => ({ path: l.path, ratio: l.ratio })),
        outputPath, modelType, 'cpu', 'fp16', 'float',
        modelType === 'anima' ? textEncoderPath : undefined,
      );
      if (res.success) { setResult(res); setSelectedLoras([]); }
      else setError(res.message ?? 'Merge failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Merge failed');
    } finally { setMerging(false); }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-700 dark:text-blue-400">
        Bake one or more LoRAs into a base checkpoint to produce a standalone full model. SD1.5, SDXL, and Anima (DiT).
      </div>

      <SectionCard title="Architecture">
        <div className="grid grid-cols-3 gap-3">
          <Button variant={modelType === 'sd' ? 'default' : 'outline'} onClick={() => setModelType('sd')}>
            SD 1.5
          </Button>
          <Button variant={modelType === 'sdxl' ? 'default' : 'outline'} onClick={() => setModelType('sdxl')}>
            SDXL
          </Button>
          <Button variant={modelType === 'anima' ? 'default' : 'outline'} onClick={() => setModelType('anima')}>
            Anima
          </Button>
        </div>
      </SectionCard>

      <SectionCard title="Base Checkpoint">
        <p className="text-xs text-muted-foreground">From your trainer pretrained_model/ and ComfyUI checkpoints folder.</p>
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        {baseModels.length > 0 ? (
          <Select value={baseModel} onValueChange={setBaseModel}>
            <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select a base checkpoint…" /></SelectTrigger>
            <SelectContent>
              {baseModels.map(f => (
                <SelectItem key={f.path} value={f.path} className="text-xs font-mono">
                  {f.name} ({f.source} · {f.size_formatted})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-6">No checkpoints found in pretrained_model/ or ComfyUI.</p>
        )}
      </SectionCard>

      {modelType === 'anima' && (
        <SectionCard title="Text Encoder">
          <p className="text-xs text-muted-foreground">Path to Qwen3 text encoder (directory or .safetensors).</p>
          <Input
            value={textEncoderPath}
            onChange={e => setTextEncoderPath(e.target.value)}
            placeholder="pretrained_model/qwen3_06b/"
            className="text-xs font-mono"
          />
        </SectionCard>
      )}

      <SectionCard title="LoRAs to Bake In">
        <p className="text-xs text-muted-foreground">From your trainer output/ and ComfyUI loras folder.</p>
        {availableLoras.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No LoRA files found in output/ or ComfyUI.</p>
        ) : (
          <div className="grid md:grid-cols-2 gap-2 max-h-72 overflow-y-auto">
            {availableLoras.map(f => (
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
              placeholder="baked_model.safetensors"
              className="text-xs font-mono"
            />
            <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
          </div>
        </SectionCard>
      )}

      <Button
        onClick={handleMerge}
        disabled={merging || !baseModel || selectedLoras.length < 1 || !outputPath || (modelType === 'anima' && !textEncoderPath)}
        className="w-full"
      >
        {merging ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Baking…</> : `Bake ${selectedLoras.length} LoRA(s) into checkpoint`}
      </Button>

      {error && <ErrorBanner message={error} />}
      {result && (result as { success: boolean }).success && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Bake successful</div>
          <div>Output: {String((result as Record<string, unknown>).output_path)}</div>
          <div>Baked {String((result as Record<string, unknown>).merged_count)} LoRA(s) · {String((result as Record<string, unknown>).file_size_mb)} MB</div>
        </SuccessBanner>
      )}
    </div>
  );
}
