'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, LoRAFile } from '@/lib/api';
import { Wrench, CheckCircle, Loader2, Minimize2, Home, Trash2, FolderOpen } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import FileBrowser from '@/components/FileBrowser';
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
            Merge, optimize, and publish your trained models
          </p>
        </div>

        <Tabs defaultValue="merge-lora">
          <TabsList className="mb-6">
            <TabsTrigger value="merge-lora">Merge LoRAs</TabsTrigger>
            <TabsTrigger value="merge-checkpoint">Merge Checkpoints</TabsTrigger>
            <TabsTrigger value="resize">Resize LoRA</TabsTrigger>
          </TabsList>
          <TabsContent value="merge-lora"><MergeLoRATab /></TabsContent>
          <TabsContent value="merge-checkpoint"><MergeCheckpointTab /></TabsContent>
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

// ─── Merge LoRAs ──────────────────────────────────────────────────────────────

function MergeLoRATab() {
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [modelType, setModelType] = useState<'sd' | 'sdxl' | 'flux' | 'svd'>('sdxl');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const res = await utilitiesAPI.listLoraFiles(dirs.output, 'safetensors', 'date');
        if (res.success) setAvailableFiles(res.files);
      } catch (e) {
        console.error('Failed to load LoRA files:', e);
        setLoadError('Could not auto-list LoRAs from output/. Use Browse to pick files from anywhere.');
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
  const addByPath = (path: string) => {
    const name = path.split(/[\\/]/).pop() || path;
    if (!selectedLoras.find(l => l.path === path))
      setSelectedLoras(prev => [...prev, { path, name, ratio: 1.0 }]);
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
        outputPath, modelType
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
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">Auto-listed from output/ — or browse to any folder.</p>
          <Button variant="outline" size="sm" onClick={() => setBrowserOpen(true)} className="shrink-0">
            <FolderOpen className="h-4 w-4 mr-1" /> Browse…
          </Button>
        </div>
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        {availableFiles.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No LoRA files in output/ — use Browse to add files.</p>
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
                  <div className="text-xs text-muted-foreground">{f.size_formatted}</div>
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

      <FileBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={addByPath}
        mode="file"
        title="Select LoRA file"
      />
    </div>
  );
}

// ─── Merge Checkpoints ────────────────────────────────────────────────────────

function MergeCheckpointTab() {
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [selected, setSelected] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [unetOnly, setUnetOnly] = useState(false);
  const [precision, setPrecision] = useState('float');
  const [savePrecision, setSavePrecision] = useState('fp16');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const res = await utilitiesAPI.listLoraFiles(dirs.pretrained_model, 'safetensors,ckpt', 'date');
        if (res.success) setAvailableFiles(res.files);
      } catch (e) {
        console.error('Failed to load checkpoint files:', e);
        setLoadError('Could not auto-list checkpoints from pretrained_model/. Use Browse to pick files from anywhere.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (selected.length >= 2) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`merged_checkpoint${unetOnly ? '_unet' : ''}_${date}.safetensors`);
    }
  }, [selected.length, unetOnly]);

  const add = (f: LoRAFile) => {
    if (!selected.find(c => c.path === f.path))
      setSelected(prev => [...prev, { path: f.path, name: f.name, ratio: 1.0 }]);
  };
  const addByPath = (path: string) => {
    const name = path.split(/[\\/]/).pop() || path;
    if (!selected.find(c => c.path === path))
      setSelected(prev => [...prev, { path, name, ratio: 1.0 }]);
  };
  const remove = (path: string) => setSelected(prev => prev.filter(c => c.path !== path));
  const setRatio = (path: string, ratio: number) =>
    setSelected(prev => prev.map(c => c.path === path ? { ...c, ratio: Math.max(0, Math.min(2, ratio)) } : c));

  const handleMerge = async () => {
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

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
        <strong>Large file warning:</strong> Checkpoint merging involves 2–7 GB files. Ensure you have sufficient disk space and RAM — this may take several minutes.
      </div>

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
              <Checkbox
                id="unet-only"
                checked={unetOnly}
                onCheckedChange={v => setUnetOnly(Boolean(v))}
              />
              <Label htmlFor="unet-only" className="text-xs cursor-pointer">
                UNet only (keep VAE &amp; TE from first model)
              </Label>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Available Checkpoints">
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">Auto-listed from pretrained_model/ — or browse to any folder.</p>
          <Button variant="outline" size="sm" onClick={() => setBrowserOpen(true)} className="shrink-0">
            <FolderOpen className="h-4 w-4 mr-1" /> Browse…
          </Button>
        </div>
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        {availableFiles.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No checkpoint files in pretrained_model/ — use Browse to add files.</p>
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
                  <div className="text-xs text-muted-foreground">{f.size_formatted}</div>
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
                  type="number"
                  value={c.ratio}
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
            <Input
              value={outputPath}
              onChange={e => setOutputPath(e.target.value)}
              placeholder="merged_checkpoint.safetensors"
              className="text-xs font-mono"
            />
            <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
          </div>
        </SectionCard>
      )}

      <Button
        onClick={handleMerge}
        disabled={merging || selected.length < 2 || !outputPath}
        className="w-full"
      >
        {merging ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Merging…</> : `Merge ${selected.length} Checkpoints`}
      </Button>

      {error && <ErrorBanner message={error} />}
      {result && (result as { success: boolean }).success && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Merge successful</div>
          <div>Output: {String((result as Record<string, unknown>).output_path)}</div>
          <div>Merged {String((result as Record<string, unknown>).merged_count)} checkpoints · {String((result as Record<string, unknown>).file_size_mb)} MB</div>
          {unetOnly && <div className="text-amber-600 dark:text-amber-400">UNet-only merge: VAE and text encoder from first model</div>}
        </SuccessBanner>
      )}

      <FileBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={addByPath}
        mode="file"
        title="Select checkpoint file"
      />
    </div>
  );
}

// ─── Resize LoRA ──────────────────────────────────────────────────────────────

function ResizeLoRATab() {
  const [inputFile, setInputFile] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [newDim, setNewDim] = useState(32);
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [availableDims, setAvailableDims] = useState<number[]>([4, 8, 16, 32, 64, 128, 256]);
  const [resizing, setResizing] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const [filesRes, dimsRes] = await Promise.all([
          utilitiesAPI.listLoraFiles(dirs.output, 'safetensors', 'date'),
          utilitiesAPI.getResizeDimensions(),
        ]);
        if (filesRes.success) {
          setAvailableFiles(filesRes.files);
          if (filesRes.files.length > 0) setInputFile(filesRes.files[0].path);
        }
        if (dimsRes.dimensions) setAvailableDims(dimsRes.dimensions);
      } catch (e) {
        console.error('Failed to load data:', e);
        setLoadError('Could not auto-list LoRAs from output/. Use Browse to pick a file from anywhere.');
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
            <div className="flex items-center justify-between gap-3">
              <Label className="text-xs">Input LoRA file</Label>
              <Button variant="outline" size="sm" onClick={() => setBrowserOpen(true)} className="shrink-0 h-7">
                <FolderOpen className="h-4 w-4 mr-1" /> Browse…
              </Button>
            </div>
            {availableFiles.length > 0 ? (
              <Select value={inputFile} onValueChange={setInputFile}>
                <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select a file…" /></SelectTrigger>
                <SelectContent>
                  {availableFiles.map(f => (
                    <SelectItem key={f.path} value={f.path} className="text-xs font-mono">
                      {f.name} ({f.size_formatted})
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
            <li>Uses Derrian&apos;s enhanced resize script or Kohya&apos;s standard script</li>
            <li>Reduces file size while preserving quality</li>
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
          <div>Dimension: {String((result as Record<string, unknown>).new_dim)} / Alpha: {String((result as Record<string, unknown>).new_alpha)}</div>
        </SuccessBanner>
      )}

      <FileBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={setInputFile}
        mode="file"
        title="Select LoRA file to resize"
      />
    </div>
  );
}
