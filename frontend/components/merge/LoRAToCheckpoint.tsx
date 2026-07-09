'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Loader2, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  SectionCard,
  ErrorBanner,
  SuccessBanner,
  loadModelFiles,
  ListedFile,
  utilitiesAPI,
} from '@/components/merge/shared';

export function LoRAToCheckpointTab() {
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
            [{ label: 'pretrained_model/', dir: dirs.pretrained_model }, { label: 'ComfyUI checkpoints', dir: dirs.comfyui_checkpoints }, { label: 'ComfyUI diffusion_models', dir: dirs.comfyui_diffusion_models }, { label: 'ComfyUI unet', dir: dirs.comfyui_unet }],
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
