'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Loader2, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
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
  LoRAFile,
  utilitiesAPI,
} from '@/components/merge/shared';

export function MergeCheckpointsTab() {
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
  const [bwPresetsAnima, setBwPresetsAnima] = useState<Record<string, number[]>>({});
  const [detectedArch, setDetectedArch] = useState<string | null>(null);
  const [detectedBlockNames, setDetectedBlockNames] = useState<string[]>([]);
  const [baseAlpha, setBaseAlpha] = useState(0.5);
  const [baseAlphaC, setBaseAlphaC] = useState(0.5);

  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const [files, presets] = await Promise.all([
          loadModelFiles(
            [{ label: 'pretrained_model/', dir: dirs.pretrained_model }, { label: 'ComfyUI checkpoints', dir: dirs.comfyui_checkpoints }, { label: 'ComfyUI diffusion_models', dir: dirs.comfyui_diffusion_models }, { label: 'ComfyUI unet', dir: dirs.comfyui_unet }],
            'safetensors,ckpt',
          ),
          utilitiesAPI.getBlockWeightPresets().catch(() => ({ sd: {}, sdxl: {}, anima: {} })),
        ]);
        setAvailableFiles(files);
        setBwPresets(presets.sd || {});
        setBwPresetsSdxl(presets.sdxl || {});
        setBwPresetsAnima(presets.anima || {});
        if (files.length > 0) { setModelA(files[0].path); if (files[1]) setModelB(files[1].path); }
      } catch (e) {
        console.error('Failed to load checkpoint files:', e);
        setLoadError('Could not list checkpoints. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (!modelA) return;
    utilitiesAPI.detectCheckpointArch(modelA)
      .then(info => {
        setDetectedArch(info.arch);
        setDetectedBlockNames(info.block_names);
      })
      .catch(() => setDetectedArch(null));
  }, [modelA]);

  useEffect(() => {
    if (mergeMode === 'basic' && selected.length >= 2) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`merged_checkpoint${unetOnly ? '_unet' : ''}_${date}.safetensors`);
    } else if (mergeMode === 'weighted' && modelA && modelB) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`mbw_${detectedArch || bwMode}_${date}.safetensors`);
    }
  }, [mergeMode, selected.length, unetOnly, modelA, modelB, bwMode, detectedArch]);

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

    const presetsByArch =
      detectedArch === 'anima' ? bwPresetsAnima :
      detectedArch === 'sdxl' ? bwPresetsSdxl :
      bwPresets;
    const weights = getResolvedBlockWeights(bwPreset, presetsByArch);
    const weightsC = bwPresetC ? getResolvedBlockWeights(bwPresetC, presetsByArch) : undefined;

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
            {detectedArch && (
              <div className="mb-2 inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium">
                <span className="text-muted-foreground">Detected:</span>
                <span>{detectedArch === 'anima' ? 'Anima DiT' : detectedArch === 'sdxl' ? 'SDXL' : 'SD1.5'}</span>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground">{detectedBlockNames.length} blocks</span>
              </div>
            )}
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
                    {(() => {
                      const presetsByArch =
                        detectedArch === 'anima' ? bwPresetsAnima :
                        detectedArch === 'sdxl' ? bwPresetsSdxl :
                        bwPresets;
                      const allEmpty = Object.keys(bwPresets).length === 0
                        && Object.keys(bwPresetsSdxl).length === 0
                        && Object.keys(bwPresetsAnima).length === 0;
                      if (allEmpty) {
                        return <SelectItem value="__loading" disabled className="text-xs text-muted-foreground">No presets loaded</SelectItem>;
                      }
                      const label = detectedArch === 'anima' ? 'Anima' : detectedArch === 'sdxl' ? 'SDXL' : 'SD1.5';
                      return Object.keys(presetsByArch).length > 0 ? (
                        <>
                          <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">{label}</div>
                          {Object.keys(presetsByArch).map(name => (
                            <SelectItem key={name} value={name} className="text-xs font-mono">{name}</SelectItem>
                          ))}
                        </>
                      ) : (
                        <SelectItem value="__no_presets" disabled className="text-xs text-muted-foreground">
                          No {label} presets — use All-Equal or enter custom weights
                        </SelectItem>
                      );
                    })()}
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
                <Label className="text-xs">Base alpha ({detectedArch === 'anima' ? 'non-DiT: TE / VAE' : 'non-UNet: TE / VAE'}) — {baseAlpha.toFixed(2)}</Label>
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
