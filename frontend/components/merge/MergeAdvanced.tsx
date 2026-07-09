'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Layers, Loader2, CheckCircle, ChevronDown, ChevronUp, Info,
} from 'lucide-react';
import { utilitiesAPI, CheckpointAdvancedMergeRequest, JobAcceptedResponse } from '@/lib/api';
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
import { SectionCard, ErrorBanner, SuccessBanner, ListedFile, loadModelFiles } from '@/components/merge/shared';

// ── Mode definitions ───────────────────────────────────────

interface ModeInfo {
  id: string;
  label: string;
  description: string;
  category: string;
  needsBeta: boolean;
  needsModelC: boolean;
}

const MODE_CATEGORIES = ['Basic', 'Difference', 'Triple', 'Advanced', 'Structure', 'Utility'] as const;

const MODES: ModeInfo[] = [
  // Basic
  { id: 'WS', label: 'WS', description: 'Weighted Sum — standard linear interpolation. α=0.45 means 45% B.', category: 'Basic', needsBeta: false, needsModelC: false },
  { id: 'SIG', label: 'SIG', description: 'Sigmoid — sigmoid-weighted blend with smoother transitions near extremes.', category: 'Basic', needsBeta: false, needsModelC: false },
  { id: 'GEO', label: 'GEO', description: 'Geometric — geometric mean interpolation. Good for preserving magnitude ratios.', category: 'Basic', needsBeta: false, needsModelC: false },
  { id: 'MAX', label: 'MAX', description: 'Maximum — element-wise max. Preserves the strongest features from each model.', category: 'Basic', needsBeta: false, needsModelC: false },
  // Difference
  { id: 'AD', label: 'AD', description: 'Additive Difference — A + (B − C). Requires Model C.', category: 'Difference', needsBeta: false, needsModelC: true },
  { id: 'sAD', label: 'sAD', description: 'Scaled Additive Difference — A + α·(B − C). Requires Model C.', category: 'Difference', needsBeta: false, needsModelC: true },
  { id: 'MD', label: 'MD', description: 'Mixed Difference — blends differences across multiple tensor groups.', category: 'Difference', needsBeta: false, needsModelC: true },
  { id: 'SIM', label: 'SIM', description: 'Similarity-weighted — uses cosine similarity to weight the blend.', category: 'Difference', needsBeta: false, needsModelC: true },
  { id: 'TD', label: 'TD', description: 'Tensor Difference — per-tensor difference blending.', category: 'Difference', needsBeta: false, needsModelC: true },
  // Triple
  { id: 'TRS', label: 'TRS', description: 'Tensor Restructuring Sum — A·α + B·β + C·(1−α−β). Requires Model C, alpha + beta.', category: 'Triple', needsBeta: true, needsModelC: true },
  { id: 'TS', label: 'TS', description: 'Tensor Sum — simple additive merge of all three models.', category: 'Triple', needsBeta: false, needsModelC: true },
  { id: 'ST', label: 'ST', description: 'Sigmoid Triple — sigmoid-weighted three-model blend.', category: 'Triple', needsBeta: true, needsModelC: true },
  // Advanced
  { id: 'DARE', label: 'DARE', description: 'Drop And REscale — stochastic merge that drops random delta values and rescales.', category: 'Advanced', needsBeta: false, needsModelC: false },
  { id: 'ORTHO', label: 'ORTHO', description: 'Orthogonal — orthogonal projection-based merge for minimal interference.', category: 'Advanced', needsBeta: false, needsModelC: false },
  { id: 'SPRSE', label: 'SPRSE', description: 'Sparse Top-k — keeps only the top-k largest deltas per tensor.', category: 'Advanced', needsBeta: false, needsModelC: false },
  { id: 'NORM', label: 'NORM', description: 'Normalized — L2-normalized weight blending.', category: 'Advanced', needsBeta: false, needsModelC: false },
  // Structure
  { id: 'CHAN', label: 'CHAN', description: 'Channel-wise — merges per-channel instead of per-tensor for finer control.', category: 'Structure', needsBeta: false, needsModelC: false },
  { id: 'FREQ', label: 'FREQ', description: 'Frequency — merges in frequency domain using DCT. Preserves high/low freq separately.', category: 'Structure', needsBeta: false, needsModelC: false },
  // Utility
  { id: 'SWAP', label: 'SWAP', description: 'Swap — swaps corresponding weights between two models.', category: 'Utility', needsBeta: false, needsModelC: false },
  { id: 'CLIPXOR', label: 'CLIPXOR', description: 'CLIP XOR — exclusive-or style merge focused on CLIP text encoder weights.', category: 'Utility', needsBeta: false, needsModelC: false },
  { id: 'XDARE', label: 'XDARE', description: 'Extended DARE — DARE with additional normalization steps.', category: 'Utility', needsBeta: false, needsModelC: false },
  { id: 'FWM', label: 'FWM', description: 'Feature Weighted Merge — experimental cross-architecture merge (e.g. SD1.5 → SDXL).', category: 'Utility', needsBeta: false, needsModelC: false },
];

const getModeById = (id: string): ModeInfo | undefined => MODES.find(m => m.id === id);

// ── Component ──────────────────────────────────────────────

export function MergeAdvancedTab() {
  // Model selection
  const [availableFiles, setAvailableFiles] = useState<ListedFile[]>([]);
  const [modelA, setModelA] = useState('');
  const [modelB, setModelB] = useState('');
  const [modelC, setModelC] = useState('');

  // Mode
  const [mode, setMode] = useState('WS');
  const currentMode = getModeById(mode);

  // Ratios
  const [useBlockWeights, setUseBlockWeights] = useState(false);
  const [alpha, setAlpha] = useState('0.45');
  const [beta, setBeta] = useState('0.20');
  const [blockWeights, setBlockWeights] = useState('');

  // Options
  const [cosineModel, setCosineModel] = useState<'none' | '0' | '1' | '2'>('none');
  const [vaePath, setVaePath] = useState('');
  const [prune, setPrune] = useState(false);
  const [keepEma, setKeepEma] = useState(false);
  const [rebasin, setRebasin] = useState(0);
  const [finetuneKeys, setFinetuneKeys] = useState('');
  const [seed, setSeed] = useState('');
  const [precision, setPrecision] = useState('float');
  const [saveSafetensors, setSaveSafetensors] = useState(true);

  // Output
  const [outputPath, setOutputPath] = useState('');

  // Status
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<JobAcceptedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showModeHelp, setShowModeHelp] = useState(true);

  // Load available files
  useEffect(() => {
    const load = async () => {
      try {
        const dirs = await utilitiesAPI.getDirectories();
        const files = await loadModelFiles(
          [
            { label: 'pretrained_model/', dir: dirs.pretrained_model },
            { label: 'ComfyUI checkpoints', dir: dirs.comfyui_checkpoints },
            { label: 'ComfyUI diffusion_models', dir: dirs.comfyui_diffusion_models },
            { label: 'ComfyUI unet', dir: dirs.comfyui_unet },
          ],
          'safetensors,ckpt',
        );
        setAvailableFiles(files);
        if (files.length > 0) { setModelA(files[0].path); if (files[1]) setModelB(files[1].path); }
      } catch (e) {
        console.error('Failed to load checkpoint files:', e);
        setLoadError('Could not list checkpoints. Check that the backend is running.');
      }
    };
    void load();
  }, []);

  // Auto-generate output name
  useEffect(() => {
    if (modelA && modelB) {
      const date = new Date().toISOString().split('T')[0];
      setOutputPath(`chattiori_${mode}_${date}.safetensors`);
    }
  }, [modelA, modelB, mode]);

  // Reset Model C when switching to modes that don't need it
  useEffect(() => {
    if (currentMode && !currentMode.needsModelC && modelC) {
      setModelC('');
    }
  }, [mode, currentMode, modelC]);

  // Handle file selection without duplicates
  const handleFileSelect = useCallback(
    (path: string, setter: (v: string) => void, others: string[]) => {
      if (!others.includes(path)) setter(path);
    },
    [],
  );

  // Handle merge
  const handleMerge = async () => {
    if (!modelA || !modelB) { setError('Select at least Model A and Model B'); return; }
    if (!outputPath) { setError('Provide an output filename'); return; }

    try {
      setMerging(true); setError(null); setResult(null);

      // Derive the model directory from Model A's path — all models must be in the same directory
      const commonDir = modelA.substring(0, modelA.lastIndexOf('/'));
      const basename = (p: string) => p.substring(p.lastIndexOf('/') + 1);

      const request: CheckpointAdvancedMergeRequest = {
        mode,
        model_path: commonDir,
        model_0: basename(modelA),
        model_1: basename(modelB),
        model_2: modelC ? basename(modelC) : undefined,
        output: outputPath.replace(/\.safetensors$/, '').replace(/\.ckpt$/, ''),
        alpha: useBlockWeights ? blockWeights : alpha,
        beta: currentMode?.needsBeta ? beta : undefined,
        device: 'cpu',
        save_safetensors: saveSafetensors,
        save_half: false,
        cosine0: cosineModel === '0',
        cosine1: cosineModel === '1',
        cosine2: cosineModel === '2',
        vae: vaePath || undefined,
        prune,
        keep_ema: prune ? keepEma : false,
        rebasin: rebasin > 0 ? rebasin : undefined,
        fine: finetuneKeys || undefined,
        seed: seed ? parseInt(seed) : undefined,
        memo: `Merged via Ktiseos-Nyx-Trainer | mode: ${mode} | ${new Date().toISOString()}`,
      };

      const res = await utilitiesAPI.mergeCheckpointAdvanced(request);
      setResult(res);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Merge failed');
    } finally { setMerging(false); }
  };

  const fileOptions = availableFiles.map(f => (
    <SelectItem key={f.path} value={f.path} className="text-xs font-mono">
      {f.name}
    </SelectItem>
  ));

  return (
    <div className="space-y-6">
      {/* Large file warning */}
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
        <strong>Large file warning:</strong> Checkpoint merging involves 2–7 GB files.
        Ensure you have sufficient disk space and RAM.
        {merging && <span className="ml-2"><Loader2 className="inline h-3 w-3 animate-spin" /> Merge in progress — track status via <strong>GET /jobs/&#123;id&#125;</strong>.</span>}
      </div>

      {/* Mode selector */}
      <SectionCard title="Merge Mode">
        <div className="space-y-1.5">
          <Select value={mode} onValueChange={setMode}>
            <SelectTrigger className="text-xs font-mono">
              <SelectValue placeholder="Select mode…" />
            </SelectTrigger>
            <SelectContent>
              {MODE_CATEGORIES.map(cat => (
                <div key={cat}>
                  <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">{cat}</div>
                  {MODES.filter(m => m.category === cat).map(m => (
                    <SelectItem key={m.id} value={m.id} className="text-xs font-mono">
                      {m.label} — {m.description.split('.')[0]}.
                    </SelectItem>
                  ))}
                </div>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Mode help card */}
        {currentMode && (
          <div className="mt-2">
            <button
              onClick={() => setShowModeHelp(!showModeHelp)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Info className="h-3 w-3" />
              Mode help
              {showModeHelp ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            {showModeHelp && (
              <div className="mt-2 rounded-md border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs text-blue-700 dark:text-blue-400">
                <p className="font-medium mb-1">{currentMode.label} ({currentMode.id})</p>
                <p>{currentMode.description}</p>
                {currentMode.needsModelC && <p className="mt-1">Requires Model C.</p>}
                {currentMode.needsBeta && <p className="mt-1">Beta ratio is available for this mode.</p>}
              </div>
            )}
          </div>
        )}
      </SectionCard>

      {/* Model selection */}
      <SectionCard title="Models">
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs">Model A</Label>
            <Select value={modelA} onValueChange={v => handleFileSelect(v, setModelA, [modelB, modelC])}>
              <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
              <SelectContent>{fileOptions}</SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Model B</Label>
            <Select value={modelB} onValueChange={v => handleFileSelect(v, setModelB, [modelA, modelC])}>
              <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
              <SelectContent>{fileOptions}</SelectContent>
            </Select>
          </div>
          {(currentMode?.needsModelC || modelC) && (
            <div className="space-y-1.5">
              <Label className="text-xs">Model C</Label>
              <Select value={modelC} onValueChange={v => handleFileSelect(v, setModelC, [modelA, modelB])}>
                <SelectTrigger className="text-xs font-mono"><SelectValue placeholder="Select…" /></SelectTrigger>
                <SelectContent>{fileOptions}</SelectContent>
              </Select>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Ratios */}
      <SectionCard title="Ratios">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Label className="text-xs shrink-0">
              {useBlockWeights ? 'Block weights' : 'Alpha'}
            </Label>
            <Input
              value={useBlockWeights ? blockWeights : alpha}
              onChange={e => useBlockWeights ? setBlockWeights(e.target.value) : setAlpha(e.target.value)}
              placeholder={useBlockWeights ? '0.1,0.2,0.5,...' : '0.45'}
              className="text-xs font-mono flex-1"
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setUseBlockWeights(!useBlockWeights)}
              className="text-xs shrink-0 h-7"
            >
              {useBlockWeights ? 'Simple' : 'Block weights'}
            </Button>
          </div>
          {currentMode?.needsBeta && (
            <div className="flex items-center gap-2">
              <Label className="text-xs shrink-0">Beta</Label>
              <Input
                value={beta}
                onChange={e => setBeta(e.target.value)}
                placeholder="0.20"
                className="text-xs font-mono flex-1"
              />
            </div>
          )}
        </div>
      </SectionCard>

      {/* Options */}
      <SectionCard title="Options">
        <div className="space-y-4">
          {/* Cosine structure */}
          <div className="flex items-center gap-3">
            <Label className="text-xs shrink-0">Cosine structure</Label>
            <Select value={cosineModel} onValueChange={(v) => setCosineModel(v as 'none' | '0' | '1' | '2')}>
              <SelectTrigger className="text-xs font-mono w-32"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none" className="text-xs">Disabled</SelectItem>
                <SelectItem value="0" className="text-xs">Model 0</SelectItem>
                <SelectItem value="1" className="text-xs">Model 1</SelectItem>
                <SelectItem value="2" className="text-xs">Model 2</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* VAE */}
          <div className="flex items-center gap-3">
            <Label className="text-xs shrink-0">VAE file</Label>
            <Input
              value={vaePath}
              onChange={e => setVaePath(e.target.value)}
              placeholder="Optional — path to VAE .safetensors"
              className="text-xs font-mono flex-1"
            />
          </div>

          {/* Prune + Keep EMA */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Checkbox id="advanced-prune" checked={prune} onCheckedChange={v => setPrune(Boolean(v))} />
              <Label htmlFor="advanced-prune" className="text-xs cursor-pointer">Prune (remove EMA/unused)</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="advanced-keep-ema" checked={keepEma} disabled={!prune}
                onCheckedChange={v => setKeepEma(Boolean(v))} />
              <Label htmlFor="advanced-keep-ema" className="text-xs cursor-pointer">Keep EMA</Label>
            </div>
          </div>

          {/* ReBasin */}
          <div className="flex items-center gap-3">
            <Label className="text-xs shrink-0">ReBasin iterations</Label>
            <Input
              type="number" min={0} max={100} step={1}
              value={rebasin}
              onChange={e => setRebasin(parseInt(e.target.value) || 0)}
              className="text-xs font-mono w-20"
            />
            <span className="text-xs text-muted-foreground">0 = disabled</span>
          </div>

          {/* Finetune keys */}
          <div className="flex items-center gap-3">
            <Label className="text-xs shrink-0">Finetune keys</Label>
            <Input
              value={finetuneKeys}
              onChange={e => setFinetuneKeys(e.target.value)}
              placeholder="Optional — comma-separated"
              className="text-xs font-mono flex-1"
            />
          </div>

          {/* Seed + Precision + Format */}
          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Seed</Label>
              <Input
                type="number" min={0}
                value={seed}
                onChange={e => setSeed(e.target.value)}
                placeholder="Optional"
                className="text-xs font-mono"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Precision</Label>
              <Select value={precision} onValueChange={setPrecision}>
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="float" className="text-xs">Float (recommended)</SelectItem>
                  <SelectItem value="fp16" className="text-xs">FP16</SelectItem>
                  <SelectItem value="bf16" className="text-xs">BF16</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Format</Label>
              <Select value={saveSafetensors ? 'safetensors' : 'ckpt'} onValueChange={v => setSaveSafetensors(v === 'safetensors')}>
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="safetensors" className="text-xs">Safetensors</SelectItem>
                  <SelectItem value="ckpt" className="text-xs">CKPT</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Output + Merge button */}
      <div className="space-y-3">
        <div className="space-y-1.5">
          <Label className="text-xs">Output filename</Label>
          <Input
            value={outputPath}
            onChange={e => setOutputPath(e.target.value)}
            placeholder="chattiori_merged.safetensors"
            className="text-xs font-mono"
          />
          <p className="text-xs text-muted-foreground">Saved into your output/ directory</p>
        </div>

        <Button
          onClick={handleMerge}
          disabled={merging || !modelA || !modelB || !outputPath}
          className="w-full"
        >
          {merging ? (
            <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Launching merge job…</>
          ) : (
            <><Layers className="mr-2 h-4 w-4" /> Start Merge ({mode})</>
          )}
        </Button>
      </div>

      {error && <ErrorBanner message={error} />}
      {result && (
        <SuccessBanner>
          <div className="flex items-center gap-2 font-semibold"><CheckCircle className="h-4 w-4" /> Merge job accepted</div>
          <div>Job ID: <code className="text-xs">{result.job_id}</code></div>
          <div>Track progress via <strong>GET /jobs/{result.job_id}/logs</strong> or the job monitor.</div>
        </SuccessBanner>
      )}
    </div>
  );
}
