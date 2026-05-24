'use client';

/**
 * GenerateUI — ComfyUI connected-state interface.
 *
 * Two-panel layout via Splitter:
 *   Left  — prompt editor, checkpoint, sampler settings, LoRA stack, generate button
 *   Right — image gallery, queue progress
 *
 * Receives connection state as props so the parent page owns the single
 * useComfyConnection() call and routes between disconnected/connected views.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Plus, Trash2, RefreshCw, Square, Shuffle } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Splitter, SplitterPanel } from '@/components/ui/splitter';
import ClickSpark from '@/components/ClickSpark';

import { comfyClient, type UseComfyConnectionReturn, type ComfyOutputFile, type LoraEntry } from '@/lib/comfy';
import { buildTxt2ImgWorkflow } from '@/lib/comfy/workflows';

// ─── Constants ────────────────────────────────────────────────────────────────

const SAMPLERS = [
  'euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2', 'dpm_2_ancestral',
  'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral', 'dpmpp_sde',
  'dpmpp_sde_gpu', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
  'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ipndm',
];

const SCHEDULERS = [
  'normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform',
];

const COMMON_SIZES = [
  { label: '512×512', w: 512, h: 512 },
  { label: '768×768', w: 768, h: 768 },
  { label: '1024×1024', w: 1024, h: 1024 },
  { label: '1024×576 (16:9)', w: 1024, h: 576 },
  { label: '832×1216 (2:3)', w: 832, h: 1216 },
  { label: '1216×832 (3:2)', w: 1216, h: 832 },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/70 mb-2">
      {children}
    </p>
  );
}

function LoraRow({
  lora,
  index,
  onChange,
  onRemove,
}: {
  lora: LoraEntry;
  index: number;
  onChange: (i: number, l: LoraEntry) => void;
  onRemove: (i: number) => void;
}) {
  return (
    <div className="flex gap-1.5 items-center">
      <Input
        value={lora.name}
        onChange={e => onChange(index, { ...lora, name: e.target.value })}
        placeholder="lora_name.safetensors"
        className="flex-1 h-8 text-xs"
      />
      <Input
        type="number"
        value={lora.modelWeight ?? 1}
        onChange={e => onChange(index, { ...lora, modelWeight: parseFloat(e.target.value) || 1 })}
        step="0.05"
        min={-2}
        max={2}
        className="w-16 h-8 text-xs text-center"
        title="Model weight"
      />
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0"
        onClick={() => onRemove(index)}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ─── Image gallery ────────────────────────────────────────────────────────────

function ImageGallery({ images }: { images: ComfyOutputFile[] }) {
  const [selected, setSelected] = useState<ComfyOutputFile | null>(null);

  if (images.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground/50 p-8 text-center">
        <div className="h-16 w-16 rounded-2xl border-2 border-dashed border-muted-foreground/20 flex items-center justify-center">
          <span className="text-2xl">🎨</span>
        </div>
        <p className="text-sm">Generated images will appear here</p>
      </div>
    );
  }

  return (
    <div className="relative h-full">
      {/* Lightbox */}
      {selected && (
        <div
          className="absolute inset-0 z-10 flex items-center justify-center bg-background/90 backdrop-blur-sm"
          onClick={() => setSelected(null)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={comfyClient.getImageUrl(selected)}
            alt="Generated image"
            className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}

      {/* Grid */}
      <div className="grid h-full auto-rows-max grid-cols-2 gap-2 overflow-auto p-3 lg:grid-cols-3">
        {images.map((img, i) => (
          <button
            key={`${img.filename}-${i}`}
            onClick={() => setSelected(img)}
            className="aspect-square overflow-hidden rounded-lg border border-border/50 bg-muted hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary transition-all"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={comfyClient.getImageUrl(img)}
              alt={img.filename}
              className="h-full w-full object-cover"
              loading="lazy"
            />
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Progress bar ─────────────────────────────────────────────────────────────

function GenerationProgress({
  progress,
  currentNode,
  isGenerating,
}: {
  progress: { value: number; max: number } | null;
  currentNode: string | null;
  isGenerating: boolean;
}) {
  if (!isGenerating) return null;

  const pct = progress ? Math.round((progress.value / progress.max) * 100) : 0;

  return (
    <div className="flex shrink-0 flex-col gap-1.5 border-t border-border/40 px-3 py-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" />
          {currentNode ? `Node: ${currentNode}` : 'Queued…'}
        </span>
        {progress && <span>{pct}%</span>}
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all duration-100"
          style={{ width: progress ? `${pct}%` : '0%' }}
        />
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

type GenerateUIProps = Pick<
  UseComfyConnectionReturn,
  'submitPrompt' | 'interrupt' | 'currentPromptId' | 'currentNode' | 'progress' | 'queueRemaining'
>;

export function GenerateUI({
  submitPrompt,
  interrupt,
  currentPromptId,
  currentNode,
  progress,
  queueRemaining,
}: GenerateUIProps) {
  // ── Prompt state
  const [positivePrompt, setPositivePrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');

  // ── Model state
  const [checkpoint, setCheckpoint] = useState('');

  // ── Sampler state
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7);
  const [sampler, setSampler] = useState('euler_ancestral');
  const [scheduler, setScheduler] = useState('karras');
  const [seed, setSeed] = useState(-1);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [batchSize, setBatchSize] = useState(1);

  // ── Detail / refinement
  const [denoise, setDenoise] = useState(1.0);
  const [vae, setVae] = useState('');
  const [clipSkip, setClipSkip] = useState(-1);

  // ── LoRA stack
  const [loras, setLoras] = useState<LoraEntry[]>([]);

  // ── Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<ComfyOutputFile[]>([]);
  const lastPromptIdRef = useRef<string | null>(null);

  // Detect generation completion and fetch result images
  useEffect(() => {
    if (currentPromptId) {
      lastPromptIdRef.current = currentPromptId;
      setIsGenerating(true);
    } else if (lastPromptIdRef.current && isGenerating) {
      // Prompt finished — fetch outputs
      const pid = lastPromptIdRef.current;
      comfyClient.getHistory(pid).then(entry => {
        if (!entry) return;
        const newImages: ComfyOutputFile[] = [];
        for (const nodeOutputs of Object.values(entry.outputs)) {
          for (const files of Object.values(nodeOutputs)) {
            newImages.push(...files);
          }
        }
        setGeneratedImages(prev => [...newImages, ...prev]);
      }).catch(console.error);
      setIsGenerating(false);
    }
  }, [currentPromptId, isGenerating]);

  // ── LoRA helpers
  const addLora = useCallback(() => {
    setLoras(prev => [...prev, { name: '', modelWeight: 1, clipWeight: 1 }]);
  }, []);

  const updateLora = useCallback((i: number, lora: LoraEntry) => {
    setLoras(prev => prev.map((l, idx) => idx === i ? lora : l));
  }, []);

  const removeLora = useCallback((i: number) => {
    setLoras(prev => prev.filter((_, idx) => idx !== i));
  }, []);

  // ── Generate
  const handleGenerate = useCallback(async () => {
    if (isGenerating || !checkpoint.trim()) return;
    const workflow = buildTxt2ImgWorkflow({
      checkpoint: checkpoint.trim(),
      positivePrompt,
      negativePrompt,
      steps,
      cfg,
      sampler,
      scheduler,
      seed,
      width,
      height,
      batchSize,
      denoise,
      vae: vae.trim() || undefined,
      clipSkip,
      loras: loras.filter(l => l.name.trim()),
    });
    try {
      await submitPrompt({ prompt: workflow });
    } catch (err) {
      console.error('Generate failed:', err);
    }
  }, [
    isGenerating, checkpoint, positivePrompt, negativePrompt,
    steps, cfg, sampler, scheduler, seed, width, height, batchSize,
    loras, submitPrompt,
  ]);

  const randomSeed = useCallback(() => {
    setSeed(Math.floor(Math.random() * 2 ** 32));
  }, []);

  // ── Render
  return (
    <Splitter defaultSize={32} minSize={20} maxSize={60} className="flex-1 min-h-0">

      {/* ── Left panel: controls ── */}
      <SplitterPanel>
        <div className="flex h-full flex-col">
          <ScrollArea className="flex-1 min-h-0">
            <div className="space-y-5 p-4">

              {/* Prompts */}
              <div className="space-y-3">
                <SectionLabel>Prompts</SectionLabel>
                <div className="space-y-1.5">
                  <Label className="text-xs">Positive</Label>
                  <Textarea
                    value={positivePrompt}
                    onChange={e => setPositivePrompt(e.target.value)}
                    placeholder="Describe what you want to generate…"
                    className="min-h-20 resize-none text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">Negative</Label>
                  <Textarea
                    value={negativePrompt}
                    onChange={e => setNegativePrompt(e.target.value)}
                    placeholder="What to avoid…"
                    className="min-h-14 resize-none text-sm"
                  />
                </div>
              </div>

              <Separator />

              {/* Checkpoint */}
              <div className="space-y-1.5">
                <SectionLabel>Model</SectionLabel>
                <Input
                  value={checkpoint}
                  onChange={e => setCheckpoint(e.target.value)}
                  placeholder="model.safetensors"
                  className="text-sm font-mono"
                />
                <p className="text-xs text-muted-foreground">
                  Filename as it appears in ComfyUI&apos;s models folder
                </p>
              </div>

              <Separator />

              {/* Image size */}
              <div className="space-y-2">
                <SectionLabel>Size</SectionLabel>
                <div className="grid grid-cols-3 gap-1">
                  {COMMON_SIZES.map(s => (
                    <Button
                      key={s.label}
                      variant={width === s.w && height === s.h ? 'secondary' : 'outline'}
                      size="sm"
                      className="h-7 text-xs px-2"
                      onClick={() => { setWidth(s.w); setHeight(s.h); }}
                    >
                      {s.label}
                    </Button>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">W</Label>
                    <Input
                      type="number"
                      value={width}
                      onChange={e => setWidth(Number(e.target.value))}
                      step={64}
                      min={64}
                      max={4096}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">H</Label>
                    <Input
                      type="number"
                      value={height}
                      onChange={e => setHeight(Number(e.target.value))}
                      step={64}
                      min={64}
                      max={4096}
                      className="h-8 text-xs"
                    />
                  </div>
                </div>
              </div>

              <Separator />

              {/* Sampler settings */}
              <div className="space-y-3">
                <SectionLabel>Sampler</SectionLabel>

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Sampler</Label>
                    <Select value={sampler} onValueChange={setSampler}>
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SAMPLERS.map(s => (
                          <SelectItem key={s} value={s} className="text-xs">{s}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Scheduler</Label>
                    <Select value={scheduler} onValueChange={setScheduler}>
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SCHEDULERS.map(s => (
                          <SelectItem key={s} value={s} className="text-xs">{s}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <Label className="text-xs">Steps</Label>
                    <span className="text-xs text-muted-foreground">{steps}</span>
                  </div>
                  <Slider value={[steps]} min={1} max={150} step={1} onValueChange={([v]) => setSteps(v)} />
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <Label className="text-xs">CFG scale</Label>
                    <span className="text-xs text-muted-foreground">{cfg.toFixed(1)}</span>
                  </div>
                  <Slider value={[cfg]} min={1} max={20} step={0.5} onValueChange={([v]) => setCfg(v)} />
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Batch size</Label>
                  <Input
                    type="number"
                    value={batchSize}
                    onChange={e => setBatchSize(Math.max(1, Number(e.target.value)))}
                    min={1}
                    max={8}
                    className="h-8 text-xs"
                  />
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Seed</Label>
                  <div className="flex gap-1.5">
                    <Input
                      type="number"
                      value={seed}
                      onChange={e => setSeed(Number(e.target.value))}
                      className="h-8 text-xs flex-1 font-mono"
                      placeholder="-1 = random"
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8 shrink-0"
                      onClick={randomSeed}
                      title="Random seed"
                    >
                      <Shuffle className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8 shrink-0"
                      onClick={() => setSeed(-1)}
                      title="Reset to -1"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">-1 lets ComfyUI randomise each run</p>
                </div>
              </div>

              <Separator />

              {/* LoRA stack */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <SectionLabel>LoRAs</SectionLabel>
                  <Button variant="ghost" size="sm" className="h-6 gap-1 text-xs" onClick={addLora}>
                    <Plus className="h-3 w-3" />
                    Add
                  </Button>
                </div>
                {loras.length === 0 && (
                  <p className="text-xs text-muted-foreground/60">No LoRAs added</p>
                )}
                <div className="space-y-1.5">
                  {loras.map((lora, i) => (
                    <LoraRow key={i} lora={lora} index={i} onChange={updateLora} onRemove={removeLora} />
                  ))}
                </div>
              </div>

              <Separator />

              {/* Detail / refinement */}
              <div className="space-y-3">
                <SectionLabel>Detail &amp; Refinement</SectionLabel>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <Label className="text-xs">Denoise</Label>
                    <span className="text-xs text-muted-foreground">{denoise.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[denoise]}
                    min={0}
                    max={1}
                    step={0.01}
                    onValueChange={([v]) => setDenoise(v)}
                  />
                  <p className="text-xs text-muted-foreground">
                    1.0 = full txt2img. Lower = blend with existing latent (HiRes fix / detailer pass).
                  </p>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">External VAE <span className="text-muted-foreground/60">(optional)</span></Label>
                  <Input
                    value={vae}
                    onChange={e => setVae(e.target.value)}
                    placeholder="vae.safetensors"
                    className="h-8 text-xs font-mono"
                  />
                  <p className="text-xs text-muted-foreground">
                    Leave blank to use the checkpoint&apos;s built-in VAE
                  </p>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">CLIP skip</Label>
                  <Input
                    type="number"
                    value={clipSkip}
                    onChange={e => setClipSkip(Number(e.target.value))}
                    min={-24}
                    max={-1}
                    step={1}
                    className="h-8 text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    -1 = last layer (SD default). -2 = penultimate (common for anime/NAI).
                  </p>
                </div>
              </div>

            </div>
          </ScrollArea>

          {/* Generate / Stop button */}
          <div className="shrink-0 border-t border-border/40 p-3">
            <ClickSpark sparkColor="hsl(var(--primary))" sparkCount={10} sparkRadius={25}>
              {isGenerating ? (
                <Button
                  variant="destructive"
                  className="w-full gap-2"
                  onClick={interrupt}
                >
                  <Square className="h-4 w-4" />
                  Stop generation
                </Button>
              ) : (
                <Button
                  className="w-full gap-2"
                  onClick={handleGenerate}
                  disabled={!checkpoint.trim()}
                >
                  Generate
                  {queueRemaining > 0 && (
                    <span className="ml-1 text-xs opacity-70">({queueRemaining} queued)</span>
                  )}
                </Button>
              )}
            </ClickSpark>
            {!checkpoint.trim() && (
              <p className="mt-1.5 text-center text-xs text-muted-foreground/60">
                Enter a checkpoint name to generate
              </p>
            )}
          </div>
        </div>
      </SplitterPanel>

      {/* ── Right panel: gallery ── */}
      <SplitterPanel>
        <div className="flex h-full flex-col">
          {/* Gallery header */}
          <div className="flex shrink-0 items-center justify-between border-b border-border/40 px-3 py-2">
            <span className="text-xs font-medium text-muted-foreground">
              {generatedImages.length > 0
                ? `${generatedImages.length} image${generatedImages.length !== 1 ? 's' : ''}`
                : 'Gallery'}
            </span>
            {generatedImages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs text-muted-foreground"
                onClick={() => setGeneratedImages([])}
              >
                Clear
              </Button>
            )}
          </div>

          {/* Images */}
          <div className="min-h-0 flex-1">
            <ImageGallery images={generatedImages} />
          </div>

          {/* Progress bar */}
          <GenerationProgress
            progress={progress}
            currentNode={currentNode}
            isGenerating={isGenerating}
          />
        </div>
      </SplitterPanel>

    </Splitter>
  );
}
