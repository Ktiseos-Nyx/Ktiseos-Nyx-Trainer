'use client';

/**
 * GenerateUI — ComfyUI connected-state interface.
 *
 * Two-panel layout via Splitter:
 *   Left  — architecture switcher, prompt editor, model fields,
 *            sampler settings, LoRA stack, generate button
 *   Right — image gallery, queue progress
 *
 * Architecture modes:
 *   anima    — Guy90s ANIMA template (AuraFlow, Qwen encoder, full custom node stack)
 *   sdxl-knx — KNX SDXL fork template (Checkpoint Loader LoraManager + illustrious VAE)
 *
 * Both modes use the real workflow JSONs via templateInjector.
 * If required custom nodes are missing, the programmatic builders serve as fallback.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Plus, Trash2, RefreshCw, Square, Shuffle, ExternalLink } from 'lucide-react';

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
import {
  Combobox,
  ComboboxAnchor,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxTrigger,
} from '@/components/ui/combobox';
import { Splitter, SplitterPanel } from '@/components/ui/splitter';
import ClickSpark from '@/components/ClickSpark';

import {
  comfyClient,
  injectTemplate,
  buildAnimaPatch,
  buildSdxlKnxPatch,
  useComfyModels,
  type UseComfyConnectionReturn,
  type ComfyOutputFile,
  type LoraEntry,
} from '@/lib/comfy';

// Workflow JSONs are imported statically so the injector can deep-clone and patch them.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const animaWorkflow = require('@/lib/comfy/templates/workflows/anima-guy90s-v10.json');
// eslint-disable-next-line @typescript-eslint/no-require-imports
const sdxlWorkflow = require('@/lib/comfy/templates/workflows/sdxl-knx-v1.json');

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

/** Recommended sizes, including ANIMA's native aspect ratios. */
const COMMON_SIZES = [
  { label: '832×1216', w: 832, h: 1216 },
  { label: '1216×832', w: 1216, h: 832 },
  { label: '896×1152', w: 896, h: 1152 },
  { label: '1024×1024', w: 1024, h: 1024 },
  { label: '1152×896', w: 1152, h: 896 },
  { label: '768×768', w: 768, h: 768 },
];

/** Sampler / scheduler defaults per architecture. */
const TEMPLATE_DEFAULTS = {
  anima: {
    steps: 30, cfg: 4, sampler: 'dpmpp_2m_sde_gpu', scheduler: 'sgm_uniform',
    width: 832, height: 1216,
  },
  'sdxl-knx': {
    steps: 30, cfg: 4, sampler: 'euler_ancestral', scheduler: 'karras',
    width: 832, height: 1216,
  },
} as const;

type TemplateMode = 'anima' | 'sdxl-knx';

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/70 mb-2">
      {children}
    </p>
  );
}

function FieldHint({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-muted-foreground">{children}</p>;
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
        title="Strength"
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

// ─── Model picker ─────────────────────────────────────────────────────────────

/**
 * Combobox for selecting a ComfyUI model file.
 *
 * Shows items fetched from ComfyUI's /models/{folder} API.
 * Still accepts free-text input when models list is empty or when the user
 * needs a filename not in the list (e.g. subfolder paths).
 */
function ModelPicker({
  value,
  onChange,
  models,
  loading = false,
  placeholder,
  onRefresh,
}: {
  value: string;
  onChange: (v: string) => void;
  models: string[];
  loading?: boolean;
  placeholder?: string;
  onRefresh?: () => void;
}) {
  return (
    <Combobox
      inputValue={value}
      onInputValueChange={onChange}
      onValueChange={(v) => onChange(v as string)}
    >
      <ComboboxAnchor className="h-8">
        <ComboboxInput
          placeholder={placeholder}
          className="font-mono text-xs"
        />
        <div className="flex shrink-0 items-center gap-0.5">
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
          ) : (
            <>
              {onRefresh && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={(e) => { e.stopPropagation(); onRefresh(); }}
                  className="h-6 w-6"
                  title="Refresh model list from ComfyUI"
                >
                  <RefreshCw className="h-3 w-3" />
                </Button>
              )}
              <ComboboxTrigger />
            </>
          )}
        </div>
      </ComboboxAnchor>
      <ComboboxContent className="max-h-64 overflow-y-auto">
        <ComboboxEmpty className="py-3 text-xs">
          {loading ? 'Loading…' : 'No models found — is ComfyUI running?'}
        </ComboboxEmpty>
        {models.map(m => (
          <ComboboxItem key={m} value={m} className="font-mono text-xs">
            {m}
          </ComboboxItem>
        ))}
      </ComboboxContent>
    </Combobox>
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
      <div className="grid h-full auto-rows-max grid-cols-2 gap-2 overflow-auto p-3 lg:grid-cols-3">
        {images.map((img, i) => (
          <Button
            key={`${img.filename}-${i}`}
            variant="outline"
            onClick={() => setSelected(img)}
            className="aspect-square h-auto overflow-hidden rounded-lg border-border/50 bg-muted p-0 hover:border-primary/50"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={comfyClient.getImageUrl(img)}
              alt={img.filename}
              className="h-full w-full object-cover"
              loading="lazy"
            />
          </Button>
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

// ─── Architecture switcher ────────────────────────────────────────────────────

function ArchSwitcher({
  mode,
  onChange,
}: {
  mode: TemplateMode;
  onChange: (m: TemplateMode) => void;
}) {
  return (
    <div className="flex gap-1 p-1 rounded-lg bg-muted/50 border border-border/40">
      {(['anima', 'sdxl-knx'] as TemplateMode[]).map(m => (
        <Button
          key={m}
          variant="ghost"
          size="sm"
          onClick={() => onChange(m)}
          className={[
            'flex-1 h-7 px-3 text-xs font-medium transition-all',
            mode === m
              ? 'bg-background shadow-sm text-foreground hover:bg-background'
              : 'text-muted-foreground hover:text-foreground',
          ].join(' ')}
        >
          {m === 'anima' ? 'ANIMA' : 'SDXL KNX'}
        </Button>
      ))}
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

  // ── Model lists from ComfyUI
  const models = useComfyModels();

  // ── Architecture
  const [templateMode, setTemplateMode] = useState<TemplateMode>('anima');

  // ── Prompt state
  const [positivePrompt, setPositivePrompt] = useState('masterpiece, best quality, highres, absurdres');
  const [negativePrompt, setNegativePrompt] = useState(
    'monochrome, worst quality, low quality, score 1, score 2, score 3, jpeg artifacts, logo, watermark, bad anatomy, bad hands'
  );

  // ── ANIMA model fields
  const [unetName, setUnetName] = useState('');
  const [clipName, setClipName] = useState('');
  const [animaVae, setAnimaVae] = useState('');

  // ── SDXL KNX model fields
  const [checkpointName, setCheckpointName] = useState('');
  const [sdxlVae, setSdxlVae] = useState('');

  // ── Shared sampler state (defaults updated when mode switches)
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(4);
  const [sampler, setSampler] = useState('dpmpp_2m_sde_gpu');
  const [scheduler, setScheduler] = useState('sgm_uniform');
  const [seed, setSeed] = useState(-1);
  const [width, setWidth] = useState(832);
  const [height, setHeight] = useState(1216);
  const [batchSize, setBatchSize] = useState(1);

  // ── LoRA stack (converted to LoRA Manager text format at submit time)
  const [loras, setLoras] = useState<LoraEntry[]>([]);

  // ── Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<ComfyOutputFile[]>([]);
  const lastPromptIdRef = useRef<string | null>(null);

  // Apply template defaults when mode switches
  useEffect(() => {
    const d = TEMPLATE_DEFAULTS[templateMode];
    setSteps(d.steps);
    setCfg(d.cfg);
    setSampler(d.sampler);
    setScheduler(d.scheduler);
    setWidth(d.width);
    setHeight(d.height);
  }, [templateMode]);

  // Detect generation completion and fetch result images
  useEffect(() => {
    if (currentPromptId) {
      lastPromptIdRef.current = currentPromptId;
      setIsGenerating(true);
    } else if (lastPromptIdRef.current && isGenerating) {
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

  /** Convert LoraEntry[] → LoRA Manager text format for template injection. */
  const loraText = loras
    .filter(l => l.name.trim())
    .map(l => `<lora:${l.name.trim()}:${(l.modelWeight ?? 1).toFixed(2)}>`)
    .join('');

  /** Whether the required model fields for the current mode are filled. */
  const canGenerate = templateMode === 'anima'
    ? Boolean(unetName.trim())
    : Boolean(checkpointName.trim());

  // ── Generate
  const handleGenerate = useCallback(async () => {
    if (isGenerating || !canGenerate) return;

    try {
      if (templateMode === 'anima') {
        const patch = buildAnimaPatch({
          unetName: unetName.trim(),
          clipName: clipName.trim() || undefined,
          vaeName: animaVae.trim() || undefined,
          positivePrompt, negativePrompt,
          steps, cfg, sampler, scheduler, seed,
          width, height, batchSize,
          loraText: loraText || undefined,
        });
        const { apiPrompt, workflow } = injectTemplate(animaWorkflow, patch);
        await submitPrompt({
          prompt: apiPrompt,
          extra_data: { extra_pnginfo: { workflow: workflow as unknown as Record<string, unknown> } },
        });
      } else {
        const patch = buildSdxlKnxPatch({
          checkpointName: checkpointName.trim(),
          vaeName: sdxlVae.trim() || undefined,
          positivePrompt, negativePrompt,
          steps, cfg, sampler, scheduler, seed,
          width, height, batchSize,
          loraText: loraText || undefined,
        });
        const { apiPrompt, workflow } = injectTemplate(sdxlWorkflow, patch);
        await submitPrompt({
          prompt: apiPrompt,
          extra_data: { extra_pnginfo: { workflow: workflow as unknown as Record<string, unknown> } },
        });
      }
    } catch (err) {
      console.error('Generate failed:', err);
    }
  }, [
    isGenerating, canGenerate, templateMode,
    unetName, clipName, animaVae, checkpointName, sdxlVae,
    positivePrompt, negativePrompt,
    steps, cfg, sampler, scheduler, seed,
    width, height, batchSize, loraText,
    submitPrompt,
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

              {/* Architecture switcher */}
              <div className="space-y-1">
                <SectionLabel>Workflow</SectionLabel>
                <ArchSwitcher mode={templateMode} onChange={setTemplateMode} />
                <FieldHint>
                  {templateMode === 'anima'
                    ? 'Guy90s ANIMA workflow — AuraFlow + Qwen encoder'
                    : 'KNX SDXL fork — NoobAI-XL / Illustrious + Adetailer'}
                </FieldHint>
              </div>

              <Separator />

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

              {/* Model — ANIMA */}
              {templateMode === 'anima' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <SectionLabel>Model — ANIMA</SectionLabel>
                    <Button
                      variant="ghost" size="sm"
                      className="h-6 gap-1 text-xs"
                      onClick={models.refresh}
                      disabled={models.loading}
                    >
                      <RefreshCw className={`h-3 w-3${models.loading ? ' animate-spin' : ''}`} />
                      {models.loading ? 'Loading…' : 'Refresh'}
                    </Button>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Diffusion model (UNET)</Label>
                    <ModelPicker
                      value={unetName}
                      onChange={setUnetName}
                      models={models.diffusionModels}
                      loading={models.loading}
                      placeholder="anima_baseV10.safetensors"
                      onRefresh={models.refresh}
                    />
                    <FieldHint>
                      Place in <code className="font-mono text-[10px]">ComfyUI/models/diffusion_models/</code>
                    </FieldHint>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Text encoder{' '}
                      <span className="text-muted-foreground/60">(optional)</span>
                    </Label>
                    <ModelPicker
                      value={clipName}
                      onChange={setClipName}
                      models={models.clipModels}
                      loading={models.loading}
                      placeholder="qwen_3_06b_base.safetensors"
                      onRefresh={models.refresh}
                    />
                    <FieldHint>
                      Qwen encoder — place in <code className="font-mono text-[10px]">ComfyUI/models/text_encoders/</code>
                    </FieldHint>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">
                      VAE{' '}
                      <span className="text-muted-foreground/60">(optional)</span>
                    </Label>
                    <ModelPicker
                      value={animaVae}
                      onChange={setAnimaVae}
                      models={models.vaeModels}
                      loading={models.loading}
                      placeholder="qwen_image_vae.safetensors"
                      onRefresh={models.refresh}
                    />
                    <FieldHint>
                      Place in <code className="font-mono text-[10px]">ComfyUI/models/vae/</code>
                    </FieldHint>
                  </div>
                </div>
              )}

              {/* Model — SDXL KNX */}
              {templateMode === 'sdxl-knx' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <SectionLabel>Model — SDXL</SectionLabel>
                    <Button
                      variant="ghost" size="sm"
                      className="h-6 gap-1 text-xs"
                      onClick={models.refresh}
                      disabled={models.loading}
                    >
                      <RefreshCw className={`h-3 w-3${models.loading ? ' animate-spin' : ''}`} />
                      {models.loading ? 'Loading…' : 'Refresh'}
                    </Button>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Checkpoint</Label>
                    <ModelPicker
                      value={checkpointName}
                      onChange={setCheckpointName}
                      models={models.checkpoints}
                      loading={models.loading}
                      placeholder="NoobAI/anime/retirementMixNAIXL_v10.safetensors"
                      onRefresh={models.refresh}
                    />
                    <FieldHint>
                      Place in <code className="font-mono text-[10px]">ComfyUI/models/checkpoints/</code> — includes MODEL + CLIP + VAE
                    </FieldHint>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Decode VAE{' '}
                      <span className="text-muted-foreground/60">(optional)</span>
                    </Label>
                    <ModelPicker
                      value={sdxlVae}
                      onChange={setSdxlVae}
                      models={models.vaeModels}
                      loading={models.loading}
                      placeholder="illustriousXLV10_v10.safetensors"
                      onRefresh={models.refresh}
                    />
                    <FieldHint>
                      Place in <code className="font-mono text-[10px]">ComfyUI/models/vae/</code> — used for image decode only; Adetailer + upscale use the checkpoint VAE.
                    </FieldHint>
                  </div>
                </div>
              )}

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
                    <Input type="number" value={width} onChange={e => setWidth(Number(e.target.value))} step={64} min={64} max={4096} className="h-8 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">H</Label>
                    <Input type="number" value={height} onChange={e => setHeight(Number(e.target.value))} step={64} min={64} max={4096} className="h-8 text-xs" />
                  </div>
                </div>
              </div>

              <Separator />

              {/* Sampler */}
              <div className="space-y-3">
                <SectionLabel>Sampler</SectionLabel>

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Sampler</Label>
                    <Select value={sampler} onValueChange={setSampler}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SAMPLERS.map(s => <SelectItem key={s} value={s} className="text-xs">{s}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Scheduler</Label>
                    <Select value={scheduler} onValueChange={setScheduler}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SCHEDULERS.map(s => <SelectItem key={s} value={s} className="text-xs">{s}</SelectItem>)}
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

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Batch size</Label>
                    <Input type="number" value={batchSize} onChange={e => setBatchSize(Math.max(1, Number(e.target.value)))} min={1} max={8} className="h-8 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Seed</Label>
                    <div className="flex gap-1">
                      <Input
                        type="number"
                        value={seed}
                        onChange={e => setSeed(Number(e.target.value))}
                        className="h-8 text-xs flex-1 font-mono"
                        placeholder="-1"
                      />
                      <Button variant="outline" size="icon" className="h-8 w-8 shrink-0" onClick={randomSeed} title="Random seed">
                        <Shuffle className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="outline" size="icon" className="h-8 w-8 shrink-0" onClick={() => setSeed(-1)} title="Reset (-1)">
                        <RefreshCw className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                </div>
                <FieldHint>Seed -1 lets ComfyUI randomise each run</FieldHint>
              </div>

              <Separator />

              {/* LoRA stack */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <SectionLabel>LoRAs</SectionLabel>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm" className="h-6 gap-1 text-xs" asChild>
                      <a href="/comfyui/" target="_blank" rel="noopener noreferrer" title="Open LoRA Manager in ComfyUI">
                        <ExternalLink className="h-3 w-3" /> Manager
                      </a>
                    </Button>
                    <Button variant="ghost" size="sm" className="h-6 gap-1 text-xs" onClick={addLora}>
                      <Plus className="h-3 w-3" /> Add
                    </Button>
                  </div>
                </div>
                {loras.length === 0 && (
                  <p className="text-xs text-muted-foreground/60">No LoRAs added</p>
                )}
                <div className="space-y-1.5">
                  {loras.map((lora, i) => (
                    <LoraRow key={i} lora={lora} index={i} onChange={updateLora} onRemove={removeLora} />
                  ))}
                </div>
                {loras.length > 0 && (
                  <FieldHint>Type the filename or use Manager → to browse and refresh your LoRA list</FieldHint>
                )}
              </div>

            </div>
          </ScrollArea>

          {/* Generate / Stop */}
          <div className="shrink-0 border-t border-border/40 p-3">
            <ClickSpark sparkColor="hsl(var(--primary))" sparkCount={10} sparkRadius={25}>
              {isGenerating ? (
                <Button variant="destructive" className="w-full gap-2" onClick={interrupt}>
                  <Square className="h-4 w-4" />
                  Stop generation
                </Button>
              ) : (
                <Button className="w-full gap-2" onClick={handleGenerate} disabled={!canGenerate}>
                  Generate
                  {queueRemaining > 0 && (
                    <span className="ml-1 text-xs opacity-70">({queueRemaining} queued)</span>
                  )}
                </Button>
              )}
            </ClickSpark>
            {!canGenerate && (
              <p className="mt-1.5 text-center text-xs text-muted-foreground/60">
                {templateMode === 'anima' ? 'Enter a UNET model name to generate' : 'Enter a checkpoint name to generate'}
              </p>
            )}
          </div>
        </div>
      </SplitterPanel>

      {/* ── Right panel: gallery ── */}
      <SplitterPanel>
        <div className="flex h-full flex-col">
          <div className="flex shrink-0 items-center justify-between border-b border-border/40 px-3 py-2">
            <span className="text-xs font-medium text-muted-foreground">
              {generatedImages.length > 0
                ? `${generatedImages.length} image${generatedImages.length !== 1 ? 's' : ''}`
                : 'Gallery'}
            </span>
            {generatedImages.length > 0 && (
              <Button variant="ghost" size="sm" className="h-6 text-xs text-muted-foreground" onClick={() => setGeneratedImages([])}>
                Clear
              </Button>
            )}
          </div>

          <div className="min-h-0 flex-1">
            <ImageGallery images={generatedImages} />
          </div>

          <GenerationProgress progress={progress} currentNode={currentNode} isGenerating={isGenerating} />
        </div>
      </SplitterPanel>

    </Splitter>
  );
}
