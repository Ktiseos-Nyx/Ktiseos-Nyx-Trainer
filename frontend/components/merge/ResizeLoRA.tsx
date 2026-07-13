'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Loader2, Minimize2 } from 'lucide-react';
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

export function ResizeLoRATab() {
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
