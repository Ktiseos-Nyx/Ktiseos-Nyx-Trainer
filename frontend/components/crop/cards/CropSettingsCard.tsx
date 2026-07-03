'use client';

import { Crop, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertDescription } from '@/components/ui/alert';

const RESOLUTION_OPTIONS = [
  { value: 256, label: '256px — Icons / Thumbnails' },
  { value: 512, label: '512px — SD 1.5' },
  { value: 768, label: '768px — Medium' },
  { value: 1024, label: '1024px — SDXL' },
  { value: 1536, label: '1536px — High-res' },
  { value: 2048, label: '2048px — Ultra' },
];

const ASPECT_OPTIONS = [
  { value: '1:1', label: '1:1', w: 1, h: 1 },
  { value: '3:2', label: '3:2', w: 3, h: 2 },
  { value: '4:3', label: '4:3', w: 4, h: 3 },
  { value: '16:9', label: '16:9', w: 16, h: 9 },
  { value: '9:16', label: '9:16 (Portrait)', w: 9, h: 16 },
];

interface CropSettingsCardProps {
  targetResolution: number;
  setTargetResolution: (value: number) => void;
  aspectRatio: string;
  setAspectRatio: (value: string) => void;
  outputMode: string;
  setOutputMode: (value: string) => void;
  outputFormat: string;
  setOutputFormat: (value: string) => void;
  quality: number;
  setQuality: (value: number) => void;
  disabled?: boolean;
}

export function CropSettingsCard({
  targetResolution,
  setTargetResolution,
  aspectRatio,
  setAspectRatio,
  outputMode,
  setOutputMode,
  outputFormat,
  setOutputFormat,
  quality,
  setQuality,
  disabled,
}: CropSettingsCardProps) {
  const showQuality = outputFormat === 'jpg' || outputFormat === 'webp';

  return (
    <Card className="border-emerald-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Crop className="h-5 w-5 text-emerald-400" />
          Crop Settings
        </CardTitle>
        <CardDescription>Configure target size, aspect ratio, and output</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Target Resolution */}
        <div className="space-y-2">
          <Label>Target Resolution</Label>
          <Select
            value={String(targetResolution)}
            onValueChange={(v) => setTargetResolution(Number(v))}
            disabled={disabled}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select resolution" />
            </SelectTrigger>
            <SelectContent>
              {RESOLUTION_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Aspect Ratio */}
        <div className="space-y-2">
          <Label>Aspect Ratio</Label>
          <Select value={aspectRatio} onValueChange={setAspectRatio} disabled={disabled}>
            <SelectTrigger>
              <SelectValue placeholder="Select aspect ratio" />
            </SelectTrigger>
            <SelectContent>
              {ASPECT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Output Format */}
        <div className="space-y-2">
          <Label>Output Format</Label>
          <Select value={outputFormat} onValueChange={setOutputFormat} disabled={disabled}>
            <SelectTrigger>
              <SelectValue placeholder="Select format" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="webp">WebP — Smallest size</SelectItem>
              <SelectItem value="jpg">JPEG — Universal compatibility</SelectItem>
              <SelectItem value="png">PNG — Lossless</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Quality Slider */}
        {showQuality && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Quality</Label>
              <span className="text-sm font-mono text-muted-foreground">{quality}%</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
              disabled={disabled}
              className="w-full accent-emerald-500"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Smaller file</span>
              <span>Higher quality</span>
            </div>
          </div>
        )}

        {/* Output Mode */}
        <div className="space-y-3">
          <Label>Output Mode</Label>
          <RadioGroup
            value={outputMode}
            onValueChange={setOutputMode}
            disabled={disabled}
            className="space-y-3"
          >
            <div className="flex items-start space-x-3 rounded-lg border p-3 hover:bg-accent/50">
              <RadioGroupItem value="new_dataset" id="crop-new-dataset" className="mt-0.5" />
              <div className="space-y-1">
                <Label htmlFor="crop-new-dataset" className="cursor-pointer font-medium">
                  New Dataset
                </Label>
                <p className="text-sm text-muted-foreground">
                  Creates a new folder (e.g., <span className="font-mono">my_dataset_512x512/</span>).
                  Originals untouched.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 rounded-lg border p-3 hover:bg-accent/50">
              <RadioGroupItem value="in-place" id="crop-in-place" className="mt-0.5" />
              <div className="space-y-1">
                <Label htmlFor="crop-in-place" className="cursor-pointer font-medium">
                  Crop In-Place
                </Label>
                <p className="text-sm text-muted-foreground">
                  Overwrites original files. Cannot be undone.
                </p>
              </div>
            </div>
          </RadioGroup>
        </div>

        {outputMode === 'in-place' && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              In-place cropping overwrites your original images. Consider backing up your
              dataset first.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

export { ASPECT_OPTIONS };
