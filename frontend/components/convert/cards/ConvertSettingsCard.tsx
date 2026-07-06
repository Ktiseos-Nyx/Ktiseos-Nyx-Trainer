/**
 * Convert Settings Card — target format, quality, output mode.
 */

'use client';

import { AlertTriangle, ArrowRightLeft } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface ConvertSettingsCardProps {
  targetFormat: string;
  setTargetFormat: (value: string) => void;
  quality: number;
  setQuality: (value: number) => void;
  outputMode: string;
  setOutputMode: (value: string) => void;
  disabled?: boolean;
}

export function ConvertSettingsCard({
  targetFormat,
  setTargetFormat,
  quality,
  setQuality,
  outputMode,
  setOutputMode,
  disabled,
}: ConvertSettingsCardProps) {
  const showQuality = targetFormat === 'jpg' || targetFormat === 'webp';

  return (
    <Card className="border-blue-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ArrowRightLeft className="h-5 w-5 text-blue-400" />
          Conversion Settings
        </CardTitle>
        <CardDescription>Choose the target format and output options</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Target Format */}
        <div className="space-y-2">
          <Label>Target Format</Label>
          <Select value={targetFormat} onValueChange={setTargetFormat} disabled={disabled}>
            <SelectTrigger>
              <SelectValue placeholder="Select format" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="webp">WebP — Smallest size, modern browsers</SelectItem>
              <SelectItem value="jpg">JPEG — Universal compatibility</SelectItem>
              <SelectItem value="png">PNG — Lossless, larger files</SelectItem>
              <SelectItem value="bmp">BMP — Uncompressed, largest files</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Quality Slider (only for lossy formats) */}
        {showQuality && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Quality</Label>
              <span className="text-sm font-mono text-muted-foreground">{quality}%</span>
            </div>
            <Slider
              value={[quality]}
              onValueChange={([v]) => setQuality(v)}
              min={1}
              max={100}
              step={1}
              disabled={disabled}
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
              <RadioGroupItem value="new_dataset" id="new-dataset" className="mt-0.5" />
              <div className="space-y-1">
                <Label htmlFor="new-dataset" className="cursor-pointer font-medium">
                  New Dataset
                </Label>
                <p className="text-sm text-muted-foreground">
                  Creates a new folder (e.g., <span className="font-mono">my_dataset_webp/</span>).
                  Originals untouched.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 rounded-lg border p-3 hover:bg-accent/50">
              <RadioGroupItem value="in-place" id="in-place" className="mt-0.5" />
              <div className="space-y-1">
                <Label htmlFor="in-place" className="cursor-pointer font-medium">
                  Convert In-Place
                </Label>
                <p className="text-sm text-muted-foreground">
                  Overwrites original files. Cannot be undone.
                </p>
              </div>
            </div>
          </RadioGroup>
        </div>

        {/* Warning for in-place mode */}
        {outputMode === 'in-place' && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              In-place conversion overwrites your original images. Consider backing up your
              dataset first.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
