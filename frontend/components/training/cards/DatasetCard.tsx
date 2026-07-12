/**
 * Dataset Configuration Card
 * Essential settings: data paths, resolution, batch size, epochs/steps
 * Now with subfolder repeat configuration for varied per-folder training.
 */

'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TextFormField, NumberFormField, ComboboxFormField } from '../fields/FormFields';
import type { DatasetSubsetConfig, TrainingConfig } from '@/lib/api';
import { datasetAPI } from '@/lib/api';
import { Database, ImageIcon, FolderSearch, Trash2, Loader2, Save } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

interface DatasetCardProps {
  form: UseFormReturn<TrainingConfig>;
  datasets: { value: string; label: string }[];
  onSave?: () => void;
}

interface SubfolderInfo {
  name: string;
  path: string;
  image_count: number;
}

export function DatasetCard({ form, datasets, onSave }: DatasetCardProps) {
  const resolution = form.watch('resolution');
  const batchSize = form.watch('train_batch_size');
  const maxEpochs = form.watch('max_train_epochs');
  const maxSteps = form.watch('max_train_steps');
  const trainDataDir = form.watch('train_data_dir');
  const subsets = form.watch('subsets') || [];

  const estimatedVRAM = resolution === 1024 ? (batchSize || 1) * 8 : (batchSize || 1) * 4;

  const [subfolders, setSubfolders] = useState<SubfolderInfo[]>([]);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  const scanSubfolders = useCallback(async () => {
    if (!trainDataDir) return;
    setScanning(true);
    setScanError(null);
    try {
      const result = await datasetAPI.listSubfolders(trainDataDir);
      setSubfolders(result.subfolders || []);
    } catch (err: any) {
      setScanError(err.message || 'Failed to scan subfolders');
      setSubfolders([]);
    } finally {
      setScanning(false);
    }
  }, [trainDataDir]);

  const prevDataDir = useRef<string | null>(null);

  // Auto-scan when dataset changes, clear stale subsets on dataset switch
  useEffect(() => {
    if (trainDataDir) {
      scanSubfolders();
      if (prevDataDir.current && prevDataDir.current !== trainDataDir) {
        form.setValue('subsets', [], { shouldDirty: false });
      }
    } else {
      setSubfolders([]);
      form.setValue('subsets', [], { shouldDirty: false });
    }
    prevDataDir.current = trainDataDir || null;
  }, [trainDataDir, scanSubfolders, form]);

  const isSubsetActive = (folderName: string) =>
    subsets.some((s: DatasetSubsetConfig) => s.image_dir === folderName);

  const toggleSubset = (folderName: string, imageCount: number) => {
    const current = form.getValues('subsets') || [];
    if (isSubsetActive(folderName)) {
      form.setValue(
        'subsets',
        current.filter((s: DatasetSubsetConfig) => s.image_dir !== folderName),
        { shouldDirty: true }
      );
    } else {
      form.setValue(
        'subsets',
        [...current, { image_dir: folderName, num_repeats: 10 }],
        { shouldDirty: true }
      );
    }
  };

  const updateSubsetRepeats = (folderName: string, numRepeats: number) => {
    const current = form.getValues('subsets') || [];
    form.setValue(
      'subsets',
      current.map((s: DatasetSubsetConfig) =>
        s.image_dir === folderName ? { ...s, num_repeats: numRepeats } : s
      ),
      { shouldDirty: true }
    );
  };

  const removeSubset = (folderName: string) => {
    const current = form.getValues('subsets') || [];
    form.setValue(
      'subsets',
      current.filter((s: DatasetSubsetConfig) => s.image_dir !== folderName),
      { shouldDirty: true }
    );
  };

  return (
    <Card className="border-blue-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-blue-400" />
            Dataset Configuration
          </CardTitle>
          <CardDescription>
            Training data paths and basic training parameters
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save Dataset
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <ComboboxFormField
          form={form}
          name="train_data_dir"
          label="Training Data Directory"
          description="Select a dataset from your /dataset folder"
          placeholder="Select a dataset..."
          options={datasets}
        />

        <TextFormField
          form={form}
          name="output_dir"
          label="Output Directory (Default)"
          description="Trained LoRA files will be saved here"
          placeholder="output/my_lora"
          readOnly={true}
        />

        {/* ── Subfolder Repeats ── */}
        {trainDataDir && (
          <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-foreground flex items-center gap-1.5">
                  <FolderSearch className="h-4 w-4 text-blue-400" />
                  Subfolder Repeats
                </h4>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Configure different repeat counts per subfolder for varied concept weighting
                </p>
              </div>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={scanSubfolders}
                disabled={scanning}
                className="gap-1 h-8 text-xs"
              >
                {scanning ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <FolderSearch className="h-3 w-3" />
                )}
                {scanning ? 'Scanning...' : 'Rescan'}
              </Button>
            </div>

            {scanError && (
              <Alert variant="destructive" className="py-2">
                <AlertDescription className="text-xs">{scanError}</AlertDescription>
              </Alert>
            )}

            {subfolders.length === 0 && !scanning && !scanError && (
              <p className="text-xs text-muted-foreground">
                No subfolders with images found. Subfolders allow different repeat counts per concept.
              </p>
            )}

            {subfolders.length > 0 && (
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                {subfolders.map((sf) => {
                  const active = isSubsetActive(sf.name);
                  const activeConfig = subsets.find(
                    (s: DatasetSubsetConfig) => s.image_dir === sf.name
                  );
                  return (
                    <div
                      key={sf.name}
                      className={`flex items-center gap-3 rounded-md border px-3 py-2 transition-colors ${
                        active
                          ? 'border-blue-500/40 bg-blue-500/10'
                          : 'border-border bg-muted/30'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => toggleSubset(sf.name, sf.image_count)}
                        className={`flex-shrink-0 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                          active
                            ? 'bg-blue-500 border-blue-500 text-white'
                            : 'border-muted-foreground/30 hover:border-blue-400/60'
                        }`}
                      >
                        {active && (
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                            <path d="M2 6L5 9L10 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </button>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium truncate">{sf.name}</span>
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-5">
                            {sf.image_count} imgs
                          </Badge>
                        </div>
                      </div>

                      {active && (
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            min={1}
                            max={1000}
                            value={activeConfig?.num_repeats ?? 10}
                            onChange={(e) =>
                              updateSubsetRepeats(sf.name, parseInt(e.target.value) || 1)
                            }
                            className="w-16 h-8 text-xs text-center"
                          />
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            onClick={() => removeSubset(sf.name)}
                            className="h-7 w-7 text-muted-foreground hover:text-red-400"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {subsets.length > 0 && (
              <p className="text-xs text-muted-foreground">
                {subsets.length} subfolder{subsets.length !== 1 ? 's' : ''} active with varied
                repeats. The global &quot;Dataset Repeats&quot; below is ignored when subfolder
                repeats are configured.
              </p>
            )}
          </div>
        )}

        {/* Resolution */}
        <NumberFormField
          form={form}
          name="resolution"
          label="Training Resolution"
          description="Must be divisible by 64. SDXL: 1024, SD1.5: 512"
          placeholder="1024"
          min={256}
          max={4096}
          step={64}
        />

        {/* Batch Size */}
        <NumberFormField
          form={form}
          name="train_batch_size"
          label="Batch Size"
          description={`Lower = less VRAM. Est. VRAM: ~${estimatedVRAM}GB`}
          placeholder="1"
          min={1}
        />

        {/* Repeats */}
        <NumberFormField
          form={form}
          name="num_repeats"
          label="Dataset Repeats"
          description={
            subsets.length > 0
              ? 'Ignored — subfolder repeats above are active'
              : 'Higher = more training on same images. Typical: 10-20'
          }
          placeholder="10"
          min={1}
          max={1000}
          disabled={subsets.length > 0}
        />

        {/* Training Duration */}
        <div className="grid grid-cols-2 gap-4">
          <NumberFormField
            form={form}
            name="max_train_epochs"
            label="Max Epochs"
            description="Number of times to train on full dataset"
            placeholder="10"
            min={0}
            max={10000}
          />

          <NumberFormField
            form={form}
            name="max_train_steps"
            label="Max Steps (Optional)"
            description="If > 0, overrides epochs. 0 = use epochs"
            placeholder="0"
            min={0}
          />
        </div>

        {maxSteps > 0 && maxEpochs > 0 && (
          <Alert>
            <AlertDescription>
              <ImageIcon className="h-4 w-4 inline mr-2" />
              Max steps ({maxSteps}) will override max epochs ({maxEpochs})
            </AlertDescription>
          </Alert>
        )}

        {/* Seed */}
        <NumberFormField
          form={form}
          name="seed"
          label="Random Seed"
          description="For reproducible results. Use same seed for consistent training"
          placeholder="42"
          min={0}
        />
      </CardContent>
    </Card>
  );
}
