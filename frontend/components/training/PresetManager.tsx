/* eslint-disable react-hooks/set-state-in-effect */
'use client';

import { Fragment, useState, useEffect } from 'react';
import { Save, FolderOpen, Trash2, Download, Upload, Sparkles } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { presetsAPI, type TrainingConfig, type PresetMetadata } from '@/lib/api';

const MODEL_TYPE_ORDER = ['SD15', 'SDXL', 'FLUX', 'SD3', 'SD3.5', 'LUMINA', 'Chroma', 'Anima', 'HunyuanImage'];

/**
 * Group presets by model type, sorting each bucket alphabetically by name.
 * Presets without a model type are placed under the 'Other' bucket.
 */
function groupByModelType<T>(presets: T[], getType: (p: T) => string | undefined, getName: (p: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const preset of presets) {
    const type = getType(preset) || 'Other';
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type)!.push(preset);
  }
  for (const items of groups.values()) {
    items.sort((a, b) => getName(a).localeCompare(getName(b)));
  }
  return groups;
}

/**
 * Return model-type bucket keys in display order: types listed in MODEL_TYPE_ORDER
 * first (preserving that order), then any remaining types (e.g. 'Other') sorted
 * alphabetically.
 */
function sortedModelTypeKeys(groups: Map<string, unknown[]>): string[] {
  return [...groups.keys()].sort((a, b) => {
    const ai = MODEL_TYPE_ORDER.indexOf(a);
    const bi = MODEL_TYPE_ORDER.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return a.localeCompare(b);
  });
}

const LEGACY_OBSOLETE_FIELDS = new Set([
  'train_text_encoder', 'xformers', 'mem_eff_attn', 'use_8bit_adam',
  'create_buckets', 'create_caption', 'full_path', 'use_latent_files',
  'num_cpu_threads_per_process', 'LoRA_type', 'noise_offset_type',
  'additional_parameters', 'scale_weight_norms', 'decompose_both',
  'save_last_n_steps',
]);

/**
 * Maps old kohya-ss GUI preset field names to the current TrainingConfig schema.
 * Only called for presets with a nested 'config' block (old format).
 */
function normalizeLegacyPresetFields(cfg: Record<string, unknown>): Partial<TrainingConfig> {
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(cfg)) {
    if (LEGACY_OBSOLETE_FIELDS.has(key)) continue;

    switch (key) {
      case 'optimizer':
        // Old field; map to current name, fix common casing variations
        if (!cfg.optimizer_type) {
          const normalized = String(value) === 'Adafactor' ? 'AdaFactor' : value;
          out.optimizer_type = normalized;
        }
        break;
      case 'epoch':
        if (!cfg.max_train_epochs) {
          const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
          if (!isNaN(n) && n > 0) out.max_train_epochs = n;
        }
        break;
      case 'learning_rate':
        if (!cfg.unet_lr) {
          const lr = typeof value === 'string' ? parseFloat(value) : Number(value);
          if (!isNaN(lr) && lr > 0) { out.unet_lr = lr; out.text_encoder_lr = lr; }
        }
        break;
      case 'batch_size':
        if (!cfg.train_batch_size) {
          const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
          if (!isNaN(n) && n > 0) out.train_batch_size = n;
        }
        break;
      case 'lr_warmup':
        if (!cfg.lr_warmup_steps) {
          const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
          if (!isNaN(n)) out.lr_warmup_steps = n;
        }
        break;
      case 'max_resolution': {
        if (!cfg.resolution) {
          const res = parseInt(String(value).split(',')[0], 10);
          if (!isNaN(res) && res >= 64) out.resolution = res;
        }
        break;
      }
      case 'dataset_repeats':
        if (!cfg.num_repeats) {
          const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
          if (!isNaN(n) && n > 0) out.num_repeats = n;
        }
        break;
      default:
        // Pass through current-schema fields, skip empty-string values that would
        // fail Zod validation on number/enum fields.
        if (value !== '' && value !== null) out[key] = value;
    }
  }

  return out as Partial<TrainingConfig>;
}

interface CustomPreset {
  id: string;
  name: string;
  description: string;
  config: Partial<TrainingConfig>;
  createdAt: number;
}

interface PresetManagerProps {
  currentConfig: Partial<TrainingConfig>;
  onLoadPreset: (config: Partial<TrainingConfig>) => void;
  onSavePreset?: (preset: CustomPreset) => void;
}

export default function PresetManager({
  currentConfig,
  onLoadPreset,
  onSavePreset,
}: PresetManagerProps) {
  // ✅ FIX: Initialize empty list first (Server Safe)
  const [customPresets, setCustomPresets] = useState<CustomPreset[]>([]);
  const [serverPresets, setServerPresets] = useState<PresetMetadata[]>([]);
  const [loadingPresets, setLoadingPresets] = useState(true);

  // ✅ FIX: Load from localStorage only in the browser
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('training-presets');
      if (saved) {
        try {
          setCustomPresets(JSON.parse(saved));
        } catch (e) {
          console.error('Failed to load custom presets:', e);
        }
      }
    }
  }, []);

  // Load server presets on mount
  useEffect(() => {
    const loadServerPresets = async () => {
      try {
        const { presets } = await presetsAPI.list();
        setServerPresets(presets);
      } catch (error) {
        console.error('Failed to load server presets:', error);
        // Fall back to localStorage only
      } finally {
        setLoadingPresets(false);
      }
    };

    loadServerPresets();
  }, []);

  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  // Save custom presets to localStorage
  useEffect(() => {
    // Only save if we have items (and ignore the initial empty state if possible)
    // or if we explicitly want to persist changes.
    // Checking window ensures we don't crash.
    if (typeof window !== 'undefined' && customPresets.length > 0) {
      localStorage.setItem('training-presets', JSON.stringify(customPresets));
    }
  }, [customPresets]);

  // Filter out project-specific fields when saving presets
  // Only save reusable training hyperparameters (like Bmaltais's JSON presets)
  const filterPresetConfig = (config: Partial<TrainingConfig>): Partial<TrainingConfig> => {
    const {
      // Exclude project/file-specific fields
      project_name,
      pretrained_model_name_or_path,
      vae_path,
      ae_path,
      clip_l_path,
      clip_g_path,
      t5xxl_path,
      continue_from_lora,
      wandb_key,
      train_data_dir,
      output_dir,
      output_name,
      resume_from_state,
      sample_prompts,
      logging_dir,
      log_tracker_name,
      log_tracker_config,
      wandb_run_name,
      gemma2,
      // Keep everything else (reusable hyperparameters)
      ...reusableConfig
    } = config;

    return reusableConfig;
  };

  const handleSavePreset = async () => {
    if (!newPresetName.trim()) {
      toast.warning('Please enter a preset name');
      return;
    }

    // Filter config to only include reusable hyperparameters
    const filteredConfig = filterPresetConfig(currentConfig);

    try {
      // Try saving to server first
      await presetsAPI.save({
        name: newPresetName,
        description: newPresetDescription || 'Custom preset',
        model_type: currentConfig.model_type,
        config: filteredConfig,
      });

      // Reload presets from server
      const { presets } = await presetsAPI.list();
      setServerPresets(presets);

      setNewPresetName('');
      setNewPresetDescription('');
      setSaveDialogOpen(false);

      toast.success(`Preset "${newPresetName}" saved`, {
        description: 'Project-specific details (dataset paths, model paths, project name) are not included in presets.',
      });
    } catch (error) {
      console.error('Failed to save to server, falling back to localStorage:', error);

      // Fallback to localStorage
      const preset: CustomPreset = {
        id: `custom_${Date.now()}`,
        name: newPresetName,
        description: newPresetDescription || 'Custom preset',
        config: filteredConfig,
        createdAt: Date.now(),
      };

      setCustomPresets((prev) => [...prev, preset]);
      onSavePreset?.(preset);

      setNewPresetName('');
      setNewPresetDescription('');
      setSaveDialogOpen(false);

      toast.success(`Preset "${newPresetName}" saved locally`, {
        description: 'Server unavailable, saved to browser storage only.',
      });
    }
  };

  const handleLoadPreset = async (presetId: string) => {
    // Check localStorage presets
    const localPreset = customPresets.find((p) => p.id === presetId);
    if (localPreset) {
      onLoadPreset(localPreset.config);
      toast.success(`Loaded preset: ${localPreset.name}`, {
        description: 'You\'ll still need to set your dataset, model paths, and project name.',
      });
      return;
    }

    // Try loading from server
    try {
      const preset = await presetsAPI.get(presetId);

      // Strip metadata fields that aren't part of TrainingConfig.
      // Also strip legacy top-level 'optimizer' and 'source' which some old presets include.
      const { name, description, notes, base_model, optimizer, is_builtin, created_at, source, ...rest } = preset as any;

      // Handle two preset formats:
      // - Flat (new): training params at root level (e.g. optimizer_type, resolution)
      // - Nested (old / user-saved): training params under a 'config' key with legacy field names
      let config: Partial<TrainingConfig>;
      if (rest.config && typeof rest.config === 'object' && !Array.isArray(rest.config)) {
        const { config: nestedConfig, ...topLevelFields } = rest;
        config = { ...topLevelFields, ...normalizeLegacyPresetFields(nestedConfig) };
      } else {
        config = rest;
      }

      onLoadPreset(config);
      toast.success(`Loaded preset: ${name || presetId}`, {
        description: 'You\'ll still need to set your dataset, model paths, and project name.',
      });
    } catch (error) {
      console.error('Failed to load preset from server:', error);
      toast.error('Failed to load preset');
    }
  };

  const handleDeletePreset = async (presetId: string) => {
    if (!confirm('Are you sure you want to delete this preset?')) return;

    // Check if it's a server preset
    const serverPreset = serverPresets.find((p) => p.id === presetId);
    if (serverPreset) {
      if (serverPreset.is_builtin) {
        toast.error('Cannot delete built-in presets');
        return;
      }

      try {
        await presetsAPI.delete(presetId);

        // Reload presets from server
        const { presets } = await presetsAPI.list();
        setServerPresets(presets);

        if (selectedPreset === presetId) {
          setSelectedPreset('');
        }

        toast.success('Preset deleted');
        return;
      } catch (error) {
        console.error('Failed to delete from server:', error);
        toast.error('Failed to delete preset from server');
        return;
      }
    }

    // Delete from localStorage
    setCustomPresets((prev) => prev.filter((p) => p.id !== presetId));
    if (selectedPreset === presetId) {
      setSelectedPreset('');
    }

    toast.success('Preset deleted from local storage');
  };

  const handleExportPresets = () => {
    const dataStr = JSON.stringify(customPresets, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `training-presets-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleImportPresets = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const imported = JSON.parse(e.target?.result as string) as CustomPreset[];
        setCustomPresets((prev) => [...prev, ...imported]);
        toast.success(`Imported ${imported.length} presets`);
      } catch (error) {
        toast.error('Failed to import presets. Invalid file format.');
      }
    };
    reader.readAsText(file);
  };

  const builtins = serverPresets.filter(p => p.is_builtin);
  const userServer = serverPresets.filter(p => !p.is_builtin);
  const builtinGroups = groupByModelType(builtins, p => p.model_type, p => p.name);
  const userServerGroups = groupByModelType(userServer, p => p.model_type, p => p.name);
  const browserGroups = groupByModelType(customPresets, p => p.config.model_type, p => p.name);

  return (
    <Card className="border-purple-500/30 bg-gradient-to-br from-purple-500/5 via-card to-card dark:from-purple-500/10 dark:via-card dark:to-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-purple-400" />
          Configuration Presets
        </CardTitle>
        <CardDescription>
          Save and load your favorite training configurations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Preset Selector */}
        <div className="space-y-2">
          <Label>Select Preset</Label>
          <div className="flex gap-2">
            <Select value={selectedPreset} onValueChange={setSelectedPreset}>
              <SelectTrigger className="flex-1">
                <SelectValue placeholder="Choose a preset..." />
              </SelectTrigger>
              <SelectContent>
                {builtins.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="font-semibold text-purple-400">Built-in Presets</SelectLabel>
                    {sortedModelTypeKeys(builtinGroups).map(type => (
                      <Fragment key={`builtin-${type}`}>
                        <SelectLabel className="text-xs text-muted-foreground pl-4">{type}</SelectLabel>
                        {builtinGroups.get(type)!.map(preset => (
                          <SelectItem key={preset.id} value={preset.id}>
                            {preset.name}
                          </SelectItem>
                        ))}
                      </Fragment>
                    ))}
                  </SelectGroup>
                )}

                {userServer.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="font-semibold text-blue-400">Your Server Presets</SelectLabel>
                    {sortedModelTypeKeys(userServerGroups).map(type => (
                      <Fragment key={`server-${type}`}>
                        <SelectLabel className="text-xs text-muted-foreground pl-4">{type}</SelectLabel>
                        {userServerGroups.get(type)!.map(preset => (
                          <SelectItem key={preset.id} value={preset.id}>
                            {preset.name}
                          </SelectItem>
                        ))}
                      </Fragment>
                    ))}
                  </SelectGroup>
                )}

                {customPresets.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="font-semibold text-cyan-400">Browser-Only Presets</SelectLabel>
                    {sortedModelTypeKeys(browserGroups).map(type => (
                      <Fragment key={`browser-${type}`}>
                        <SelectLabel className="text-xs text-muted-foreground pl-4">{type}</SelectLabel>
                        {browserGroups.get(type)!.map(preset => (
                          <SelectItem key={preset.id} value={preset.id}>
                            {preset.name}
                          </SelectItem>
                        ))}
                      </Fragment>
                    ))}
                  </SelectGroup>
                )}
              </SelectContent>
            </Select>

            <Button
              type="button"
              onClick={() => selectedPreset && handleLoadPreset(selectedPreset)}
              disabled={!selectedPreset}
              className="bg-purple-600 hover:bg-purple-700"
            >
              <FolderOpen className="h-4 w-4 mr-2" />
              Load
            </Button>
          </div>

          {selectedPreset && (
            <Alert className="bg-accent/50 border-border">
              <AlertDescription>
                {serverPresets.find((p) => p.id === selectedPreset)?.description ||
                  customPresets.find((p) => p.id === selectedPreset)?.description}
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-2">
          {/* Save Current Config */}
          <Dialog open={saveDialogOpen} onOpenChange={setSaveDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full" type="button">
                <Save className="h-4 w-4 mr-2" />
                Save Current
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Save Current Configuration</DialogTitle>
                <DialogDescription>
                  Save your current training settings as a reusable preset
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="preset-name">Preset Name</Label>
                  <Input
                    id="preset-name"
                    value={newPresetName}
                    onChange={(e) => setNewPresetName(e.target.value)}
                    placeholder="My Character Training"
                  />
                </div>
                <div>
                  <Label htmlFor="preset-description">Description (Optional)</Label>
                  <Input
                    id="preset-description"
                    value={newPresetDescription}
                    onChange={(e) => setNewPresetDescription(e.target.value)}
                    placeholder="Optimized settings for..."
                  />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={handleSavePreset} className="bg-green-600 hover:bg-green-700" type="button">
                  <Save className="h-4 w-4 mr-2" />
                  Save Preset
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Button
            variant="outline"
            type="button"
            onClick={() => selectedPreset && handleDeletePreset(selectedPreset)}
            disabled={
              !selectedPreset ||
              // Can't delete built-in server presets
              serverPresets.find((p) => p.id === selectedPreset && p.is_builtin) !== undefined ||
              // Can only delete localStorage presets or user server presets
              (!customPresets.find((p) => p.id === selectedPreset) &&
               !serverPresets.find((p) => p.id === selectedPreset && !p.is_builtin))
            }
            className="w-full hover:bg-red-500/20 hover:border-red-500/50"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>

          <Button
            variant="outline"
            type="button"
            onClick={handleExportPresets}
            disabled={customPresets.length === 0}
            className="w-full"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>

          <Button variant="outline" className="w-full relative" asChild>
            <label>
              <Upload className="h-4 w-4 mr-2" />
              Import
              <input
                type="file"
                accept=".json"
                onChange={handleImportPresets}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
            </label>
          </Button>
        </div>

        <div className="text-xs text-muted-foreground text-center">
          {loadingPresets ? (
            'Loading presets...'
          ) : (
            <>
              {serverPresets.filter(p => p.is_builtin).length > 0 && (
                <span className="text-green-400">
                  {serverPresets.filter(p => p.is_builtin).length} community preset{serverPresets.filter(p => p.is_builtin).length !== 1 ? 's' : ''}
                </span>
              )}
              {serverPresets.filter(p => !p.is_builtin).length > 0 && (
                <>
                  {serverPresets.filter(p => p.is_builtin).length > 0 && ' • '}
                  <span className="text-blue-400">
                    {serverPresets.filter(p => !p.is_builtin).length} user server preset{serverPresets.filter(p => !p.is_builtin).length !== 1 ? 's' : ''}
                  </span>
                </>
              )}
              {customPresets.length > 0 && (
                <>
                  {(serverPresets.filter(p => p.is_builtin).length > 0 || serverPresets.filter(p => !p.is_builtin).length > 0) && ' • '}
                  <span className="text-cyan-400">
                    {customPresets.length} browser preset{customPresets.length !== 1 ? 's' : ''}
                  </span>
                </>
              )}
              {serverPresets.length === 0 && customPresets.length === 0 && (
                'No custom presets saved yet'
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
