/* eslint-disable react-hooks/set-state-in-effect */
'use client';

import { useState, useEffect } from 'react';
import { Save, FolderOpen, Trash2, Download, Upload, Sparkles } from 'lucide-react';
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
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { trainingPresets } from '@/hooks/useTrainingForm';
import { presetsAPI, type TrainingConfig, type PresetMetadata } from '@/lib/api';

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
      alert('Please enter a preset name');
      return;
    }

    // Filter config to only include reusable hyperparameters
    const filteredConfig = filterPresetConfig(currentConfig);

    try {
      // Try saving to server first
      const result = await presetsAPI.save({
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

      alert(`✅ Preset "${newPresetName}" saved successfully!\n\nNote: Project-specific details (dataset paths, model paths, project name) are not included in presets.`);
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

      alert(`✅ Preset "${newPresetName}" saved locally!\n\nNote: Server unavailable, saved to browser storage only.`);
    }
  };

  const handleLoadPreset = async (presetId: string) => {
    // Check hardcoded presets first (from useTrainingForm hook)
    if (trainingPresets[presetId]) {
      onLoadPreset(trainingPresets[presetId].config);
      alert(`✅ Loaded preset: ${trainingPresets[presetId].name}\n\nReminder: You'll still need to set your dataset, model paths, and project name.`);
      return;
    }

    // Check localStorage presets
    const localPreset = customPresets.find((p) => p.id === presetId);
    if (localPreset) {
      onLoadPreset(localPreset.config);
      alert(`✅ Loaded preset: ${localPreset.name}\n\nReminder: You'll still need to set your dataset, model paths, and project name.`);
      return;
    }

    // Try loading from server
    try {
      const preset = await presetsAPI.get(presetId);

      // Server presets have config at root level (not nested under .config)
      // Extract metadata fields that aren't part of TrainingConfig
      const { name, description, notes, base_model, optimizer, is_builtin, created_at, ...config } = preset as any;

      onLoadPreset(config);
      alert(`✅ Loaded preset: ${name || presetId}\n\nReminder: You'll still need to set your dataset, model paths, and project name.`);
    } catch (error) {
      console.error('Failed to load preset from server:', error);
      alert('❌ Failed to load preset');
    }
  };

  const handleDeletePreset = async (presetId: string) => {
    if (!confirm('Are you sure you want to delete this preset?')) return;

    // Check if it's a server preset
    const serverPreset = serverPresets.find((p) => p.id === presetId);
    if (serverPreset) {
      if (serverPreset.is_builtin) {
        alert('❌ Cannot delete built-in presets');
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

        alert('✅ Preset deleted');
        return;
      } catch (error) {
        console.error('Failed to delete from server:', error);
        alert('❌ Failed to delete preset from server');
        return;
      }
    }

    // Delete from localStorage
    setCustomPresets((prev) => prev.filter((p) => p.id !== presetId));
    if (selectedPreset === presetId) {
      setSelectedPreset('');
    }

    alert('✅ Preset deleted from local storage');
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
        alert(`✅ Imported ${imported.length} presets`);
      } catch (error) {
        alert('❌ Failed to import presets. Invalid file format.');
      }
    };
    reader.readAsText(file);
  };

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
                <div className="px-2 py-1.5 text-xs font-semibold text-purple-400">
                  Built-in Presets
                </div>
                {Object.entries(trainingPresets).map(([key, preset]) => (
                  <SelectItem key={key} value={key}>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {preset.config.model_type}
                      </Badge>
                      {preset.name}
                    </div>
                  </SelectItem>
                ))}

                {serverPresets.length > 0 && (
                  <>
                    <div className="px-2 py-1.5 text-xs font-semibold text-green-400 mt-2">
                      Community Presets
                    </div>
                    {serverPresets.filter(p => p.is_builtin).map((preset) => (
                      <SelectItem key={preset.id} value={preset.id}>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {preset.model_type}
                          </Badge>
                          {preset.name}
                        </div>
                      </SelectItem>
                    ))}

                    {serverPresets.filter(p => !p.is_builtin).length > 0 && (
                      <>
                        <div className="px-2 py-1.5 text-xs font-semibold text-blue-400 mt-2">
                          Your Server Presets
                        </div>
                        {serverPresets.filter(p => !p.is_builtin).map((preset) => (
                          <SelectItem key={preset.id} value={preset.id}>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">
                                {preset.model_type}
                              </Badge>
                              {preset.name}
                            </div>
                          </SelectItem>
                        ))}
                      </>
                    )}
                  </>
                )}

                {customPresets.length > 0 && (
                  <>
                    <div className="px-2 py-1.5 text-xs font-semibold text-cyan-400 mt-2">
                      Browser-Only Presets
                    </div>
                    {customPresets.map((preset) => (
                      <SelectItem key={preset.id} value={preset.id}>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {preset.config.model_type}
                          </Badge>
                          {preset.name}
                        </div>
                      </SelectItem>
                    ))}
                  </>
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
                {trainingPresets[selectedPreset]?.description ||
                  serverPresets.find((p) => p.id === selectedPreset)?.description ||
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
              // Can't delete hardcoded presets or built-in server presets
              trainingPresets[selectedPreset] !== undefined ||
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
