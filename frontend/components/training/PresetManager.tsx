/**
 * Preset Manager Component
 * Allows users to save, load, and manage training configuration presets
 * Huge UX improvement - no more reconfiguring everything from scratch!
 */

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
import type { TrainingConfig } from '@/lib/api';

interface CustomPreset {
  id: string;
  name: string;
  description: string;
  config: Partial<TrainingConfig>;
  createdAt: number;
}

interface PresetManagerProps {
  /** Current configuration */
  currentConfig: Partial<TrainingConfig>;
  /** Callback when user loads a preset */
  onLoadPreset: (config: Partial<TrainingConfig>) => void;
  /** Callback when user saves current config */
  onSavePreset?: (preset: CustomPreset) => void;
}

export default function PresetManager({
  currentConfig,
  onLoadPreset,
  onSavePreset,
}: PresetManagerProps) {
  // Load custom presets from localStorage on mount
  const [customPresets, setCustomPresets] = useState<CustomPreset[]>(() => {
    const saved = localStorage.getItem('training-presets');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error('Failed to load custom presets:', e);
        return [];
      }
    }
    return [];
  });
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  // Save custom presets to localStorage
  useEffect(() => {
    if (customPresets.length > 0) {
      localStorage.setItem('training-presets', JSON.stringify(customPresets));
    }
  }, [customPresets]);

  /**
   * Save current configuration as a new preset
   */
  const handleSavePreset = () => {
    if (!newPresetName.trim()) {
      alert('Please enter a preset name');
      return;
    }

    const preset: CustomPreset = {
      id: `custom_${Date.now()}`,
      name: newPresetName,
      description: newPresetDescription || 'Custom preset',
      config: currentConfig,
      createdAt: Date.now(),
    };

    setCustomPresets((prev) => [...prev, preset]);
    onSavePreset?.(preset);

    // Reset dialog
    setNewPresetName('');
    setNewPresetDescription('');
    setSaveDialogOpen(false);

    alert(`✅ Preset "${newPresetName}" saved successfully!`);
  };

  /**
   * Load a built-in or custom preset
   */
  const handleLoadPreset = (presetId: string) => {
    // Check built-in presets first
    if (trainingPresets[presetId]) {
      onLoadPreset(trainingPresets[presetId].config);
      alert(`✅ Loaded preset: ${trainingPresets[presetId].name}`);
      return;
    }

    // Check custom presets
    const customPreset = customPresets.find((p) => p.id === presetId);
    if (customPreset) {
      onLoadPreset(customPreset.config);
      alert(`✅ Loaded preset: ${customPreset.name}`);
      return;
    }
  };

  /**
   * Delete a custom preset
   */
  const handleDeletePreset = (presetId: string) => {
    if (!confirm('Are you sure you want to delete this preset?')) return;

    setCustomPresets((prev) => prev.filter((p) => p.id !== presetId));
    if (selectedPreset === presetId) {
      setSelectedPreset('');
    }

    alert('✅ Preset deleted');
  };

  /**
   * Export all custom presets to JSON file
   */
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

  /**
   * Import presets from JSON file
   */
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
    <Card className="border-purple-500/30 bg-slate-900/50">
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
                {/* Built-in Presets */}
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

                {/* Custom Presets */}
                {customPresets.length > 0 && (
                  <>
                    <div className="px-2 py-1.5 text-xs font-semibold text-blue-400 mt-2">
                      Your Presets
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
              onClick={() => selectedPreset && handleLoadPreset(selectedPreset)}
              disabled={!selectedPreset}
              className="bg-purple-600 hover:bg-purple-700"
            >
              <FolderOpen className="h-4 w-4 mr-2" />
              Load
            </Button>
          </div>

          {/* Preset Description */}
          {selectedPreset && (
            <Alert className="bg-slate-800/50 border-slate-700">
              <AlertDescription>
                {trainingPresets[selectedPreset]?.description ||
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
              <Button variant="outline" className="w-full">
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
                <Button onClick={handleSavePreset} className="bg-green-600 hover:bg-green-700">
                  <Save className="h-4 w-4 mr-2" />
                  Save Preset
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Delete Custom Preset */}
          <Button
            variant="outline"
            onClick={() => selectedPreset && handleDeletePreset(selectedPreset)}
            disabled={
              !selectedPreset ||
              !customPresets.find((p) => p.id === selectedPreset)
            }
            className="w-full hover:bg-red-500/20 hover:border-red-500/50"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>

          {/* Export Presets */}
          <Button
            variant="outline"
            onClick={handleExportPresets}
            disabled={customPresets.length === 0}
            className="w-full"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>

          {/* Import Presets */}
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

        {/* Preset Count */}
        <div className="text-xs text-gray-500 text-center">
          {customPresets.length === 0
            ? 'No custom presets saved yet'
            : `${customPresets.length} custom preset${customPresets.length !== 1 ? 's' : ''} saved`}
        </div>
      </CardContent>
    </Card>
  );
}
