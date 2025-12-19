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
import type { TrainingConfig } from '@/lib/api';

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

    setNewPresetName('');
    setNewPresetDescription('');
    setSaveDialogOpen(false);

    alert(`✅ Preset "${newPresetName}" saved successfully!`);
  };

  const handleLoadPreset = (presetId: string) => {
    if (trainingPresets[presetId]) {
      onLoadPreset(trainingPresets[presetId].config);
      alert(`✅ Loaded preset: ${trainingPresets[presetId].name}`);
      return;
    }

    const customPreset = customPresets.find((p) => p.id === presetId);
    if (customPreset) {
      onLoadPreset(customPreset.config);
      alert(`✅ Loaded preset: ${customPreset.name}`);
      return;
    }
  };

  const handleDeletePreset = (presetId: string) => {
    if (!confirm('Are you sure you want to delete this preset?')) return;

    setCustomPresets((prev) => prev.filter((p) => p.id !== presetId));
    if (selectedPreset === presetId) {
      setSelectedPreset('');
    }

    alert('✅ Preset deleted');
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
              !customPresets.find((p) => p.id === selectedPreset)
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

        <div className="text-xs text-gray-500 text-center">
          {customPresets.length === 0
            ? 'No custom presets saved yet'
            : `${customPresets.length} custom preset${customPresets.length !== 1 ? 's' : ''} saved`}
        </div>
      </CardContent>
    </Card>
  );
}
