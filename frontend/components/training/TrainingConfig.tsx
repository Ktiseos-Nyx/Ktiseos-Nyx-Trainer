'use client';

import { useState, useEffect } from 'react';
import { configAPI, trainingAPI } from '@/lib/api';
import { Save, Play, FolderOpen, Wand2, Info, Settings } from 'lucide-react';

interface ConfigFormData {
  model_name: string;
  pretrained_model_name_or_path: string;
  output_dir: string;
  train_data_dir: string;
  resolution: number;
  train_batch_size: number;
  max_train_steps: number;
  max_train_epochs?: number;
  learning_rate: number;
  network_module: string;
  network_dim: number;
  network_alpha: number;
  gradient_checkpointing: boolean;
  mixed_precision: string;
  save_precision: string;
  optimizer_type: string;
  lr_scheduler: string;
  lr_warmup_steps: number;
}

export default function TrainingConfig() {
  const [config, setConfig] = useState<ConfigFormData>({
    model_name: 'my_lora',
    pretrained_model_name_or_path: 'runwayml/stable-diffusion-v1-5',
    output_dir: '/workspace/output/my_lora',
    train_data_dir: '/workspace/datasets/my_dataset',
    resolution: 512,
    train_batch_size: 1,
    max_train_steps: 1000,
    learning_rate: 1e-4,
    network_module: 'networks.lora',
    network_dim: 32,
    network_alpha: 32,
    gradient_checkpointing: true,
    mixed_precision: 'fp16',
    save_precision: 'fp16',
    optimizer_type: 'AdamW8bit',
    lr_scheduler: 'cosine',
    lr_warmup_steps: 100,
  });

  const [templates, setTemplates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Load config templates
  useEffect(() => {
    const loadTemplates = async () => {
      try {
        const data = await configAPI.templates();
        setTemplates(data.templates || []);
      } catch (err) {
        console.error('Failed to load templates:', err);
      }
    };
    loadTemplates();
  }, []);

  // Load default config
  useEffect(() => {
    const loadDefaults = async () => {
      try {
        const defaults = await configAPI.defaults();
        setConfig((prev) => ({ ...prev, ...defaults }));
      } catch (err) {
        console.error('Failed to load defaults:', err);
      }
    };
    loadDefaults();
  }, []);

  // Handle form input changes
  const handleChange = (field: keyof ConfigFormData, value: any) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  // Load template
  const loadTemplate = async (templatePath: string) => {
    try {
      setLoading(true);
      const data = await configAPI.load(templatePath);
      setConfig((prev) => ({ ...prev, ...data.config }));
      setSuccess('Template loaded!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(`Failed to load template: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Save config
  const handleSave = async () => {
    try {
      setLoading(true);
      setError(null);
      await configAPI.save(config.model_name, config);
      setSuccess('Configuration saved!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(`Failed to save: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Start training
  const handleStart = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await trainingAPI.start(config);
      setSuccess(`Training started! ID: ${result.training_id}`);
    } catch (err) {
      setError(`Failed to start training: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Training Configuration</h2>
        <p className="text-gray-600 dark:text-gray-400">
          Configure your LoRA training parameters (no race conditions! 🎉)
        </p>
      </div>

      {/* Alerts */}
      {error && (
        <div className="mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}
      {success && (
        <div className="mb-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          {success}
        </div>
      )}

      {/* Template Selector */}
      {templates.length > 0 && (
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <label className="block text-sm font-medium mb-2">
            Load Template
          </label>
          <select
            onChange={(e) => e.target.value && loadTemplate(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            defaultValue=""
          >
            <option value="">-- Select a template --</option>
            {templates.map((template) => (
              <option key={template.path} value={template.path}>
                {template.name} - {template.description}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Configuration Form */}
      <div className="space-y-6">
        {/* Basic Settings */}
        <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Basic Settings
          </h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Model Name *
              </label>
              <input
                type="text"
                value={config.model_name}
                onChange={(e) => handleChange('model_name', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="my_awesome_lora"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Base Model *
              </label>
              <select
                value={config.pretrained_model_name_or_path}
                onChange={(e) => handleChange('pretrained_model_name_or_path', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="runwayml/stable-diffusion-v1-5">SD 1.5</option>
                <option value="stabilityai/stable-diffusion-xl-base-1.0">SDXL Base</option>
                <option value="stabilityai/stable-diffusion-2-1">SD 2.1</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Dataset Path *
                <button
                  type="button"
                  className="ml-2 text-blue-500 hover:text-blue-700"
                  title="Browse files"
                >
                  <FolderOpen className="w-4 h-4 inline" />
                </button>
              </label>
              <input
                type="text"
                value={config.train_data_dir}
                onChange={(e) => handleChange('train_data_dir', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="/workspace/datasets/my_dataset"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Output Directory *
              </label>
              <input
                type="text"
                value={config.output_dir}
                onChange={(e) => handleChange('output_dir', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="/workspace/output/my_lora"
              />
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Wand2 className="w-5 h-5" />
            Training Parameters
          </h3>

          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Resolution
              </label>
              <select
                value={config.resolution}
                onChange={(e) => handleChange('resolution', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="512">512x512 (SD 1.5/2.x)</option>
                <option value="768">768x768</option>
                <option value="1024">1024x1024 (SDXL)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Batch Size
              </label>
              <input
                type="number"
                value={config.train_batch_size}
                onChange={(e) => handleChange('train_batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Max Steps
              </label>
              <input
                type="number"
                value={config.max_train_steps}
                onChange={(e) => handleChange('max_train_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                value={config.learning_rate}
                onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                step="0.00001"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Optimizer
              </label>
              <select
                value={config.optimizer_type}
                onChange={(e) => handleChange('optimizer_type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="AdamW8bit">AdamW 8bit</option>
                <option value="AdamW">AdamW</option>
                <option value="Prodigy">Prodigy</option>
                <option value="Lion">Lion</option>
                <option value="DAdaptation">DAdaptation</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                LR Scheduler
              </label>
              <select
                value={config.lr_scheduler}
                onChange={(e) => handleChange('lr_scheduler', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="cosine">Cosine</option>
                <option value="cosine_with_restarts">Cosine with Restarts</option>
                <option value="constant">Constant</option>
                <option value="linear">Linear</option>
                <option value="polynomial">Polynomial</option>
              </select>
            </div>
          </div>
        </div>

        {/* LoRA Settings */}
        <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
          <h3 className="text-lg font-semibold mb-4">LoRA Settings</h3>

          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Network Type
              </label>
              <select
                value={config.network_module}
                onChange={(e) => handleChange('network_module', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="networks.lora">LoRA</option>
                <option value="lycoris.kohya">LyCORIS</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Network Dim (Rank)
                <Info className="w-3 h-3 inline ml-1 text-gray-400" title="Higher = more capacity, larger file" />
              </label>
              <input
                type="number"
                value={config.network_dim}
                onChange={(e) => handleChange('network_dim', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Network Alpha
              </label>
              <input
                type="number"
                value={config.network_alpha}
                onChange={(e) => handleChange('network_alpha', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        <div>
          <h3 className="text-lg font-semibold mb-4">Advanced Settings</h3>

          <div className="space-y-3">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.gradient_checkpointing}
                onChange={(e) => handleChange('gradient_checkpointing', e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm">
                Gradient Checkpointing (saves VRAM)
              </span>
            </label>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Mixed Precision
                </label>
                <select
                  value={config.mixed_precision}
                  onChange={(e) => handleChange('mixed_precision', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="no">No (FP32)</option>
                  <option value="fp16">FP16</option>
                  <option value="bf16">BF16</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Save Precision
                </label>
                <select
                  value={config.save_precision}
                  onChange={(e) => handleChange('save_precision', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="fp16">FP16</option>
                  <option value="bf16">BF16</option>
                  <option value="float">FP32</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-8 flex gap-3">
        <button
          onClick={handleStart}
          disabled={loading}
          className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <Play className="w-5 h-5" />
          {loading ? 'Starting...' : 'Start Training'}
        </button>

        <button
          onClick={handleSave}
          disabled={loading}
          className="flex items-center gap-2 bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <Save className="w-5 h-5" />
          Save Config
        </button>
      </div>

      {/* Info Footer */}
      <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
        <p className="text-sm text-center">
          ✨ <strong>No more race conditions!</strong> Configuration state is properly managed with React.
          <br />
          🎉 Your settings are safe and won't randomly reset like Jupyter widgets!
        </p>
      </div>
    </div>
  );
}
