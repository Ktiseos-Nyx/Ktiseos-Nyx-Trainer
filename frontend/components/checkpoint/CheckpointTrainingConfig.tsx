'use client';

import { useState, useEffect } from 'react';
import { trainingAPI } from '@/lib/api';
import { Save, Play, FolderOpen, AlertCircle } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import FileBrowser from '@/components/FileBrowser';

interface CheckpointConfig {
  // Project & Model
  project_name: string;
  model_type: 'SD1.5' | 'SDXL' | 'Flux' | 'SD3' | 'Lumina';
  pretrained_model_name_or_path: string;
  vae_path?: string;

  // Dataset
  train_data_dir: string;
  output_dir: string;
  resolution: number;
  num_repeats: number;
  max_train_epochs: number;
  train_batch_size: number;
  seed: number;

  // Learning Rates
  unet_lr: number;
  text_encoder_lr: number;
  lr_scheduler: string;
  lr_warmup_ratio: number;

  // Optimizer
  optimizer_type: string;
  weight_decay: number;
  gradient_accumulation_steps: number;

  // Training Options
  flip_aug: boolean;
  shuffle_caption: boolean;
  gradient_checkpointing: boolean;
  mixed_precision: string;
  cache_latents: boolean;

  // Saving
  save_every_n_epochs: number;
  save_model_as: string;
  save_precision: string;
}

export default function CheckpointTrainingConfig() {
  const [config, setConfig] = useState<CheckpointConfig>({
    project_name: 'my_checkpoint',
    model_type: 'SDXL',
    pretrained_model_name_or_path: 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_path: '',
    train_data_dir: 'datasets/my_dataset',
    output_dir: 'output/checkpoints',
    resolution: 1024,
    num_repeats: 10,
    max_train_epochs: 10,
    train_batch_size: 2,
    seed: 42,
    unet_lr: 1e-5,
    text_encoder_lr: 5e-6,
    lr_scheduler: 'cosine',
    lr_warmup_ratio: 0.1,
    optimizer_type: 'AdamW8bit',
    weight_decay: 0.01,
    gradient_accumulation_steps: 2,
    flip_aug: false,
    shuffle_caption: true,
    gradient_checkpointing: true,
    mixed_precision: 'bf16',
    cache_latents: true,
    save_every_n_epochs: 2,
    save_model_as: 'safetensors',
    save_precision: 'fp16',
  });

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDatasetBrowser, setShowDatasetBrowser] = useState(false);
  const [showModelBrowser, setShowModelBrowser] = useState(false);

  const updateConfig = (key: keyof CheckpointConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleStartTraining = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Map to backend training config
      const trainingConfig = {
        ...config,
        training_mode: 'checkpoint',
        // Add checkpoint-specific parameters
        full_fp16: config.mixed_precision === 'fp16',
      };

      const result = await trainingAPI.start(trainingConfig as any);

      if (result.success) {
        const jobId = result.job_id || result.training_id;
        setSuccess(`Training started! Job ID: ${jobId}`);

        // Store job ID for monitor
        if (jobId) {
          localStorage.setItem('current_training_job_id', jobId);
        }

        // Dispatch event for monitor
        window.dispatchEvent(new CustomEvent('training-started', { detail: { jobId } }));
      } else {
        setError(result.error || 'Training failed to start');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Checkpoint Training Configuration</CardTitle>
        <CardDescription>
          Configure full model fine-tuning parameters (No LoRA adapters - trains entire model)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="learning">Learning</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="saving">Saving</TabsTrigger>
          </TabsList>

          {/* BASIC TAB */}
          <TabsContent value="basic" className="space-y-6">
            {/* Project Name */}
            <div className="space-y-2">
              <Label htmlFor="project_name">Project Name</Label>
              <Input
                id="project_name"
                value={config.project_name}
                onChange={(e) => updateConfig('project_name', e.target.value)}
                placeholder="my_checkpoint"
              />
            </div>

            {/* Model Type */}
            <div className="space-y-2">
              <Label htmlFor="model_type">Model Architecture</Label>
              <Select
                value={config.model_type}
                onValueChange={(value: any) => updateConfig('model_type', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="SD1.5">Stable Diffusion 1.5</SelectItem>
                  <SelectItem value="SDXL">Stable Diffusion XL</SelectItem>
                  <SelectItem value="Flux">Flux</SelectItem>
                  <SelectItem value="SD3">Stable Diffusion 3</SelectItem>
                  <SelectItem value="Lumina">Lumina</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Base Model Path */}
            <div className="space-y-2">
              <Label htmlFor="pretrained_model">Base Model Path</Label>
              <div className="flex gap-2">
                <Input
                  id="pretrained_model"
                  value={config.pretrained_model_name_or_path}
                  onChange={(e) => updateConfig('pretrained_model_name_or_path', e.target.value)}
                  placeholder="stabilityai/stable-diffusion-xl-base-1.0"
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setShowModelBrowser(true)}
                >
                  <FolderOpen className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Dataset Directory */}
            <div className="space-y-2">
              <Label htmlFor="train_data_dir">Dataset Directory</Label>
              <div className="flex gap-2">
                <Input
                  id="train_data_dir"
                  value={config.train_data_dir}
                  onChange={(e) => updateConfig('train_data_dir', e.target.value)}
                  placeholder="datasets/my_dataset"
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setShowDatasetBrowser(true)}
                >
                  <FolderOpen className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Output Directory */}
            <div className="space-y-2">
              <Label htmlFor="output_dir">Output Directory</Label>
              <Input
                id="output_dir"
                value={config.output_dir}
                onChange={(e) => updateConfig('output_dir', e.target.value)}
                placeholder="output/checkpoints"
              />
            </div>

            {/* Resolution & Batch Size */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="resolution">Resolution</Label>
                <Select
                  value={config.resolution.toString()}
                  onValueChange={(value) => updateConfig('resolution', parseInt(value))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="512">512 (SD1.5)</SelectItem>
                    <SelectItem value="1024">1024 (SDXL)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="train_batch_size">Batch Size</Label>
                <Input
                  id="train_batch_size"
                  type="number"
                  value={config.train_batch_size}
                  onChange={(e) => updateConfig('train_batch_size', parseInt(e.target.value))}
                  min={1}
                  max={8}
                />
                <p className="text-xs text-muted-foreground">
                  Checkpoint training uses much more VRAM. Keep low (1-2).
                </p>
              </div>
            </div>

            {/* Epochs & Repeats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="max_train_epochs">Epochs</Label>
                <Input
                  id="max_train_epochs"
                  type="number"
                  value={config.max_train_epochs}
                  onChange={(e) => updateConfig('max_train_epochs', parseInt(e.target.value))}
                  min={1}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="num_repeats">Dataset Repeats</Label>
                <Input
                  id="num_repeats"
                  type="number"
                  value={config.num_repeats}
                  onChange={(e) => updateConfig('num_repeats', parseInt(e.target.value))}
                  min={1}
                />
              </div>
            </div>
          </TabsContent>

          {/* LEARNING TAB */}
          <TabsContent value="learning" className="space-y-6">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Lower Learning Rates for Checkpoints</AlertTitle>
              <AlertDescription>
                Full model training uses much lower learning rates than LoRA (1e-5 to 1e-6 range)
              </AlertDescription>
            </Alert>

            {/* Learning Rates */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="unet_lr">UNet Learning Rate</Label>
                <Input
                  id="unet_lr"
                  type="number"
                  value={config.unet_lr}
                  onChange={(e) => updateConfig('unet_lr', parseFloat(e.target.value))}
                  step="0.000001"
                  placeholder="0.000001"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="text_encoder_lr">Text Encoder LR</Label>
                <Input
                  id="text_encoder_lr"
                  type="number"
                  value={config.text_encoder_lr}
                  onChange={(e) => updateConfig('text_encoder_lr', parseFloat(e.target.value))}
                  step="0.000001"
                  placeholder="0.000001"
                />
              </div>
            </div>

            {/* LR Scheduler */}
            <div className="space-y-2">
              <Label htmlFor="lr_scheduler">Learning Rate Scheduler</Label>
              <Select
                value={config.lr_scheduler}
                onValueChange={(value) => updateConfig('lr_scheduler', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="constant">Constant</SelectItem>
                  <SelectItem value="cosine">Cosine</SelectItem>
                  <SelectItem value="linear">Linear</SelectItem>
                  <SelectItem value="polynomial">Polynomial</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Warmup Ratio */}
            <div className="space-y-2">
              <Label htmlFor="lr_warmup_ratio">Warmup Ratio</Label>
              <Input
                id="lr_warmup_ratio"
                type="number"
                value={config.lr_warmup_ratio}
                onChange={(e) => updateConfig('lr_warmup_ratio', parseFloat(e.target.value))}
                step="0.01"
                min={0}
                max={1}
              />
            </div>

            {/* Optimizer */}
            <div className="space-y-2">
              <Label htmlFor="optimizer_type">Optimizer</Label>
              <Select
                value={config.optimizer_type}
                onValueChange={(value) => updateConfig('optimizer_type', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="AdamW">AdamW</SelectItem>
                  <SelectItem value="AdamW8bit">AdamW8bit (Memory Efficient)</SelectItem>
                  <SelectItem value="Lion">Lion</SelectItem>
                  <SelectItem value="SGDNesterov">SGD Nesterov</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="weight_decay">Weight Decay</Label>
                <Input
                  id="weight_decay"
                  type="number"
                  value={config.weight_decay}
                  onChange={(e) => updateConfig('weight_decay', parseFloat(e.target.value))}
                  step="0.01"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="gradient_accumulation_steps">Gradient Accumulation</Label>
                <Input
                  id="gradient_accumulation_steps"
                  type="number"
                  value={config.gradient_accumulation_steps}
                  onChange={(e) => updateConfig('gradient_accumulation_steps', parseInt(e.target.value))}
                  min={1}
                />
              </div>
            </div>
          </TabsContent>

          {/* ADVANCED TAB */}
          <TabsContent value="advanced" className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="gradient_checkpointing"
                  checked={config.gradient_checkpointing}
                  onCheckedChange={(checked) => updateConfig('gradient_checkpointing', checked)}
                />
                <Label htmlFor="gradient_checkpointing" className="cursor-pointer">
                  Gradient Checkpointing (Saves VRAM)
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="cache_latents"
                  checked={config.cache_latents}
                  onCheckedChange={(checked) => updateConfig('cache_latents', checked)}
                />
                <Label htmlFor="cache_latents" className="cursor-pointer">
                  Cache Latents (Faster training)
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="flip_aug"
                  checked={config.flip_aug}
                  onCheckedChange={(checked) => updateConfig('flip_aug', checked)}
                />
                <Label htmlFor="flip_aug" className="cursor-pointer">
                  Horizontal Flip Augmentation
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="shuffle_caption"
                  checked={config.shuffle_caption}
                  onCheckedChange={(checked) => updateConfig('shuffle_caption', checked)}
                />
                <Label htmlFor="shuffle_caption" className="cursor-pointer">
                  Shuffle Caption Tokens
                </Label>
              </div>
            </div>

            {/* Mixed Precision */}
            <div className="space-y-2">
              <Label htmlFor="mixed_precision">Mixed Precision</Label>
              <Select
                value={config.mixed_precision}
                onValueChange={(value) => updateConfig('mixed_precision', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="no">No (FP32)</SelectItem>
                  <SelectItem value="fp16">FP16</SelectItem>
                  <SelectItem value="bf16">BF16 (Recommended for A100/H100)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Seed */}
            <div className="space-y-2">
              <Label htmlFor="seed">Random Seed</Label>
              <Input
                id="seed"
                type="number"
                value={config.seed}
                onChange={(e) => updateConfig('seed', parseInt(e.target.value))}
              />
            </div>
          </TabsContent>

          {/* SAVING TAB */}
          <TabsContent value="saving" className="space-y-6">
            <Alert className="border-yellow-500/50 bg-yellow-500/10">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertTitle className="text-yellow-600 dark:text-yellow-400">Storage Warning</AlertTitle>
              <AlertDescription className="text-yellow-600/80 dark:text-yellow-400/80">
                Each checkpoint save is 2-7 GB. Be conservative with save frequency.
              </AlertDescription>
            </Alert>

            <div className="space-y-2">
              <Label htmlFor="save_every_n_epochs">Save Every N Epochs</Label>
              <Input
                id="save_every_n_epochs"
                type="number"
                value={config.save_every_n_epochs}
                onChange={(e) => updateConfig('save_every_n_epochs', parseInt(e.target.value))}
                min={0}
              />
              <p className="text-xs text-muted-foreground">
                0 = disabled. Recommended: 2-5 for checkpoint training.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="save_model_as">Save Format</Label>
                <Select
                  value={config.save_model_as}
                  onValueChange={(value) => updateConfig('save_model_as', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="safetensors">SafeTensors (Recommended)</SelectItem>
                    <SelectItem value="ckpt">Checkpoint (.ckpt)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_precision">Save Precision</Label>
                <Select
                  value={config.save_precision}
                  onValueChange={(value) => updateConfig('save_precision', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fp16">FP16 (Smaller)</SelectItem>
                    <SelectItem value="fp32">FP32 (Larger)</SelectItem>
                    <SelectItem value="bf16">BF16</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="mt-6 flex gap-4">
          <Button onClick={handleStartTraining} disabled={loading} className="flex-1">
            <Play className="w-4 h-4 mr-2" />
            {loading ? 'Starting Training...' : 'Start Checkpoint Training'}
          </Button>
        </div>

        {/* Success/Error Messages */}
        {success && (
          <Alert className="mt-4 border-green-500/50 bg-green-500/10">
            <AlertTitle className="text-green-600 dark:text-green-400">Success</AlertTitle>
            <AlertDescription className="text-green-600/80 dark:text-green-400/80">
              {success}
            </AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert className="mt-4 border-red-500/50 bg-red-500/10">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertTitle className="text-red-600 dark:text-red-400">Error</AlertTitle>
            <AlertDescription className="text-red-600/80 dark:text-red-400/80">
              {error}
            </AlertDescription>
          </Alert>
        )}

        {/* File Browsers */}
        {showDatasetBrowser && (
          <FileBrowser
            open={showDatasetBrowser}
            onOpenChange={setShowDatasetBrowser}
            onSelect={(path) => {
              updateConfig('train_data_dir', path);
              setShowDatasetBrowser(false);
            }}
            mode="directory"
            title="Select Dataset Directory"
          />
        )}

        {showModelBrowser && (
          <FileBrowser
            open={showModelBrowser}
            onOpenChange={setShowModelBrowser}
            onSelect={(path) => {
              updateConfig('pretrained_model_name_or_path', path);
              setShowModelBrowser(false);
            }}
            mode="file"
            title="Select Base Model"
          />
        )}
      </CardContent>
    </Card>
  );
}
