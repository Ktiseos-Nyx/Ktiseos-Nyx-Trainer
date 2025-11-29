'use client';

import { useState, useEffect } from 'react';
import { configAPI, trainingAPI, utilitiesAPI, TrainingConfig as TrainingConfigType, ValidationError } from '@/lib/api';
import { Save, Play, FolderOpen, AlertCircle, Calculator, AlertTriangle, Info } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import FileBrowser from '@/components/FileBrowser';
import { useSettings } from '@/hooks/useSettings';

export default function TrainingConfig() {
  // Load user settings for conditional rendering
  const settings = useSettings();
  const [config, setConfig] = useState<Partial<TrainingConfigType>>({
    // ========== PROJECT & MODEL SETUP ==========
    project_name: 'my_lora',
    model_type: 'SDXL',
    pretrained_model_name_or_path: 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_path: '',
    clip_l_path: '',
    clip_g_path: '',
    t5xxl_path: '',
    continue_from_lora: '',
    wandb_key: '',

    // ========== DATASET & BASIC TRAINING ==========
    train_data_dir: 'datasets/my_dataset',
    output_dir: 'output/my_lora',
    resolution: 1024,
    num_repeats: 10,
    max_train_epochs: 10,
    max_train_steps: 0,
    train_batch_size: 4,
    seed: 42,
    flip_aug: false,
    random_crop: false,
    color_aug: false,
    shuffle_caption: true,

    // ========== LEARNING RATES ==========
    unet_lr: 5e-4,
    text_encoder_lr: 1e-4,
    lr_scheduler: 'cosine',
    lr_scheduler_number: 3,
    lr_warmup_ratio: 0.05,
    lr_warmup_steps: 0,
    lr_power: 1.0,

    // ========== LORA STRUCTURE ==========
    lora_type: 'LoRA',
    network_module: 'networks.lora',
    network_dim: 16,
    network_alpha: 8,
    conv_dim: 16,
    conv_alpha: 8,
    network_dropout: 0.0,
    dim_from_weights: false,
    factor: -1,
    train_norm: false,
    rank_dropout: 0.0,
    module_dropout: 0.0,
    down_lr_weight: '',
    mid_lr_weight: '',
    up_lr_weight: '',
    block_lr_zero_threshold: '',
    block_dims: '',
    block_alphas: '',
    conv_block_dims: '',
    conv_block_alphas: '',

    // ========== OPTIMIZER ==========
    optimizer_type: 'AdamW8bit',
    weight_decay: 0.01,
    gradient_accumulation_steps: 1,
    max_grad_norm: 1.0,
    optimizer_args: '',

    // ========== CAPTION & TOKEN CONTROL ==========
    keep_tokens: 0,
    clip_skip: 2,
    max_token_length: 75,
    caption_dropout_rate: 0.0,
    caption_tag_dropout_rate: 0.0,
    caption_dropout_every_n_epochs: 0,
    keep_tokens_separator: '',
    secondary_separator: '',
    enable_wildcard: false,
    weighted_captions: false,

    // ========== BUCKETING ==========
    enable_bucket: true,
    sdxl_bucket_optimization: false,
    min_bucket_reso: 256,
    max_bucket_reso: 2048,
    bucket_no_upscale: false,

    // ========== ADVANCED TRAINING ==========
    min_snr_gamma_enabled: true,
    min_snr_gamma: 5.0,
    ip_noise_gamma_enabled: false,
    ip_noise_gamma: 0.05,
    multinoise: false,
    multires_noise_discount: 0.25,
    noise_offset: 0.0,
    adaptive_noise_scale: 0.0,
    zero_terminal_snr: false,

    // ========== MEMORY & PERFORMANCE ==========
    gradient_checkpointing: true,
    mixed_precision: 'fp16',
    full_fp16: false,
    fp8_base: false,
    vae_batch_size: 1,
    no_half_vae: false,
    cache_latents: true,
    cache_latents_to_disk: true,
    cache_text_encoder_outputs: false,
    cross_attention: 'sdpa',
    persistent_data_loader_workers: 0,
    no_token_padding: false,

    // ========== SAVING & CHECKPOINTS ==========
    save_every_n_epochs: 1,
    save_every_n_steps: 0,
    save_last_n_epochs: 0,
    save_last_n_epochs_state: 0,
    save_state: false,
    save_last_n_steps_state: 0,
    save_model_as: 'safetensors',
    save_precision: 'fp16',
    output_name: '',
    no_metadata: false,

    // ========== SAMPLE GENERATION ==========
    sample_every_n_epochs: 0,
    sample_every_n_steps: 0,
    sample_prompts: '',
    sample_sampler: 'euler_a',

    // ========== LOGGING ==========
    logging_dir: '',
    log_with: '',
    log_prefix: '',

    // ========== SD 2.x & ADVANCED ==========
    v2: false,
    v_parameterization: false,
    network_train_unet_only: false,
    prior_loss_weight: 1.0,

    // ========== FLUX-SPECIFIC PARAMETERS ==========
    ae_path: '',
    t5xxl_max_token_length: undefined,
    apply_t5_attn_mask: false,
    guidance_scale: 3.5,
    timestep_sampling: 'sigma',
    sigmoid_scale: 1.0,
    model_prediction_type: 'raw',
    blocks_to_swap: undefined,

    // ========== LUMINA-SPECIFIC PARAMETERS ==========
    gemma2: '',
    gemma2_max_token_length: undefined,
    // ae_path, timestep_sampling, sigmoid_scale, blocks_to_swap are shared with Flux
  });

  const [templates, setTemplates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);

  // File browser state
  const [browserOpen, setBrowserOpen] = useState<{
    field: keyof TrainingConfigType | null;
    mode: 'file' | 'directory';
    title: string;
    description: string;
  }>({
    field: null,
    mode: 'directory',
    title: '',
    description: '',
  });

  // Step calculator state
  const [stepCalc, setStepCalc] = useState<{
    images: number;
    repeats: number;
    total_steps: number;
    loading: boolean;
  }>({
    images: 0,
    repeats: 0,
    total_steps: 0,
    loading: false,
  });

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

  // Load default config (optional - component works without it)
  useEffect(() => {
    const loadDefaults = async () => {
      try {
        const defaults = await configAPI.defaults();
        setConfig((prev) => ({ ...prev, ...defaults }));
      } catch (err) {
        console.error('Failed to load defaults:', err);
        // Silently continue - component has built-in defaults already
      }
    };
    loadDefaults();
  }, []);

  // Validate config on change
  useEffect(() => {
    const validateConfig = async () => {
      try {
        const result = await trainingAPI.validate(config as TrainingConfigType);
        setWarnings(result.warnings || []);
      } catch (err) {
        console.error('Validation error:', err);
      }
    };

    // Debounce validation
    const timer = setTimeout(validateConfig, 500);
    return () => clearTimeout(timer);
  }, [config]);

  // Auto-calculate training steps when dataset/params change
  useEffect(() => {
    const calculateSteps = async () => {
      if (!config.train_data_dir) return;

      try {
        setStepCalc(prev => ({ ...prev, loading: true }));
        const result = await utilitiesAPI.calculateSteps({
          dataset_path: config.train_data_dir,
          epochs: config.max_train_epochs || 10,
          batch_size: config.train_batch_size || 4,
        });

        setStepCalc({
          images: result.images,
          repeats: result.repeats,
          total_steps: result.total_steps,
          loading: false,
        });
      } catch (err) {
        console.error('Step calculation error:', err);
        setStepCalc({ images: 0, repeats: 0, total_steps: 0, loading: false });
      }
    };

    // Debounce calculation
    const timer = setTimeout(calculateSteps, 500);
    return () => clearTimeout(timer);
  }, [config.train_data_dir, config.max_train_epochs, config.train_batch_size, config.num_repeats]);

  // Handle form input changes
  const handleChange = (field: keyof TrainingConfigType, value: any) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  // Open file browser
  const openBrowser = (
    field: keyof TrainingConfigType,
    mode: 'file' | 'directory',
    title: string,
    description: string
  ) => {
    setBrowserOpen({ field, mode, title, description });
  };

  // Handle file browser selection
  const handleBrowserSelect = (path: string) => {
    if (browserOpen.field) {
      handleChange(browserOpen.field, path);
    }
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
      await configAPI.save(config.project_name || 'my_lora', config);
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
      setSuccess(null);
      setValidationErrors([]);
      setWarnings([]);

      const result = await trainingAPI.start(config as TrainingConfigType);

      // Handle validation errors from backend
      if (result.validation_errors && result.validation_errors.length > 0) {
        setValidationErrors(result.validation_errors);

        // If there are critical errors, don't show success
        const hasErrors = result.validation_errors.some((e: ValidationError) => e.severity === 'error');
        if (hasErrors) {
          setError(result.message || 'Configuration has errors. Please fix them before starting training.');
          return;
        }
      }

      // Training started successfully
      if (result.success) {
        setSuccess(`Training started! ID: ${result.training_id}`);

        // Notify TrainingMonitor to start polling
        window.dispatchEvent(new CustomEvent('training-started'));
      } else {
        setError(result.message || 'Failed to start training');
      }

      // Backwards compatibility with old warnings format
      if (result.warnings && result.warnings.length > 0) {
        setWarnings(result.warnings);
      }
    } catch (err) {
      setError(`Failed to start training: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Training Configuration</CardTitle>
        <CardDescription>Configure your LoRA training parameters</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Alerts */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert>
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        {/* Validation Errors - New Structured Format */}
        {validationErrors.length > 0 && (
          <div className="space-y-2">
            {validationErrors.filter(e => e.severity === 'error').length > 0 && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Configuration Errors</AlertTitle>
                <AlertDescription>
                  <ul className="list-none space-y-2 mt-2">
                    {validationErrors.filter(e => e.severity === 'error').map((err, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-mono text-xs bg-red-100 text-red-800 px-2 py-1 rounded">
                          {err.field}
                        </span>
                        <span className="flex-1">{err.message}</span>
                      </li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {validationErrors.filter(e => e.severity === 'warning').length > 0 && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Configuration Warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-none space-y-2 mt-2">
                    {validationErrors.filter(e => e.severity === 'warning').map((err, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-mono text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                          {err.field}
                        </span>
                        <span className="flex-1">{err.message}</span>
                      </li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {validationErrors.filter(e => e.severity === 'info').length > 0 && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Helpful Tips</AlertTitle>
                <AlertDescription>
                  <ul className="list-none space-y-2 mt-2">
                    {validationErrors.filter(e => e.severity === 'info').map((err, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-mono text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          {err.field}
                        </span>
                        <span className="flex-1">{err.message}</span>
                      </li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {/* Template Selector */}
        {templates.length > 0 && (
          <div className="space-y-2">
            <Label>Load Template</Label>
            <Select onValueChange={loadTemplate}>
              <SelectTrigger>
                <SelectValue placeholder="Select a template" />
              </SelectTrigger>
              <SelectContent>
                {templates.map((template) => (
                  <SelectItem key={template.path} value={template.path}>
                    {template.name} - {template.description}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Tabs */}
        <Tabs defaultValue="setup" className="w-full">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="setup">Setup</TabsTrigger>
            <TabsTrigger value="dataset">Dataset</TabsTrigger>
            <TabsTrigger value="lora">LoRA</TabsTrigger>
            <TabsTrigger value="learning">Learning Rate</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="saving">Saving</TabsTrigger>
          </TabsList>

          {/* Tab 1: Setup */}
          <TabsContent value="setup" className="space-y-4">
            <h3 className="text-lg font-semibold">Project & Model Setup</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="project_name">Project Name *</Label>
                <Input
                  id="project_name"
                  value={config.project_name}
                  onChange={(e) => handleChange('project_name', e.target.value)}
                  placeholder="my_awesome_lora"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="model_type">Model Type *</Label>
                <Select value={config.model_type} onValueChange={(val) => handleChange('model_type', val)}>
                  <SelectTrigger id="model_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SD1.5">SD 1.5</SelectItem>
                    <SelectItem value="SDXL">SDXL</SelectItem>
                    <SelectItem value="Flux">Flux (24GB VRAM)</SelectItem>
                    <SelectItem value="SD3">SD3 (24GB VRAM)</SelectItem>
                    <SelectItem value="Lumina">Lumina (Experimental)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="md:col-span-2 space-y-2">
                <Label htmlFor="base_model">Base Model Path *</Label>
                <div className="flex gap-2">
                  <Input
                    id="base_model"
                    value={config.pretrained_model_name_or_path}
                    onChange={(e) => handleChange('pretrained_model_name_or_path', e.target.value)}
                    placeholder="stabilityai/stable-diffusion-xl-base-1.0 or /path/to/model"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      openBrowser(
                        'pretrained_model_name_or_path',
                        'file',
                        'Select Base Model',
                        'Choose a model file (.safetensors or .ckpt) or enter a HuggingFace model ID'
                      )
                    }
                  >
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="vae_path">VAE Path (optional)</Label>
                <div className="flex gap-2">
                  <Input
                    id="vae_path"
                    value={config.vae_path}
                    onChange={(e) => handleChange('vae_path', e.target.value)}
                    placeholder="/path/to/vae.safetensors"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      openBrowser(
                        'vae_path',
                        'file',
                        'Select VAE File',
                        'Choose a VAE file (.safetensors or .ckpt)'
                      )
                    }
                  >
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="continue_from_lora">Continue from LoRA (optional)</Label>
                <div className="flex gap-2">
                  <Input
                    id="continue_from_lora"
                    value={config.continue_from_lora}
                    onChange={(e) => handleChange('continue_from_lora', e.target.value)}
                    placeholder="/path/to/existing_lora.safetensors"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      openBrowser(
                        'continue_from_lora',
                        'file',
                        'Select Existing LoRA',
                        'Choose a LoRA file to continue training from'
                      )
                    }
                  >
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Conditional: Flux/SD3 Paths */}
            {(config.model_type === 'Flux' || config.model_type === 'SD3') && (
              <div className="border-t pt-4 space-y-4">
                <h4 className="text-md font-medium">Flux/SD3 Text Encoder Paths</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="clip_l_path">CLIP-L Path</Label>
                    <div className="flex gap-2">
                      <Input
                        id="clip_l_path"
                        value={config.clip_l_path}
                        onChange={(e) => handleChange('clip_l_path', e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() =>
                          openBrowser(
                            'clip_l_path',
                            'file',
                            'Select CLIP-L File',
                            'Choose the CLIP-L text encoder file'
                          )
                        }
                      >
                        <FolderOpen className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="clip_g_path">CLIP-G Path</Label>
                    <div className="flex gap-2">
                      <Input
                        id="clip_g_path"
                        value={config.clip_g_path}
                        onChange={(e) => handleChange('clip_g_path', e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() =>
                          openBrowser(
                            'clip_g_path',
                            'file',
                            'Select CLIP-G File',
                            'Choose the CLIP-G text encoder file'
                          )
                        }
                      >
                        <FolderOpen className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="md:col-span-2 space-y-2">
                    <Label htmlFor="t5xxl_path">T5XXL Path</Label>
                    <div className="flex gap-2">
                      <Input
                        id="t5xxl_path"
                        value={config.t5xxl_path}
                        onChange={(e) => handleChange('t5xxl_path', e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() =>
                          openBrowser(
                            't5xxl_path',
                            'file',
                            'Select T5XXL File',
                            'Choose the T5XXL text encoder file'
                          )
                        }
                      >
                        <FolderOpen className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Conditional: Flux-Specific Parameters */}
            {config.model_type === 'Flux' && (
              <div className="border-t pt-4 space-y-4">
                <h4 className="text-md font-medium text-purple-400">Flux-Specific Parameters</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="ae_path">AutoEncoder Path</Label>
                    <div className="flex gap-2">
                      <Input
                        id="ae_path"
                        value={config.ae_path}
                        onChange={(e) => handleChange('ae_path', e.target.value)}
                        placeholder="/path/to/ae.safetensors"
                        className="flex-1"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() =>
                          openBrowser(
                            'ae_path',
                            'file',
                            'Select Flux AutoEncoder',
                            'Choose the Flux AE file (.safetensors)'
                          )
                        }
                      >
                        <FolderOpen className="w-4 h-4" />
                      </Button>
                    </div>
                    <p className="text-xs text-gray-400">Optional: Flux AutoEncoder model</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="guidance_scale">Guidance Scale</Label>
                    <Input
                      id="guidance_scale"
                      type="number"
                      value={config.guidance_scale}
                      onChange={(e) => handleChange('guidance_scale', parseFloat(e.target.value))}
                      step="0.1"
                      min="0"
                      max="20"
                    />
                    <p className="text-xs text-gray-400">Recommended: 3.5 for Flux.1 dev</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="timestep_sampling">Timestep Sampling</Label>
                    <Select value={config.timestep_sampling} onValueChange={(val) => handleChange('timestep_sampling', val)}>
                      <SelectTrigger id="timestep_sampling">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sigma">Sigma (Default)</SelectItem>
                        <SelectItem value="uniform">Uniform Random</SelectItem>
                        <SelectItem value="sigmoid">Sigmoid</SelectItem>
                        <SelectItem value="shift">Shift</SelectItem>
                        <SelectItem value="flux_shift">Flux Shift</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="model_prediction_type">Model Prediction Type</Label>
                    <Select value={config.model_prediction_type} onValueChange={(val) => handleChange('model_prediction_type', val)}>
                      <SelectTrigger id="model_prediction_type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="raw">Raw</SelectItem>
                        <SelectItem value="additive">Additive (for dev model)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="blocks_to_swap">Blocks to Swap (Memory Optimization)</Label>
                    <Input
                      id="blocks_to_swap"
                      type="number"
                      value={config.blocks_to_swap || ''}
                      onChange={(e) => handleChange('blocks_to_swap', e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="18 (recommended for 24GB VRAM)"
                      min="0"
                      max="38"
                    />
                    <p className="text-xs text-gray-400">Leave empty for auto, or set to 18 for GPUs with &lt;48GB VRAM</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="t5xxl_max_token_length">T5XXL Max Token Length</Label>
                    <Input
                      id="t5xxl_max_token_length"
                      type="number"
                      value={config.t5xxl_max_token_length || ''}
                      onChange={(e) => handleChange('t5xxl_max_token_length', e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="256 (schnell) or 512 (dev)"
                      min="128"
                      max="1024"
                    />
                  </div>

                  <div className="space-y-2 md:col-span-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="apply_t5_attn_mask"
                        checked={config.apply_t5_attn_mask}
                        onCheckedChange={(checked) => handleChange('apply_t5_attn_mask', checked)}
                      />
                      <Label htmlFor="apply_t5_attn_mask" className="font-normal">Apply T5-XXL Attention Mask</Label>
                    </div>
                    <p className="text-xs text-gray-400">Apply attention mask to T5-XXL encoder and FLUX double blocks</p>
                  </div>
                </div>
              </div>
            )}

            {config.model_type === 'Lumina' && (
              <div className="border-t pt-4 space-y-4">
                <h4 className="text-md font-medium text-purple-400">Lumina-Specific Parameters</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="gemma2">Gemma2 Model Path *</Label>
                    <Input
                      id="gemma2"
                      value={config.gemma2}
                      onChange={(e) => handleChange('gemma2', e.target.value)}
                      placeholder="/path/to/gemma2.safetensors or .sft"
                      required
                    />
                    <p className="text-xs text-gray-400">Path to Gemma2 model (*.sft or *.safetensors), should be float16</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="ae_path">AutoEncoder Path *</Label>
                    <Input
                      id="ae_path"
                      value={config.ae_path}
                      onChange={(e) => handleChange('ae_path', e.target.value)}
                      placeholder="/path/to/ae.safetensors or .sft"
                      required
                    />
                    <p className="text-xs text-gray-400">Path to AutoEncoder (shared with Flux)</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="gemma2_max_token_length">Gemma2 Max Token Length</Label>
                    <Input
                      id="gemma2_max_token_length"
                      type="number"
                      value={config.gemma2_max_token_length || ''}
                      onChange={(e) => handleChange('gemma2_max_token_length', e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="256 (default)"
                    />
                    <p className="text-xs text-gray-400">Maximum token length for Gemma2. Default: 256</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="timestep_sampling">Timestep Sampling</Label>
                    <Select value={config.timestep_sampling} onValueChange={(value) => handleChange('timestep_sampling', value)}>
                      <SelectTrigger id="timestep_sampling">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="shift">shift (default)</SelectItem>
                        <SelectItem value="nextdit_shift">nextdit_shift</SelectItem>
                        <SelectItem value="sigma">sigma</SelectItem>
                        <SelectItem value="uniform">uniform</SelectItem>
                        <SelectItem value="sigmoid">sigmoid</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-400">Method to sample timesteps during training</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="sigmoid_scale">Sigmoid Scale</Label>
                    <Input
                      id="sigmoid_scale"
                      type="number"
                      step="0.1"
                      value={config.sigmoid_scale}
                      onChange={(e) => handleChange('sigmoid_scale', parseFloat(e.target.value) || 1.0)}
                    />
                    <p className="text-xs text-gray-400">Scale for sigmoid timestep sampling (only used when timestep_sampling is "sigmoid")</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="blocks_to_swap">Blocks to Swap (Memory Optimization)</Label>
                    <Input
                      id="blocks_to_swap"
                      type="number"
                      value={config.blocks_to_swap || ''}
                      onChange={(e) => handleChange('blocks_to_swap', e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="Leave empty for no swapping"
                    />
                    <p className="text-xs text-gray-400">Number of blocks to swap to reduce VRAM usage. Recommended for GPUs with &lt;48GB VRAM</p>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="wandb_key">WandB API Key (optional)</Label>
              <Input
                id="wandb_key"
                type="password"
                value={config.wandb_key}
                onChange={(e) => handleChange('wandb_key', e.target.value)}
                placeholder="Your WandB API key for logging"
              />
            </div>
          </TabsContent>

          {/* Tab 2: Dataset */}
          <TabsContent value="dataset" className="space-y-4">
            <h3 className="text-lg font-semibold">Dataset & Training</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="md:col-span-2 space-y-2">
                <Label htmlFor="train_data_dir">Dataset Directory *</Label>
                <div className="flex gap-2">
                  <Input
                    id="train_data_dir"
                    value={config.train_data_dir}
                    onChange={(e) => handleChange('train_data_dir', e.target.value)}
                    placeholder="/workspace/datasets/my_dataset"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      openBrowser(
                        'train_data_dir',
                        'directory',
                        'Select Dataset Directory',
                        'Choose the directory containing your training images'
                      )
                    }
                  >
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="md:col-span-2 space-y-2">
                <Label htmlFor="output_dir">Output Directory *</Label>
                <div className="flex gap-2">
                  <Input
                    id="output_dir"
                    value={config.output_dir}
                    onChange={(e) => handleChange('output_dir', e.target.value)}
                    placeholder="/workspace/output/my_lora"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      openBrowser(
                        'output_dir',
                        'directory',
                        'Select Output Directory',
                        'Choose where to save your trained LoRA models'
                      )
                    }
                  >
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="resolution">Resolution</Label>
                <Select value={String(config.resolution)} onValueChange={(val) => handleChange('resolution', parseInt(val))}>
                  <SelectTrigger id="resolution">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="512">512 (SD 1.5)</SelectItem>
                    <SelectItem value="768">768</SelectItem>
                    <SelectItem value="1024">1024 (SDXL)</SelectItem>
                    <SelectItem value="2048">2048</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="num_repeats">Num Repeats (CRITICAL for Kohya!)</Label>
                <Input
                  id="num_repeats"
                  type="number"
                  value={config.num_repeats}
                  onChange={(e) => handleChange('num_repeats', parseInt(e.target.value))}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="max_train_epochs">Max Train Epochs</Label>
                <Input
                  id="max_train_epochs"
                  type="number"
                  value={config.max_train_epochs}
                  onChange={(e) => handleChange('max_train_epochs', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="max_train_steps">Max Train Steps (0 = use epochs)</Label>
                <Input
                  id="max_train_steps"
                  type="number"
                  value={config.max_train_steps}
                  onChange={(e) => handleChange('max_train_steps', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="train_batch_size">Batch Size</Label>
                <Input
                  id="train_batch_size"
                  type="number"
                  value={config.train_batch_size}
                  onChange={(e) => handleChange('train_batch_size', parseInt(e.target.value))}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="seed">Seed</Label>
                <Input
                  id="seed"
                  type="number"
                  value={config.seed}
                  onChange={(e) => handleChange('seed', parseInt(e.target.value))}
                />
              </div>
            </div>

            {/* Step Calculator Widget */}
            <div className="border-t pt-4">
              <div className="bg-gradient-to-br from-cyan-500/10 to-purple-500/10 border-2 border-cyan-400/30 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Calculator className="w-6 h-6 text-cyan-400" />
                  <h4 className="text-lg font-bold text-white">Training Steps Calculator</h4>
                </div>

                {stepCalc.loading ? (
                  <div className="text-center text-gray-400 py-4">
                    Calculating steps...
                  </div>
                ) : stepCalc.total_steps > 0 ? (
                  <div className="space-y-3">
                    <div className="text-center">
                      <div className="text-4xl font-bold text-cyan-400 mb-2">
                        {stepCalc.total_steps.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-400">Total Training Steps</div>
                    </div>

                    <div className="bg-slate-800/50 rounded-lg p-4 font-mono text-sm text-center">
                      <div className="text-gray-300">
                        {stepCalc.images} images × {config.num_repeats || stepCalc.repeats} repeats × {config.max_train_epochs || 10} epochs
                      </div>
                      <div className="text-gray-400 text-xs mt-1">
                        ÷ {config.train_batch_size || 4} batch size = {stepCalc.total_steps} steps
                      </div>
                    </div>

                    <div className="text-xs text-gray-500 text-center">
                      💡 This calculation updates automatically when you change dataset, epochs, or batch size
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-4">
                    Enter a dataset directory to calculate training steps
                  </div>
                )}
              </div>
            </div>

            <div className="border-t pt-4 space-y-3">
              <h4 className="text-md font-medium">Data Augmentation</h4>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="flip_aug"
                  checked={config.flip_aug}
                  onCheckedChange={(checked) => handleChange('flip_aug', checked)}
                />
                <Label htmlFor="flip_aug" className="font-normal">Flip Augmentation (horizontal flip)</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="random_crop"
                  checked={config.random_crop}
                  onCheckedChange={(checked) => handleChange('random_crop', checked)}
                />
                <Label htmlFor="random_crop" className="font-normal">Random Crop</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="color_aug"
                  checked={config.color_aug}
                  onCheckedChange={(checked) => handleChange('color_aug', checked)}
                />
                <Label htmlFor="color_aug" className="font-normal">Color Augmentation</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="shuffle_caption"
                  checked={config.shuffle_caption}
                  onCheckedChange={(checked) => handleChange('shuffle_caption', checked)}
                />
                <Label htmlFor="shuffle_caption" className="font-normal">Shuffle Caption Tags</Label>
              </div>
            </div>

            <div className="border-t pt-4 space-y-3">
              <h4 className="text-md font-medium">Bucketing</h4>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="enable_bucket"
                  checked={config.enable_bucket}
                  onCheckedChange={(checked) => handleChange('enable_bucket', checked)}
                />
                <Label htmlFor="enable_bucket" className="font-normal">Enable Bucketing</Label>
              </div>

              {config.enable_bucket && (
                <div className="grid md:grid-cols-2 gap-4 ml-6">
                  <div className="space-y-2">
                    <Label htmlFor="min_bucket_reso">Min Bucket Resolution</Label>
                    <Input
                      id="min_bucket_reso"
                      type="number"
                      value={config.min_bucket_reso}
                      onChange={(e) => handleChange('min_bucket_reso', parseInt(e.target.value))}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="max_bucket_reso">Max Bucket Resolution</Label>
                    <Input
                      id="max_bucket_reso"
                      type="number"
                      value={config.max_bucket_reso}
                      onChange={(e) => handleChange('max_bucket_reso', parseInt(e.target.value))}
                    />
                  </div>

                  <div className="flex items-center space-x-2 md:col-span-2">
                    <Checkbox
                      id="sdxl_bucket_optimization"
                      checked={config.sdxl_bucket_optimization}
                      onCheckedChange={(checked) => handleChange('sdxl_bucket_optimization', checked)}
                    />
                    <Label htmlFor="sdxl_bucket_optimization" className="font-normal">SDXL Bucket Optimization</Label>
                  </div>

                  <div className="flex items-center space-x-2 md:col-span-2">
                    <Checkbox
                      id="bucket_no_upscale"
                      checked={config.bucket_no_upscale}
                      onCheckedChange={(checked) => handleChange('bucket_no_upscale', checked)}
                    />
                    <Label htmlFor="bucket_no_upscale" className="font-normal">No Upscale (don't upscale smaller images)</Label>
                  </div>
                </div>
              )}
            </div>

            <div className="border-t pt-4 space-y-3">
              <h4 className="text-md font-medium">Caption & Token Control</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="keep_tokens">Keep Tokens</Label>
                  <Input
                    id="keep_tokens"
                    type="number"
                    value={config.keep_tokens}
                    onChange={(e) => handleChange('keep_tokens', parseInt(e.target.value))}
                    min="0"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="clip_skip">CLIP Skip</Label>
                  <Input
                    id="clip_skip"
                    type="number"
                    value={config.clip_skip}
                    onChange={(e) => handleChange('clip_skip', parseInt(e.target.value))}
                    min="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max_token_length">Max Token Length</Label>
                  <Input
                    id="max_token_length"
                    type="number"
                    value={config.max_token_length}
                    onChange={(e) => handleChange('max_token_length', parseInt(e.target.value))}
                    min="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="caption_dropout_rate">Caption Dropout Rate</Label>
                  <Input
                    id="caption_dropout_rate"
                    type="number"
                    value={config.caption_dropout_rate}
                    onChange={(e) => handleChange('caption_dropout_rate', parseFloat(e.target.value))}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="caption_tag_dropout_rate">Caption Tag Dropout Rate</Label>
                  <Input
                    id="caption_tag_dropout_rate"
                    type="number"
                    value={config.caption_tag_dropout_rate}
                    onChange={(e) => handleChange('caption_tag_dropout_rate', parseFloat(e.target.value))}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="caption_dropout_every_n_epochs">Caption Dropout Every N Epochs</Label>
                  <Input
                    id="caption_dropout_every_n_epochs"
                    type="number"
                    value={config.caption_dropout_every_n_epochs}
                    onChange={(e) => handleChange('caption_dropout_every_n_epochs', parseInt(e.target.value))}
                    min="0"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="enable_wildcard"
                    checked={config.enable_wildcard}
                    onCheckedChange={(checked) => handleChange('enable_wildcard', checked)}
                  />
                  <Label htmlFor="enable_wildcard" className="font-normal">Enable Wildcard</Label>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="keep_tokens_separator">Keep Tokens Separator</Label>
                  <Input
                    id="keep_tokens_separator"
                    value={config.keep_tokens_separator}
                    onChange={(e) => handleChange('keep_tokens_separator', e.target.value)}
                    placeholder="|||"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="secondary_separator">Secondary Separator</Label>
                  <Input
                    id="secondary_separator"
                    value={config.secondary_separator}
                    onChange={(e) => handleChange('secondary_separator', e.target.value)}
                    placeholder=";;"
                  />
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Tab 3: LoRA */}
          <TabsContent value="lora" className="space-y-4">
            <h3 className="text-lg font-semibold">LoRA Structure</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="lora_type">LoRA Type</Label>
                <Select value={config.lora_type} onValueChange={(val) => handleChange('lora_type', val)}>
                  <SelectTrigger id="lora_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="LoRA">LoRA</SelectItem>
                    <SelectItem value="LoCon">LoCon</SelectItem>
                    <SelectItem value="LoKR">LoKR</SelectItem>
                    <SelectItem value="DyLoRA">DyLoRA</SelectItem>
                    <SelectItem value="DoRA">DoRA</SelectItem>
                    <SelectItem value="LoHa">LoHa</SelectItem>
                    <SelectItem value="(IA)³">(IA)³</SelectItem>
                    <SelectItem value="GLoRA">GLoRA</SelectItem>
                    <SelectItem value="Native Fine-Tuning">Native Fine-Tuning</SelectItem>
                    <SelectItem value="Diag-OFT">Diag-OFT</SelectItem>
                    <SelectItem value="BOFT">BOFT</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="network_module">Network Module</Label>
                <Select value={config.network_module} onValueChange={(val) => handleChange('network_module', val)}>
                  <SelectTrigger id="network_module">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="networks.lora">networks.lora</SelectItem>
                    <SelectItem value="lycoris.kohya">lycoris.kohya</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="network_dim">Network Dim (Rank)</Label>
                <Input
                  id="network_dim"
                  type="number"
                  value={config.network_dim}
                  onChange={(e) => handleChange('network_dim', parseInt(e.target.value))}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="network_alpha">Network Alpha</Label>
                <Input
                  id="network_alpha"
                  type="number"
                  value={config.network_alpha}
                  onChange={(e) => handleChange('network_alpha', parseInt(e.target.value))}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="conv_dim">Conv Dim (for textures/details)</Label>
                <Input
                  id="conv_dim"
                  type="number"
                  value={config.conv_dim}
                  onChange={(e) => handleChange('conv_dim', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="conv_alpha">Conv Alpha</Label>
                <Input
                  id="conv_alpha"
                  type="number"
                  value={config.conv_alpha}
                  onChange={(e) => handleChange('conv_alpha', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="network_dropout">Network Dropout</Label>
                <Input
                  id="network_dropout"
                  type="number"
                  value={config.network_dropout}
                  onChange={(e) => handleChange('network_dropout', parseFloat(e.target.value))}
                  step="0.01"
                  min="0"
                  max="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="factor">Factor (LoKR decomposition, -1=auto)</Label>
                <Input
                  id="factor"
                  type="number"
                  value={config.factor}
                  onChange={(e) => handleChange('factor', parseInt(e.target.value))}
                />
              </div>

              <div className="md:col-span-2 space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="dim_from_weights"
                    checked={config.dim_from_weights}
                    onCheckedChange={(checked) => handleChange('dim_from_weights', checked)}
                  />
                  <Label htmlFor="dim_from_weights" className="font-normal">Dimension from Weights</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="train_norm"
                    checked={config.train_norm}
                    onCheckedChange={(checked) => handleChange('train_norm', checked)}
                  />
                  <Label htmlFor="train_norm" className="font-normal">Train Norm (LyCORIS)</Label>
                </div>
              </div>

              {/* Advanced LyCORIS Parameters - Conditional */}
              {settings.showAdvancedLycoris && (
                <div className="border-t pt-4 space-y-3">
                  <h4 className="text-md font-medium text-purple-400">Advanced LyCORIS Parameters</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="rank_dropout">Rank Dropout</Label>
                      <Input
                        id="rank_dropout"
                        type="number"
                        value={config.rank_dropout}
                        onChange={(e) => handleChange('rank_dropout', parseFloat(e.target.value))}
                        step="0.05"
                        min="0"
                        max="1"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="module_dropout">Module Dropout</Label>
                      <Input
                        id="module_dropout"
                        type="number"
                        value={config.module_dropout}
                        onChange={(e) => handleChange('module_dropout', parseFloat(e.target.value))}
                        step="0.05"
                        min="0"
                        max="1"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Block-wise Learning Rates - Conditional */}
              {settings.showBlockwiseLR && (
                <div className="border-t pt-4 space-y-3">
                  <h4 className="text-md font-medium text-purple-400">Block-wise Learning Rates (Very Advanced)</h4>
                  <p className="text-sm text-gray-400">Fine-grained control over per-block learning rates and dimensions</p>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="down_lr_weight">Down LR Weight</Label>
                      <Input
                        id="down_lr_weight"
                        value={config.down_lr_weight}
                        onChange={(e) => handleChange('down_lr_weight', e.target.value)}
                        placeholder="e.g., 1,1,1,1,1,1,1,1,1,1,1,1 (12 values)"
                      />
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="mid_lr_weight">Mid LR Weight</Label>
                        <Input
                          id="mid_lr_weight"
                          value={config.mid_lr_weight}
                          onChange={(e) => handleChange('mid_lr_weight', e.target.value)}
                          placeholder="e.g., 1"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="block_lr_zero_threshold">Block LR Zero Threshold</Label>
                        <Input
                          id="block_lr_zero_threshold"
                          value={config.block_lr_zero_threshold}
                          onChange={(e) => handleChange('block_lr_zero_threshold', e.target.value)}
                          placeholder="e.g., 0.1"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="up_lr_weight">Up LR Weight</Label>
                      <Input
                        id="up_lr_weight"
                        value={config.up_lr_weight}
                        onChange={(e) => handleChange('up_lr_weight', e.target.value)}
                        placeholder="e.g., 1,1,1,1,1,1,1,1,1,1,1,1 (12 values)"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="block_dims">Block Dims</Label>
                      <Input
                        id="block_dims"
                        value={config.block_dims}
                        onChange={(e) => handleChange('block_dims', e.target.value)}
                        placeholder="e.g., 2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,2,2,2 (25 values)"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="block_alphas">Block Alphas</Label>
                      <Input
                        id="block_alphas"
                        value={config.block_alphas}
                        onChange={(e) => handleChange('block_alphas', e.target.value)}
                        placeholder="e.g., 1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1 (25 values)"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="conv_block_dims">Conv Block Dims</Label>
                      <Input
                        id="conv_block_dims"
                        value={config.conv_block_dims}
                        onChange={(e) => handleChange('conv_block_dims', e.target.value)}
                        placeholder="e.g., 2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,2,2,2 (25 values)"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="conv_block_alphas">Conv Block Alphas</Label>
                      <Input
                        id="conv_block_alphas"
                        value={config.conv_block_alphas}
                        onChange={(e) => handleChange('conv_block_alphas', e.target.value)}
                        placeholder="e.g., 1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1 (25 values)"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Tab 4: Learning Rate */}
          <TabsContent value="learning" className="space-y-4">
            <h3 className="text-lg font-semibold">Learning Rate Configuration</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="unet_lr">UNet Learning Rate</Label>
                <Input
                  id="unet_lr"
                  type="number"
                  value={config.unet_lr}
                  onChange={(e) => handleChange('unet_lr', parseFloat(e.target.value))}
                  step="0.00001"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="text_encoder_lr">Text Encoder Learning Rate</Label>
                <Input
                  id="text_encoder_lr"
                  type="number"
                  value={config.text_encoder_lr}
                  onChange={(e) => handleChange('text_encoder_lr', parseFloat(e.target.value))}
                  step="0.00001"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="lr_scheduler">LR Scheduler</Label>
                <Select value={config.lr_scheduler} onValueChange={(val) => handleChange('lr_scheduler', val)}>
                  <SelectTrigger id="lr_scheduler">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cosine">Cosine</SelectItem>
                    <SelectItem value="cosine_with_restarts">Cosine with Restarts</SelectItem>
                    <SelectItem value="constant">Constant</SelectItem>
                    <SelectItem value="constant_with_warmup">Constant with Warmup</SelectItem>
                    <SelectItem value="linear">Linear</SelectItem>
                    <SelectItem value="polynomial">Polynomial</SelectItem>
                    <SelectItem value="rex">REX</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="lr_scheduler_number">LR Scheduler Number (for restarts/polynomial)</Label>
                <Input
                  id="lr_scheduler_number"
                  type="number"
                  value={config.lr_scheduler_number}
                  onChange={(e) => handleChange('lr_scheduler_number', parseInt(e.target.value))}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="lr_warmup_ratio">LR Warmup Ratio</Label>
                <Input
                  id="lr_warmup_ratio"
                  type="number"
                  value={config.lr_warmup_ratio}
                  onChange={(e) => handleChange('lr_warmup_ratio', parseFloat(e.target.value))}
                  step="0.01"
                  min="0"
                  max="1"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="lr_warmup_steps">LR Warmup Steps (0 = use ratio)</Label>
                <Input
                  id="lr_warmup_steps"
                  type="number"
                  value={config.lr_warmup_steps}
                  onChange={(e) => handleChange('lr_warmup_steps', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="lr_power">LR Power (for polynomial scheduler)</Label>
                <Input
                  id="lr_power"
                  type="number"
                  value={config.lr_power}
                  onChange={(e) => handleChange('lr_power', parseFloat(e.target.value))}
                  step="0.1"
                />
              </div>
            </div>

            <div className="border-t pt-4">
              <h4 className="text-md font-medium mb-3">Optimizer</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="optimizer_type">Optimizer Type</Label>
                  <Select value={config.optimizer_type} onValueChange={(val) => handleChange('optimizer_type', val)}>
                    <SelectTrigger id="optimizer_type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AdamW8bit">AdamW 8bit</SelectItem>
                      <SelectItem value="AdamW">AdamW</SelectItem>
                      <SelectItem value="PagedAdamW8bit">Paged AdamW 8bit</SelectItem>
                      <SelectItem value="PagedAdamW32bit">Paged AdamW 32bit</SelectItem>
                      <SelectItem value="Lion">Lion</SelectItem>
                      <SelectItem value="Lion8bit">Lion 8bit</SelectItem>
                      <SelectItem value="PagedLion8bit">Paged Lion 8bit</SelectItem>
                      <SelectItem value="SGDNesterov">SGD Nesterov</SelectItem>
                      <SelectItem value="SGDNesterov8bit">SGD Nesterov 8bit</SelectItem>
                      <SelectItem value="DAdaptation">DAdaptation</SelectItem>
                      <SelectItem value="DAdaptAdam">DAdapt Adam</SelectItem>
                      <SelectItem value="DAdaptAdaGrad">DAdapt AdaGrad</SelectItem>
                      <SelectItem value="DAdaptAdan">DAdapt Adan</SelectItem>
                      <SelectItem value="DAdaptSGD">DAdapt SGD</SelectItem>
                      <SelectItem value="DAdaptLion">DAdapt Lion</SelectItem>
                      <SelectItem value="Prodigy">Prodigy</SelectItem>
                      <SelectItem value="AdaFactor">AdaFactor</SelectItem>
                      <SelectItem value="LoraEasyCustomOptimizer.came.CAME">CAME</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="weight_decay">Weight Decay</Label>
                  <Input
                    id="weight_decay"
                    type="number"
                    value={config.weight_decay}
                    onChange={(e) => handleChange('weight_decay', parseFloat(e.target.value))}
                    step="0.001"
                    min="0"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gradient_accumulation_steps">Gradient Accumulation Steps</Label>
                  <Input
                    id="gradient_accumulation_steps"
                    type="number"
                    value={config.gradient_accumulation_steps}
                    onChange={(e) => handleChange('gradient_accumulation_steps', parseInt(e.target.value))}
                    min="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max_grad_norm">Max Gradient Norm</Label>
                  <Input
                    id="max_grad_norm"
                    type="number"
                    value={config.max_grad_norm}
                    onChange={(e) => handleChange('max_grad_norm', parseFloat(e.target.value))}
                    step="0.1"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="optimizer_args">Optimizer Args (JSON)</Label>
                  <Input
                    id="optimizer_args"
                    value={config.optimizer_args}
                    onChange={(e) => handleChange('optimizer_args', e.target.value)}
                    placeholder='{"weight_decay": 0.01}'
                  />
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Tab 5: Advanced */}
          <TabsContent value="advanced" className="space-y-4">
            <h3 className="text-lg font-semibold">Advanced Training Techniques</h3>

            <div className="border-t pt-4 space-y-4">
              <h4 className="text-md font-medium">SNR & Noise Control</h4>

              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="min_snr_gamma_enabled"
                    checked={config.min_snr_gamma_enabled}
                    onCheckedChange={(checked) => handleChange('min_snr_gamma_enabled', checked)}
                  />
                  <Label htmlFor="min_snr_gamma_enabled" className="font-normal">Enable Min SNR Gamma</Label>
                </div>

                {config.min_snr_gamma_enabled && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="min_snr_gamma">Min SNR Gamma Value</Label>
                    <Input
                      id="min_snr_gamma"
                      type="number"
                      value={config.min_snr_gamma}
                      onChange={(e) => handleChange('min_snr_gamma', parseFloat(e.target.value))}
                      step="0.1"
                    />
                  </div>
                )}

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="ip_noise_gamma_enabled"
                    checked={config.ip_noise_gamma_enabled}
                    onCheckedChange={(checked) => handleChange('ip_noise_gamma_enabled', checked)}
                  />
                  <Label htmlFor="ip_noise_gamma_enabled" className="font-normal">Enable IP Noise Gamma</Label>
                </div>

                {config.ip_noise_gamma_enabled && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="ip_noise_gamma">IP Noise Gamma Value</Label>
                    <Input
                      id="ip_noise_gamma"
                      type="number"
                      value={config.ip_noise_gamma}
                      onChange={(e) => handleChange('ip_noise_gamma', parseFloat(e.target.value))}
                      step="0.01"
                    />
                  </div>
                )}

                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="noise_offset">Noise Offset</Label>
                    <Input
                      id="noise_offset"
                      type="number"
                      value={config.noise_offset}
                      onChange={(e) => handleChange('noise_offset', parseFloat(e.target.value))}
                      step="0.01"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="adaptive_noise_scale">Adaptive Noise Scale</Label>
                    <Input
                      id="adaptive_noise_scale"
                      type="number"
                      value={config.adaptive_noise_scale}
                      onChange={(e) => handleChange('adaptive_noise_scale', parseFloat(e.target.value))}
                      step="0.01"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="multires_noise_discount">Multires Noise Discount</Label>
                    <Input
                      id="multires_noise_discount"
                      type="number"
                      value={config.multires_noise_discount}
                      onChange={(e) => handleChange('multires_noise_discount', parseFloat(e.target.value))}
                      step="0.01"
                    />
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="multinoise"
                    checked={config.multinoise}
                    onCheckedChange={(checked) => handleChange('multinoise', checked)}
                  />
                  <Label htmlFor="multinoise" className="font-normal">Multi-Noise</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="zero_terminal_snr"
                    checked={config.zero_terminal_snr}
                    onCheckedChange={(checked) => handleChange('zero_terminal_snr', checked)}
                  />
                  <Label htmlFor="zero_terminal_snr" className="font-normal">Zero Terminal SNR</Label>
                </div>
              </div>
            </div>

            <div className="border-t pt-4 space-y-4">
              <h4 className="text-md font-medium">Memory & Performance</h4>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="gradient_checkpointing"
                  checked={config.gradient_checkpointing}
                  onCheckedChange={(checked) => handleChange('gradient_checkpointing', checked)}
                />
                <Label htmlFor="gradient_checkpointing" className="font-normal">Gradient Checkpointing (saves VRAM)</Label>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="mixed_precision">Mixed Precision</Label>
                  <Select value={config.mixed_precision} onValueChange={(val) => handleChange('mixed_precision', val)}>
                    <SelectTrigger id="mixed_precision">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="no">No (FP32)</SelectItem>
                      <SelectItem value="fp16">FP16</SelectItem>
                      <SelectItem value="bf16">BF16</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="vae_batch_size">VAE Batch Size</Label>
                  <Input
                    id="vae_batch_size"
                    type="number"
                    value={config.vae_batch_size}
                    onChange={(e) => handleChange('vae_batch_size', parseInt(e.target.value))}
                    min="1"
                  />
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="full_fp16"
                    checked={config.full_fp16}
                    onCheckedChange={(checked) => handleChange('full_fp16', checked)}
                  />
                  <Label htmlFor="full_fp16" className="font-normal">Full FP16 Training</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="no_half_vae"
                    checked={config.no_half_vae}
                    onCheckedChange={(checked) => handleChange('no_half_vae', checked)}
                  />
                  <Label htmlFor="no_half_vae" className="font-normal">No Half VAE</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="cache_latents"
                    checked={config.cache_latents}
                    onCheckedChange={(checked) => handleChange('cache_latents', checked)}
                  />
                  <Label htmlFor="cache_latents" className="font-normal">Cache Latents</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="cache_latents_to_disk"
                    checked={config.cache_latents_to_disk}
                    onCheckedChange={(checked) => handleChange('cache_latents_to_disk', checked)}
                  />
                  <Label htmlFor="cache_latents_to_disk" className="font-normal">Cache Latents to Disk</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="cache_text_encoder_outputs"
                    checked={config.cache_text_encoder_outputs}
                    onCheckedChange={(checked) => handleChange('cache_text_encoder_outputs', checked)}
                  />
                  <Label htmlFor="cache_text_encoder_outputs" className="font-normal">Cache Text Encoder Outputs</Label>
                </div>
              </div>

              <div className="space-y-3 border-t pt-4">
                <h4 className="text-md font-medium">Cross Attention & Precision</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="cross_attention">Cross Attention Type</Label>
                    <Select value={config.cross_attention} onValueChange={(val) => handleChange('cross_attention', val)}>
                      <SelectTrigger id="cross_attention">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sdpa">SDPA (Scaled Dot Product)</SelectItem>
                        <SelectItem value="xformers">xFormers</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center space-x-2 mt-7">
                    <Checkbox
                      id="fp8_base"
                      checked={config.fp8_base}
                      onCheckedChange={(checked) => handleChange('fp8_base', checked)}
                    />
                    <Label htmlFor="fp8_base" className="font-normal">FP8 Base (experimental, great for VRAM savings)</Label>
                  </div>
                </div>
              </div>

              <div className="space-y-3 border-t pt-4">
                <h4 className="text-md font-medium">Training Targets</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="network_train_unet_only"
                      checked={config.network_train_unet_only}
                      onCheckedChange={(checked) => handleChange('network_train_unet_only', checked)}
                    />
                    <Label htmlFor="network_train_unet_only" className="font-normal">Train U-Net Only (recommended for SDXL)</Label>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="prior_loss_weight">Prior Loss Weight (for regularization)</Label>
                    <Input
                      id="prior_loss_weight"
                      type="number"
                      value={config.prior_loss_weight}
                      onChange={(e) => handleChange('prior_loss_weight', parseFloat(e.target.value))}
                      step="0.1"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              {/* SD 2.x Parameters - Conditional */}
              {settings.showSD2Params && (
                <div className="space-y-3 border-t pt-4">
                  <h4 className="text-md font-medium text-purple-400">SD 2.x Parameters</h4>
                  <p className="text-sm text-gray-400">For Stable Diffusion 2.0/2.1 models</p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="v2"
                        checked={config.v2}
                        onCheckedChange={(checked) => handleChange('v2', checked)}
                      />
                      <Label htmlFor="v2" className="font-normal">SD 2.x Base Model</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="v_parameterization"
                        checked={config.v_parameterization}
                        onCheckedChange={(checked) => handleChange('v_parameterization', checked)}
                      />
                      <Label htmlFor="v_parameterization" className="font-normal">V-Parameterization (SD 2.x 768px or SDXL v-pred)</Label>
                    </div>
                  </div>
                </div>
              )}

              {/* Performance Tuning - Conditional */}
              {settings.showPerformanceTuning && (
                <div className="space-y-3 border-t pt-4">
                  <h4 className="text-md font-medium text-purple-400">Performance Tuning</h4>
                  <p className="text-sm text-gray-400">Advanced data loading and memory optimizations</p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="persistent_data_loader_workers">Persistent Data Loader Workers (0=auto)</Label>
                      <Input
                        id="persistent_data_loader_workers"
                        type="number"
                        value={config.persistent_data_loader_workers}
                        onChange={(e) => handleChange('persistent_data_loader_workers', parseInt(e.target.value))}
                        min="0"
                      />
                    </div>

                    <div className="flex items-center space-x-2 mt-7">
                      <Checkbox
                        id="no_token_padding"
                        checked={config.no_token_padding}
                        onCheckedChange={(checked) => handleChange('no_token_padding', checked)}
                      />
                      <Label htmlFor="no_token_padding" className="font-normal">No Token Padding (memory optimization)</Label>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Tab 6: Saving */}
          <TabsContent value="saving" className="space-y-4">
            <h3 className="text-lg font-semibold">Saving & Checkpoints</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="save_every_n_epochs">Save Every N Epochs</Label>
                <Input
                  id="save_every_n_epochs"
                  type="number"
                  value={config.save_every_n_epochs}
                  onChange={(e) => handleChange('save_every_n_epochs', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_every_n_steps">Save Every N Steps (0 = off)</Label>
                <Input
                  id="save_every_n_steps"
                  type="number"
                  value={config.save_every_n_steps}
                  onChange={(e) => handleChange('save_every_n_steps', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_last_n_epochs">Save Last N Epochs (keep only)</Label>
                <Input
                  id="save_last_n_epochs"
                  type="number"
                  value={config.save_last_n_epochs}
                  onChange={(e) => handleChange('save_last_n_epochs', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_last_n_epochs_state">Save Last N Epoch States (keep only)</Label>
                <Input
                  id="save_last_n_epochs_state"
                  type="number"
                  value={config.save_last_n_epochs_state}
                  onChange={(e) => handleChange('save_last_n_epochs_state', parseInt(e.target.value))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_model_as">Save Model As</Label>
                <Select value={config.save_model_as} onValueChange={(val) => handleChange('save_model_as', val)}>
                  <SelectTrigger id="save_model_as">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="safetensors">SafeTensors</SelectItem>
                    <SelectItem value="ckpt">Checkpoint (.ckpt)</SelectItem>
                    <SelectItem value="pt">PyTorch (.pt)</SelectItem>
                    <SelectItem value="diffusers">Diffusers</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="save_precision">Save Precision</Label>
                <Select value={config.save_precision} onValueChange={(val) => handleChange('save_precision', val)}>
                  <SelectTrigger id="save_precision">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fp16">FP16</SelectItem>
                    <SelectItem value="bf16">BF16</SelectItem>
                    <SelectItem value="float">FP32</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="output_name">Output Name (optional)</Label>
                <Input
                  id="output_name"
                  value={config.output_name}
                  onChange={(e) => handleChange('output_name', e.target.value)}
                  placeholder="Leave empty to use project name"
                />
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="no_metadata"
                  checked={config.no_metadata}
                  onCheckedChange={(checked) => handleChange('no_metadata', checked)}
                />
                <Label htmlFor="no_metadata" className="font-normal">No Metadata (smaller file size)</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="save_state"
                  checked={config.save_state}
                  onCheckedChange={(checked) => handleChange('save_state', checked)}
                />
                <Label htmlFor="save_state" className="font-normal">Save Training State (for resuming training)</Label>
              </div>

              {/* Experimental Features - Conditional */}
              {settings.showExperimentalFeatures && (
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="weighted_captions"
                    checked={config.weighted_captions}
                    onCheckedChange={(checked) => handleChange('weighted_captions', checked)}
                  />
                  <Label htmlFor="weighted_captions" className="font-normal">Weighted Captions (experimental)</Label>
                </div>
              )}

              {settings.showExperimentalFeatures && (
                <div className="space-y-2">
                  <Label htmlFor="save_last_n_steps_state">Save Last N Steps State (0=disabled, experimental)</Label>
                  <Input
                    id="save_last_n_steps_state"
                    type="number"
                    value={config.save_last_n_steps_state}
                    onChange={(e) => handleChange('save_last_n_steps_state', parseInt(e.target.value))}
                    min="0"
                  />
                </div>
              )}
            </div>

            <div className="border-t pt-4">
              <h4 className="text-md font-medium mb-3">Sample Generation</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="sample_every_n_epochs">Sample Every N Epochs</Label>
                  <Input
                    id="sample_every_n_epochs"
                    type="number"
                    value={config.sample_every_n_epochs}
                    onChange={(e) => handleChange('sample_every_n_epochs', parseInt(e.target.value))}
                    min="0"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sample_every_n_steps">Sample Every N Steps (0 = off)</Label>
                  <Input
                    id="sample_every_n_steps"
                    type="number"
                    value={config.sample_every_n_steps}
                    onChange={(e) => handleChange('sample_every_n_steps', parseInt(e.target.value))}
                    min="0"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sample_sampler">Sample Sampler</Label>
                  <Select value={config.sample_sampler} onValueChange={(val) => handleChange('sample_sampler', val)}>
                    <SelectTrigger id="sample_sampler">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="euler_a">Euler A</SelectItem>
                      <SelectItem value="euler">Euler</SelectItem>
                      <SelectItem value="ddim">DDIM</SelectItem>
                      <SelectItem value="ddpm">DDPM</SelectItem>
                      <SelectItem value="pndm">PNDM</SelectItem>
                      <SelectItem value="k_lms">K-LMS</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="md:col-span-2 space-y-2">
                  <Label htmlFor="sample_prompts">Sample Prompts (one per line)</Label>
                  <Textarea
                    id="sample_prompts"
                    value={config.sample_prompts}
                    onChange={(e) => handleChange('sample_prompts', e.target.value)}
                    rows={4}
                    placeholder="a cat sitting on a chair&#10;a dog running in a field&#10;an astronaut riding a horse"
                  />
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <Button onClick={handleStart} disabled={loading} className="flex-1" size="lg">
            <Play className="w-5 h-5 mr-2" />
            {loading ? 'Starting...' : 'Start Training'}
          </Button>

          <Button onClick={handleSave} disabled={loading} variant="outline" size="lg">
            <Save className="w-5 h-5 mr-2" />
            Save Config
          </Button>
        </div>
      </CardContent>

      {/* File Browser Dialog */}
      <FileBrowser
        open={browserOpen.field !== null}
        onOpenChange={(open) => {
          if (!open) {
            setBrowserOpen({ field: null, mode: 'directory', title: '', description: '' });
          }
        }}
        onSelect={handleBrowserSelect}
        mode={browserOpen.mode}
        title={browserOpen.title}
        description={browserOpen.description}
        startPath="/workspace"
      />
    </Card>
  );
}
