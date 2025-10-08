# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/training_widget.py

import os
import warnings

import ipywidgets as widgets
from IPython.display import display

from core.refactored_training_manager import RefactoredTrainingManager

from .training_monitor_widget import TrainingMonitorWidget

# Suppress FutureWarnings at import time
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

class TrainingWidget:
    def __init__(self, training_manager=None):
        # Use dependency injection - accept manager instance or create default
        if training_manager is None:
            training_manager = RefactoredTrainingManager()

        self.manager = training_manager
        # Race condition guard: prevent observer conflicts during programmatic changes
        self._programmatic_change = False
        self.create_widgets()

    def _parse_learning_rate(self, lr_text, widget=None):
        """Parse learning rate from text input supporting both scientific notation and decimals"""
        try:
            # Handle both scientific notation (5e-4) and decimal (0.0005)
            lr_value = float(lr_text)
            
            # Validate bounds - no negative learning rates (sorry, no memes!)
            if lr_value < 0:
                raise ValueError(f"Learning rate cannot be negative: {lr_value}")
            if lr_value > 1.0:
                print(f"⚠️ Warning: Very high learning rate {lr_value}. Consider using < 1.0")
            
            return lr_value
        except ValueError as e:
            default_value = "1e-4"
            print(f"⚠️ Invalid learning rate: {lr_text} ({e}). Using default {default_value}")
            
            # Update the UI widget to show the corrected value
            if widget is not None:
                self._programmatic_change = True
                try:
                    widget.value = default_value
                finally:
                    self._programmatic_change = False
            
            return float(default_value)

    def create_widgets(self):
        header_icon = "⭐"
        header_main = widgets.HTML(f"<h2>{header_icon} 3. Training Configuration</h2>")

        # --- Project Settings ---
        project_desc = widgets.HTML("<h3>▶️ Project Settings</h3><p>Define your project name, the path to your base model, and your dataset directory. You can also specify an existing LoRA to continue training from and your Weights & Biases API key for logging.</p>")
        self.project_name = widgets.Text(description="Project Name:", placeholder="e.g., my-awesome-lora (no spaces or special characters)", layout=widgets.Layout(width='99%'))
        self.model_type = widgets.Dropdown(options=['SD1.5/2.0', 'SDXL', 'Flux', 'SD3'], value='SDXL', description='Model Type:', style={'description_width': 'initial'})
        # Model selection with auto-populated dropdown
        self.model_dropdown = widgets.Dropdown(
            options=[('Select a model...', '')],
            description='Base Model:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        self.model_refresh_btn = widgets.Button(description="🔄 Refresh", button_style='info', layout=widgets.Layout(width='100px'))
        self.model_path = widgets.Text(description="Or custom path:", placeholder="Full path to model file", layout=widgets.Layout(width='99%'))
        self.vae_path = widgets.Text(description="VAE Path:", placeholder="Optional VAE file path", layout=widgets.Layout(width='99%'))

        # Auto-populate models on widget creation
        self._refresh_model_list()

        # Connect refresh button
        self.model_refresh_btn.on_click(lambda b: self._refresh_model_list())

        # Update model_path when dropdown changes
        def on_model_selected(change):
            if change['new']:
                self.model_path.value = change['new']
        self.model_dropdown.observe(on_model_selected, names='value')

        # Flux/SD3 specific widgets (individual widgets for better unified config integration)
        self.clip_l_path = widgets.Text(description="CLIP-L Path:", placeholder="Path to clip_l.safetensors", layout=widgets.Layout(width='99%', display='none'))
        self.clip_g_path = widgets.Text(description="CLIP-G Path:", placeholder="Path to clip_g.safetensors (for SD3)", layout=widgets.Layout(width='99%', display='none'))
        self.t5xxl_path = widgets.Text(description="T5-XXL Path:", placeholder="Path to t5xxl.safetensors", layout=widgets.Layout(width='99%', display='none'))
        # Legacy container for backward compatibility (but now empty)
        self.flux_sd3_widgets = widgets.VBox([])
        self.flux_sd3_widgets.layout.display = 'none'

        self.dataset_dir = widgets.Text(description="Dataset Dir:", placeholder="Absolute path to your dataset directory (e.g., /path/to/my_dataset)", layout=widgets.Layout(width='99%'))
        self.continue_from_lora = widgets.Text(description="Continue from LoRA:", placeholder="Absolute path to an existing LoRA to continue training (optional)", layout=widgets.Layout(width='99%'))
        self.wandb_key = widgets.Password(description="WandB API Key:", placeholder="Your key will be hidden (e.g., ••••••••••••••••••••••••)", layout=widgets.Layout(width='99%'))

        def _on_model_type_change(change):
            if change['new'] in ['Flux', 'SD3']:
                self.clip_l_path.layout.display = 'block'
                self.clip_g_path.layout.display = 'block'
                self.t5xxl_path.layout.display = 'block'
            else:
                self.clip_l_path.layout.display = 'none'
                self.clip_g_path.layout.display = 'none'
                self.t5xxl_path.layout.display = 'none'

        self.model_type.observe(_on_model_type_change, names='value')

        # Model selection layout
        model_selection_box = widgets.HBox([self.model_dropdown, self.model_refresh_btn])

        project_box = widgets.VBox([project_desc, self.project_name, self.model_type, model_selection_box, self.model_path, self.vae_path, self.clip_l_path, self.clip_g_path, self.t5xxl_path, self.dataset_dir, self.continue_from_lora, self.wandb_key])

        # --- Training Configuration (merged: Basic Settings + Learning Rate + Training Options) ---
        training_config_desc = widgets.HTML("""<h3>▶️ Training Configuration</h3>
        <p>Configure all core training parameters including dataset settings, learning rates, optimizer, and training options. Total steps calculated automatically based on your dataset.</p>""")
        self.resolution = widgets.IntText(value=1024, description='Resolution:', style={'description_width': 'initial'})
        self.num_repeats = widgets.IntText(value=10, min=1, description='Num Repeats:', style={'description_width': 'initial'})
        self.epochs = widgets.IntText(value=10, min=1, description='Epochs:', style={'description_width': 'initial'})
        self.max_train_steps = widgets.IntText(value=0, min=0, description='Max Train Steps (0=disabled):', style={'description_width': 'initial'})
        self.train_batch_size = widgets.IntText(value=4, min=1, description='Train Batch Size:', style={'description_width': 'initial'})
        self.seed = widgets.IntText(value=42, description='Training Seed:', style={'description_width': 'initial'})
        self.flip_aug = widgets.Checkbox(value=False, description="Flip Augmentation (data augmentation)", indent=False)
        self.shuffle_caption = widgets.Checkbox(value=True, description="Shuffle Captions (improves variety, incompatible with text encoder caching)", indent=False)

        # Basic training options (moved from advanced)
        self.keep_tokens = widgets.IntText(value=0, description='Keep Tokens:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.clip_skip = widgets.IntText(value=2, description='Clip Skip:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        
        # Caption dropout controls (moved from advanced)
        self.caption_dropout_rate = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Caption Dropout Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        self.caption_tag_dropout_rate = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Tag Dropout Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        # Advanced caption controls
        self.caption_dropout_every_n_epochs = widgets.IntText(
            value=0, min=0,
            description='Caption Dropout Every N Epochs (0=disabled):',
            style={'description_width': 'initial'}
        )
        self.keep_tokens_separator = widgets.Text(
            value='',
            description='Keep Tokens Separator:',
            style={'description_width': 'initial'},
            placeholder='|||'
        )
        self.secondary_separator = widgets.Text(
            value='',
            description='Secondary Caption Separator:',
            style={'description_width': 'initial'},
            placeholder=';'
        )
        self.enable_wildcard = widgets.Checkbox(
            value=False,
            description="Enable Wildcard Notation in Captions",
            indent=False
        )

        # Auto-detected dataset size (will be updated when dataset directory changes)
        self.dataset_size = widgets.IntText(value=0, description="Dataset Size:", style={'description_width': 'initial'}, disabled=True)
        self.step_calculator = widgets.HTML()

        # Auto-detect dataset size when dataset_dir changes
        def update_dataset_size(*args):
            # Race condition guard: skip if we're in the middle of programmatic changes
            if self._programmatic_change:
                return
                
            dataset_path = self.dataset_dir.value.strip()
            
            # Set flag to prevent observer loops
            self._programmatic_change = True
            try:
                if dataset_path:
                    try:
                        from core.image_utils import count_images_in_directory
                        image_count = count_images_in_directory(dataset_path)
                        self.dataset_size.value = image_count  # Observer blocked by flag
                    except Exception as e:
                        print(f"⚠️ Could not count images in {dataset_path}: {e}")
                        self.dataset_size.value = 0
                else:
                    self.dataset_size.value = 0
            finally:
                # Always clear the flag, even if an error occurred
                self._programmatic_change = False
            
            # Manually trigger calculation once after all changes
            update_step_calculation()

        self.dataset_dir.observe(update_dataset_size, names='value')

        def update_step_calculation(*args):
            # Race condition guard: skip if we're in the middle of programmatic changes
            if self._programmatic_change:
                return
                
            images = self.dataset_size.value
            repeats = self.num_repeats.value
            epochs = self.epochs.value
            batch_size = self.train_batch_size.value

            if batch_size > 0 and images > 0 and repeats > 0 and epochs > 0:
                total_steps = (images * repeats * epochs) // batch_size

                # Neutral color scheme
                color = "#17a2b8"  # Blue-teal, neutral

                self.step_calculator.value = f"""
                <div style='background: {color}20; padding: 10px; border-left: 4px solid {color}; margin: 5px 0;'>
                <strong>📊 Total Steps: {total_steps}</strong><br>
                {images} images × {repeats} repeats × {epochs} epochs ÷ {batch_size} batch = {total_steps} steps
                </div>
                """
            else:
                # No dataset or invalid parameters
                self.step_calculator.value = """
                <div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 5px 0;'>
                <strong>📊 Total Steps: Pending</strong><br>
                <em>Select a dataset directory to calculate training steps</em>
                </div>
                """

        # Attach observers to update calculation
        self.dataset_size.observe(update_step_calculation, names='value')
        self.num_repeats.observe(update_step_calculation, names='value')
        self.epochs.observe(update_step_calculation, names='value')
        self.train_batch_size.observe(update_step_calculation, names='value')

        # Initial calculation
        update_step_calculation()

        # Configuration warnings
        self.config_warnings = widgets.HTML()

        def check_config_conflicts(*args):
            # Race condition guard: skip if we're in the middle of programmatic changes
            if self._programmatic_change:
                return
                
            warnings = []

            # Check text encoder caching vs shuffle caption conflict
            if self.cache_text_encoder_outputs.value and self.shuffle_caption.value:
                warnings.append("⚠️ Cannot use Caption Shuffling with Text Encoder Caching")

            # Check text encoder caching vs text encoder training conflict
            if self.cache_text_encoder_outputs.value and float(self.text_encoder_lr.value) > 0:
                warnings.append("⚠️ Cannot cache Text Encoder while training it (set Text LR to 0)")
                self.text_encoder_lr.value = '0'

            # Check random crop vs latent caching conflict
            if self.random_crop.value and self.cache_latents.value:
                warnings.append("⚠️ Cannot use Random Crop with Latent Caching - choose one or the other")

            if warnings:
                warning_html = "<div style='padding: 10px; border: 1px solid #856404; border-radius: 5px; margin: 5px 0;'>"
                warning_html += "<br>".join(warnings)
                warning_html += "<br><em>💡 Fix these conflicts to enable training</em></div>"
                self.config_warnings.value = warning_html
            else:
                self.config_warnings.value = ""

        # Note: Observers will be attached after all widgets are created

        # Learning rate widgets (no separate section header)
        self.unet_lr = widgets.Text(value='5e-4', placeholder='e.g., 5e-4 or 0.0005', description='🧠 UNet LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.text_encoder_lr = widgets.Text(value='1e-4', placeholder='e.g., 1e-4 or 0.0001', description='📝 Text LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.lr_scheduler = widgets.Dropdown(options=['cosine', 'cosine_with_restarts', 'constant', 'linear', 'polynomial', 'rex'], value='cosine', description='Scheduler:', style={'description_width': 'initial'})
        self.lr_scheduler_number = widgets.IntSlider(value=3, min=1, max=10, description='Scheduler Num (for restarts/polynomial):', style={'description_width': 'initial'}, continuous_update=False)
        self.lr_warmup_ratio = widgets.Text(value='0.05', description='Warmup Ratio:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.lr_warmup_steps = widgets.IntText(value=0, description='Warmup Steps (0=use ratio):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.lr_power = widgets.Text(value='1.0', description='LR Power (for polynomial scheduler):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.min_snr_gamma_enabled = widgets.Checkbox(value=True, description="Enable Min SNR Gamma (recommended for better results)", indent=False)
        self.min_snr_gamma = widgets.Text(value='5.0', description='Min SNR Gamma:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.ip_noise_gamma_enabled = widgets.Checkbox(value=False, description="Enable IP Noise Gamma", indent=False)
        self.ip_noise_gamma = widgets.FloatSlider(value=0.05, min=0.0, max=0.1, step=0.01, description='IP Noise Gamma:', style={'description_width': 'initial'}, continuous_update=False)
        self.multinoise = widgets.Checkbox(value=False, description="Multi-noise (can help with color balance)", indent=False)
        self.multires_noise_discount = widgets.Text(value='0.25', description='Multi-Res Noise Discount:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.noise_offset = widgets.Text(value='0.0', description='Noise Offset:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.adaptive_noise_scale = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.02, step=0.001,
            description='Adaptive Noise Scale:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # Loss section widgets - HUBER LOSS DELETED (was corrupting safetensors files)
        # self.loss_type = widgets.Dropdown(options=['l2', 'huber'], value='l2', description='Loss Type:', style={'description_width': 'initial'})
        # self.huber_c = widgets.Text(value='0.1', description='Huber C:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        # self.huber_schedule = widgets.Text(value='snr', description='Huber Schedule:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        # Training options widgets (no separate section header)
        self.zero_terminal_snr = widgets.Checkbox(value=False, description="Zero Terminal SNR (recommended for SDXL)", indent=False)
        self.enable_bucket = widgets.Checkbox(value=True, description="Enable Bucket (resolution bucketing)", indent=False)
        self.gradient_checkpointing = widgets.Checkbox(value=True, description="Gradient Checkpointing (saves memory - REQUIRED for 4090)", indent=False)
        self.gradient_accumulation_steps = widgets.IntText(value=1, description='Gradient Accumulation Steps:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.max_grad_norm = widgets.Text(value='1.0', description='Max Grad Norm:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.full_fp16 = widgets.Checkbox(value=False, description="Full FP16 (more aggressive mixed precision)", indent=False)
        self.random_crop = widgets.Checkbox(value=False, description="Random Crop (data augmentation)", indent=False)

        # Bucketing settings (moved from "advanced")
        self.sdxl_bucket_optimization = widgets.Checkbox(value=False, description="📐 SDXL Bucket Optimization (32 steps vs 64 standard)", indent=False)
        self.min_bucket_reso = widgets.IntSlider(value=256, min=128, max=512, step=64, description='Min Bucket Resolution:', style={'description_width': 'initial'}, continuous_update=False)
        self.max_bucket_reso = widgets.IntSlider(value=2048, min=1024, max=4096, step=512, description='Max Bucket Resolution:', style={'description_width': 'initial'}, continuous_update=False)
        self.bucket_no_upscale = widgets.Checkbox(value=False, description="No Bucket Upscale (prevent upscaling small images)", indent=False)

        # VAE settings (moved from "advanced")
        self.vae_batch_size = widgets.IntSlider(value=1, min=1, max=8, step=1, description='VAE Batch Size:', style={'description_width': 'initial'}, continuous_update=False)
        self.no_half_vae = widgets.Checkbox(value=False, description="No Half VAE (fixes some VAE issues, uses more VRAM)", indent=False)

        # LoRA Structure widgets (moved to unified config)
        self.lora_type = widgets.Dropdown(
            options=[
                'LoRA', 'LoCon', 'LoKR', 'DyLoRA',
                'DoRA (Weight Decomposition)',
                'LoHa (Hadamard Product)',
                '(IA)³ (Few Parameters)', 
                'GLoRA (Generalized LoRA)',
                'Native Fine-Tuning (Full)',
                'Diag-OFT (Orthogonal Fine-Tuning)',
                'BOFT (Butterfly Transform)'
            ],
            value='LoRA',
            description='LoRA Type:',
            style={'description_width': 'initial'}
        )
        self.network_dim = widgets.IntText(value=16, min=4, description='Network Dim:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.network_alpha = widgets.IntText(value=8, min=1, description='Network Alpha:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.dim_from_weights = widgets.Checkbox(value=False, description="Auto-determine dims from weights (overrides network_dim)", indent=False)
        self.network_dropout = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='Network Dropout:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.conv_dim = widgets.IntText(value=16, min=4, description='🧩 Conv Dim (for textures/details):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.conv_alpha = widgets.IntText(value=8, min=1, description='🧩 Conv Alpha (conv learning rate):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        # LyCORIS advanced parameters
        self.factor = widgets.IntText(value=-1, description='🔧 Factor (LoKR decomposition, -1=auto):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.train_norm = widgets.Checkbox(value=False, description="Train Normalization Layers (LyCORIS)", indent=False)
        # LyCORIS dropout parameters
        self.rank_dropout = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='🧩 Rank Dropout (LyCORIS):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.module_dropout = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='🧩 Module Dropout (LyCORIS):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        
        # Advanced LoRA configuration (Weights, Blocks, Conv)
        self.down_lr_weight = widgets.Text(value='', description='Down LR Weight:', placeholder='e.g., 1,1,1,1,1,1,1,1,1,1,1,1 (12 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.mid_lr_weight = widgets.Text(value='', description='Mid LR Weight:', placeholder='e.g., 1 (leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.up_lr_weight = widgets.Text(value='', description='Up LR Weight:', placeholder='e.g., 1,1,1,1,1,1,1,1,1,1,1,1 (12 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.block_lr_zero_threshold = widgets.Text(value='', description='Block LR Zero Threshold:', placeholder='e.g., 0.1 (leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.block_dims = widgets.Text(value='', description='Block Dims:', placeholder='e.g., 2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,2,2,2 (25 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.block_alphas = widgets.Text(value='', description='Block Alphas:', placeholder='e.g., 1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1 (25 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.conv_block_dims = widgets.Text(value='', description='Conv Block Dims:', placeholder='e.g., 2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,2,2,2 (25 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.conv_block_alphas = widgets.Text(value='', description='Conv Block Alphas:', placeholder='e.g., 1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1 (25 values, leave empty for default)', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))

        # Sample generation removed - but keep basic properties for config compatibility
        self.sample_prompt = widgets.Text(value='', disabled=True, layout=widgets.Layout(display='none'))
        self.sample_num_images = widgets.IntText(value=0, disabled=True, layout=widgets.Layout(display='none'))
        self.sample_resolution = widgets.IntText(value=512, disabled=True, layout=widgets.Layout(display='none'))
        self.sample_seed = widgets.IntText(value=42, disabled=True, layout=widgets.Layout(display='none'))

        # Training Options widgets (moved to unified config)
        self.optimizer = widgets.Dropdown(options=['AdamW', 'AdamW8bit', 'Prodigy', 'DAdaptation', 'DadaptAdam', 'DadaptLion', 'Lion', 'SGDNesterov', 'SGDNesterov8bit', 'AdaFactor', 'LoraEasyCustomOptimizer.came.CAME'], value='AdamW', description='Optimizer:', style={'description_width': 'initial'})
        self.weight_decay = widgets.Text(value='0.01', description='Weight Decay:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.optimizer_args = widgets.Text(placeholder='e.g., betas=0.9,0.99,0.999', description='Optimizer Extra Args:', style={'description_width': 'initial'}, layout=widgets.Layout(width='99%'))
        self.cross_attention = widgets.Dropdown(options=['sdpa', 'xformers'], value='sdpa', description='Cross Attention:', style={'description_width': 'initial'})
        self.precision = widgets.Dropdown(options=['fp16', 'bf16', 'float'], value='fp16', description='Precision:', style={'description_width': 'initial'})
        self.fp8_base = widgets.Checkbox(value=False, description="FP8 Base (experimental, requires PyTorch 2.1+)", indent=False)
        self.cache_latents = widgets.Checkbox(value=True, description="Cache Latents (saves memory)", indent=False)
        self.cache_latents_to_disk = widgets.Checkbox(value=True, description="Cache Latents to Disk (uses disk space, saves more memory)", indent=False)
        self.cache_text_encoder_outputs = widgets.Checkbox(value=False, description="Cache Text Encoder Outputs (disables text encoder training)", indent=False)
        self.v2 = widgets.Checkbox(value=False, description="SD 2.x Base Model (enable for SD 2.0/2.1 base models)", indent=False)
        self.v_parameterization = widgets.Checkbox(value=False, description="V-Parameterization (enable for SDXL v-pred models or SD 2.x 768px models)", indent=False)
        self.network_train_unet_only = widgets.Checkbox(value=False, description="🎯 Train U-Net Only (highly recommended for SDXL LoRA)", indent=False)
        
        # Additional missing parameters from Kohya wiki
        self.max_token_length = widgets.IntText(value=75, description='Max Token Length:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.save_every_n_steps = widgets.IntText(value=0, description='Save Every N Steps (0=disabled):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.save_every_n_epochs = widgets.IntText(value=1, description='Save Every N Epochs:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.keep_only_last_n_epochs = widgets.IntText(value=5, description='Keep Last N Epochs:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.save_state = widgets.Checkbox(value=False, description="Save training state for resuming (uses more disk space)", indent=False)
        self.save_last_n_steps_state = widgets.IntText(value=0, description='Save Last N Steps State (0=disabled):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.prior_loss_weight = widgets.Text(value='1.0', description='Prior Loss Weight:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.color_aug = widgets.Checkbox(value=False, description="Color Augmentation", indent=False)
        self.persistent_data_loader_workers = widgets.IntText(value=0, description='Persistent Data Loader Workers (0=auto):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.no_token_padding = widgets.Checkbox(value=False, description="No Token Padding (memory optimization)", indent=False)
        self.weighted_captions = widgets.Checkbox(value=False, description="Weighted Captions", indent=False)

        # Create unified training configuration box (combines ALL training settings)
        training_config_box = widgets.VBox([
            training_config_desc,
            # Resume Training Settings
            widgets.HTML("<h4>🔄 Resume Training Settings</h4><p>Continue from existing LoRA checkpoint.</p>"),
            self.continue_from_lora,
            # Dataset and basic settings
            self.dataset_size, self.resolution, self.num_repeats, self.epochs, self.max_train_steps, self.train_batch_size, self.seed, self.step_calculator,
            self.flip_aug, self.shuffle_caption, self.keep_tokens, self.clip_skip,
            # Caption dropout (moved from advanced)
            self.caption_dropout_rate, self.caption_tag_dropout_rate,
            # Advanced caption controls
            self.caption_dropout_every_n_epochs, self.keep_tokens_separator, self.secondary_separator, self.enable_wildcard,
            # Learning rate settings
            self.unet_lr, self.text_encoder_lr, self.lr_scheduler, self.lr_scheduler_number, self.lr_warmup_ratio, self.lr_warmup_steps, self.lr_power,
            # Noise & stability (moved from advanced)
            self.min_snr_gamma_enabled, self.min_snr_gamma, self.ip_noise_gamma_enabled, self.ip_noise_gamma,
            self.multinoise, self.multires_noise_discount, self.noise_offset, self.adaptive_noise_scale,
            # Loss configuration - HUBER LOSS DELETED (was corrupting files)
            # widgets.HTML("<h4>📉 Loss Configuration</h4><p>Loss function settings for training stability.</p>"),
            # self.loss_type, self.huber_c, self.huber_schedule,
            # LoRA Structure (merged from separate section)
            widgets.HTML("<h4>🧩 LoRA Structure</h4><p><strong>16 dim/8 alpha</strong> balances capacity and stability (~100MB). Conv layers help with textures/details.</p>"),
            self.lora_type, self.network_dim, self.network_alpha, self.dim_from_weights, self.network_dropout, 
            self.conv_dim, self.conv_alpha, self.factor, self.train_norm,
            # LyCORIS dropout parameters
            self.rank_dropout, self.module_dropout,
            self.down_lr_weight, self.mid_lr_weight, self.up_lr_weight, self.block_lr_zero_threshold,
            self.block_dims, self.block_alphas, self.conv_block_dims, self.conv_block_alphas,
            # Training options
            widgets.HTML("<h4>🛠️ Training Options</h4><p>Optimizer, precision, and memory settings.</p>"),
            self.optimizer, self.weight_decay, self.optimizer_args, self.cross_attention, self.precision, self.fp8_base,
            self.cache_latents, self.cache_latents_to_disk, self.cache_text_encoder_outputs,
            self.v2, self.v_parameterization, self.network_train_unet_only, self.zero_terminal_snr, self.enable_bucket,
            self.gradient_checkpointing, self.gradient_accumulation_steps, self.max_grad_norm, self.full_fp16, self.random_crop,
            self.color_aug, self.weighted_captions, self.no_token_padding, self.max_token_length, self.prior_loss_weight,
            self.persistent_data_loader_workers,
            # Bucketing and VAE settings
            self.sdxl_bucket_optimization, self.min_bucket_reso, self.max_bucket_reso, self.bucket_no_upscale,
            self.vae_batch_size, self.no_half_vae,
            self.save_every_n_epochs, self.save_every_n_steps, self.keep_only_last_n_epochs, 
            self.save_state, self.save_last_n_steps_state,
            # Config warnings at the end
            self.config_warnings
        ])

        # Advanced sections removed - all moved to unified training config above

        # --- Accordion (simplified: just project settings and unified training config) ---
        accordion = widgets.Accordion(children=[
            project_box,
            training_config_box
        ])
        accordion.set_title(0, "▶️ Project Settings")
        accordion.set_title(1, "▶️ Complete Training Configuration")

        # Attach observers for real-time validation (after all widgets are created)
        self.cache_text_encoder_outputs.observe(check_config_conflicts, names='value')
        self.shuffle_caption.observe(check_config_conflicts, names='value')
        self.text_encoder_lr.observe(check_config_conflicts, names='value')
        self.random_crop.observe(check_config_conflicts, names='value')
        self.cache_latents.observe(check_config_conflicts, names='value')

        # New button to prepare/commit configuration
        self.prepare_config_button = widgets.Button(
            description="✅ Prepare Training Configuration",
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )
        self.prepare_config_button.on_click(self.run_training) # This will now trigger config collection

        # Progress description (was missing!)
        progress_desc = widgets.HTML("""<h3>📊 Training Progress</h3>
        <p>Monitor your training progress and status below. The system will automatically update with real-time information.</p>""")

        # Status bar widget (was also missing!)
        self.status_bar = widgets.HTML(value="<div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px;'><strong>📊 Status:</strong> Ready to configure training</div>")

        # Training output widget
        self.training_output = widgets.Output()

        self.widget_box = widgets.VBox([
            header_main,
            accordion,
            self.prepare_config_button, # Add the new button here
            progress_desc,
            self.status_bar,
            self.training_output
        ])

    def _update_status(self, message, status_type="info"):
        """Update the status bar with current training progress"""
        status_colors = {
            "info": "#007acc",      # Blue
            "success": "#28a745",   # Green
            "warning": "#ffc107",   # Yellow
            "error": "#dc3545",     # Red
            "progress": "#17a2b8"   # Teal
        }
        color = status_colors.get(status_type, "#007acc")

        self.status_bar.value = f"<div style='padding: 10px; border: 1px solid {color}; border-radius: 5px;'><strong>📊 Status:</strong> {message}</div>"

    def run_training(self, b):
        # Check for overwrite before proceeding
        project_name = self.project_name.value.strip()
        if not project_name:
            self._update_status("❌ Please enter a project name first!", "error")
            return

        # Check if output file already exists
        output_path = os.path.join("output", f"{project_name}.safetensors")
        if os.path.exists(output_path):
            # Show overwrite warning
            self._show_overwrite_warning(output_path)
            return

        # Proceed with training preparation
        self._proceed_with_training()

    def _show_overwrite_warning(self, output_path):
        """Show user-friendly overwrite warning with file information"""
        try:
            # Get file information
            import time
            file_stats = os.stat(output_path)
            file_size_mb = file_stats.st_size / (1024 * 1024)
            file_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(file_stats.st_mtime))

            # Create warning widgets
            warning_title = widgets.HTML(
                "<h3 style='color: #ff6b35;'>⚠️ File Already Exists!</h3>"
            )

            file_info = widgets.HTML(f"""
                <div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <strong>📁 File:</strong> {os.path.basename(output_path)}<br>
                    <strong>📅 Created:</strong> {file_date}<br>
                    <strong>📊 Size:</strong> {file_size_mb:.1f} MB<br>
                    <strong>📍 Location:</strong> {output_path}
                </div>
            """)

            warning_message = widgets.HTML("""
                <p style='color: #856404; font-size: 14px;'>
                    <strong>This will permanently replace your existing LoRA!</strong><br>
                    Make sure this is what you want to do. Consider using a different project name if you want to keep both files.
                </p>
            """)

            # Confirmation checkbox (ADHD-friendly - requires explicit action)
            self.overwrite_confirm = widgets.Checkbox(
                value=False,
                description="Yes, I want to replace my existing LoRA file",
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='10px 0px')
            )

            # Suggestion for alternative name
            alternative_name = f"{self.project_name.value}_v2"
            suggestion = widgets.HTML(f"""
                <div style='background: #e7f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    <strong>💡 Suggestion:</strong> Try a different name like <code>{alternative_name}</code>
                </div>
            """)

            # Action buttons
            proceed_button = widgets.Button(
                description="🚀 Proceed with Overwrite",
                button_style='danger',
                disabled=True,  # Initially disabled
                layout=widgets.Layout(width='200px', margin='5px')
            )

            cancel_button = widgets.Button(
                description="🔙 Cancel & Choose Different Name",
                button_style='warning',
                layout=widgets.Layout(width='250px', margin='5px')
            )

            button_row = widgets.HBox([proceed_button, cancel_button])

            # Enable proceed button only when checkbox is checked
            def on_checkbox_change(change):
                proceed_button.disabled = not change['new']

            self.overwrite_confirm.observe(on_checkbox_change, names='value')

            # Button click handlers
            def on_proceed_click(b):
                self._clear_overwrite_warning()
                self._proceed_with_training()

            def on_cancel_click(b):
                self._clear_overwrite_warning()
                self._update_status("💡 Training cancelled. Please choose a different project name.", "info")

            proceed_button.on_click(on_proceed_click)
            cancel_button.on_click(on_cancel_click)

            # Create and display warning dialog
            self.overwrite_warning_box = widgets.VBox([
                warning_title,
                file_info,
                warning_message,
                self.overwrite_confirm,
                suggestion,
                button_row
            ], layout=widgets.Layout(
                border='2px solid #ff6b35',
                padding='20px',
                border_radius='10px',
                margin='20px 0px'
            ))

            # Insert warning before the status bar
            children_list = list(self.widget_box.children)
            # Find the position of accordion (should be first major component)
            insert_position = 1 if len(children_list) > 1 else 0
            children_list.insert(insert_position, self.overwrite_warning_box)
            self.widget_box.children = children_list

            self._update_status("⚠️ Please confirm overwrite above to proceed with training.", "warning")

        except Exception as e:
            print(f"Error showing overwrite warning: {e}")
            # Fallback to simple confirmation
            self._proceed_with_training()

    def _clear_overwrite_warning(self):
        """Remove the overwrite warning from the UI"""
        if hasattr(self, 'overwrite_warning_box'):
            children_list = list(self.widget_box.children)
            if self.overwrite_warning_box in children_list:
                children_list.remove(self.overwrite_warning_box)
                self.widget_box.children = children_list
            delattr(self, 'overwrite_warning_box')

    def _proceed_with_training(self):
        """Actually generate the TOML files and prepare training"""
        config = self._get_training_config()
        try:
            self.manager.prepare_config_only(config)  # Generate TOML files only
            self._update_status("✅ Configuration TOML files generated! Now click 'Start LoRA Training' below!", "success")
        except Exception as e:
            self._update_status(f"❌ Failed to generate config: {str(e)}", "error")
            return

        # Get the existing monitor if available
        try:
            import __main__
            if hasattr(__main__, 'training_monitor'):
                monitor = __main__.training_monitor
            else:
                # Fallback: create and display monitor
                # The TrainingMonitorWidget now needs the TrainingManager instance
                monitor = TrainingMonitorWidget(training_manager_instance=self.manager)
                __main__.training_monitor = monitor  # Store globally
                display(monitor.widget_box)
        except Exception as e:
            # Fallback: create and display monitor
            print(f"Error getting/creating monitor: {e}")
            monitor = TrainingMonitorWidget(training_manager_instance=self.manager)
            display(monitor.widget_box)

        # Pass the training configuration to the monitor
        config = self._get_training_config()
        monitor.set_training_config(config)

        with self.training_output:
            self.training_output.clear_output()
        # The monitor's start button will now trigger the training via its own manager reference

    def _get_training_config(self):
        """Helper method to gather all config settings from widget values"""
        return {
            'project_name': self.project_name.value,
            'model_type': self.model_type.value.lower().replace('/', '_').replace('.', ''), # e.g. sd1.5/2.0 -> sd1_5_2_0
            'model_path': self.model_path.value,
            'vae_path': self.vae_path.value,
            'clip_l_path': self.clip_l_path.value,
            'clip_g_path': self.clip_g_path.value,
            't5xxl_path': self.t5xxl_path.value,
            'dataset_path': self.dataset_dir.value,
            'continue_from_lora': self.continue_from_lora.value,
            'wandb_key': self.wandb_key.value,
            'resolution': self.resolution.value,
            'num_repeats': self.num_repeats.value,
            'epochs': self.epochs.value,
            'max_train_steps': self.max_train_steps.value,
            'train_batch_size': self.train_batch_size.value,
            'seed': self.seed.value,
            'flip_aug': self.flip_aug.value,
            'unet_lr': self._parse_learning_rate(self.unet_lr.value, self.unet_lr),
            'text_encoder_lr': self._parse_learning_rate(self.text_encoder_lr.value, self.text_encoder_lr),
            'lr_scheduler': self.lr_scheduler.value,
            'lr_scheduler_number': self.lr_scheduler_number.value,
            'lr_warmup_ratio': self.lr_warmup_ratio.value,
            'lr_warmup_steps': self.lr_warmup_steps.value,
            'lr_power': self.lr_power.value,
            'min_snr_gamma_enabled': self.min_snr_gamma_enabled.value,
            'min_snr_gamma': self.min_snr_gamma.value,
            'ip_noise_gamma_enabled': self.ip_noise_gamma_enabled.value,
            'ip_noise_gamma': self.ip_noise_gamma.value,
            'multinoise': self.multinoise.value,
            'multires_noise_discount': self.multires_noise_discount.value,
            'lora_type': self.lora_type.value,
            'network_dim': self.network_dim.value,
            'network_alpha': self.network_alpha.value,
            'dim_from_weights': self.dim_from_weights.value,
            'network_dropout': self.network_dropout.value,
            'conv_dim': self.conv_dim.value,
            'conv_alpha': self.conv_alpha.value,
            'rank_dropout': self.rank_dropout.value,
            'module_dropout': self.module_dropout.value,
            'factor': self.factor.value,
            'train_norm': self.train_norm.value,
            # Advanced LoRA configuration
            'down_lr_weight': self.down_lr_weight.value,
            'mid_lr_weight': self.mid_lr_weight.value,
            'up_lr_weight': self.up_lr_weight.value,
            'block_lr_zero_threshold': self.block_lr_zero_threshold.value,
            'block_dims': self.block_dims.value,
            'block_alphas': self.block_alphas.value,
            'conv_block_dims': self.conv_block_dims.value,
            'conv_block_alphas': self.conv_block_alphas.value,
            'sample_prompt': self.sample_prompt.value,
            'sample_num_images': self.sample_num_images.value,
            'sample_resolution': self.sample_resolution.value,
            'sample_seed': self.sample_seed.value,
            'optimizer': self.optimizer.value,
            'weight_decay': self.weight_decay.value,
            'optimizer_args': self.optimizer_args.value,
            'cross_attention': self.cross_attention.value,
            'precision': self.precision.value,
            'fp8_base': self.fp8_base.value,
            'cache_latents': self.cache_latents.value,
            'cache_latents_to_disk': self.cache_latents_to_disk.value,
            'cache_text_encoder_outputs': self.cache_text_encoder_outputs.value,
            'shuffle_caption': self.shuffle_caption.value,
            'v2': self.v2.value,
            'v_parameterization': self.v_parameterization.value,
            'network_train_unet_only': self.network_train_unet_only.value,
            'save_every_n_epochs': self.save_every_n_epochs.value,
            'save_every_n_steps': self.save_every_n_steps.value,
            'keep_only_last_n_epochs': self.keep_only_last_n_epochs.value,
            'save_state': self.save_state.value,
            'save_last_n_steps_state': self.save_last_n_steps_state.value,
            'max_token_length': self.max_token_length.value,
            'prior_loss_weight': self.prior_loss_weight.value,
            'color_aug': self.color_aug.value,
            'persistent_data_loader_workers': self.persistent_data_loader_workers.value,
            'no_token_padding': self.no_token_padding.value,
            'weighted_captions': self.weighted_captions.value,
            # Advanced training options
            'caption_dropout_rate': self.caption_dropout_rate.value,
            'caption_tag_dropout_rate': self.caption_tag_dropout_rate.value,
            'caption_dropout_every_n_epochs': self.caption_dropout_every_n_epochs.value if self.caption_dropout_every_n_epochs.value > 0 else None,
            'keep_tokens_separator': self.keep_tokens_separator.value or None,
            'secondary_separator': self.secondary_separator.value or None,
            'enable_wildcard': self.enable_wildcard.value,
            'keep_tokens': self.keep_tokens.value,
            'noise_offset': self.noise_offset.value,
            'adaptive_noise_scale': self.adaptive_noise_scale.value,
            'zero_terminal_snr': self.zero_terminal_snr.value,
            'clip_skip': self.clip_skip.value,
            'enable_bucket': self.enable_bucket.value,
            'gradient_checkpointing': self.gradient_checkpointing.value,
            'gradient_accumulation_steps': self.gradient_accumulation_steps.value,
            'max_grad_norm': self.max_grad_norm.value,
            'full_fp16': self.full_fp16.value,
            'random_crop': self.random_crop.value,
            'bucket_reso_steps': 32 if self.sdxl_bucket_optimization.value else 64,
            'min_bucket_reso': self.min_bucket_reso.value,
            'max_bucket_reso': self.max_bucket_reso.value,
            'bucket_no_upscale': self.bucket_no_upscale.value,
            'vae_batch_size': self.vae_batch_size.value,
            'no_half_vae': self.no_half_vae.value,
            # Advanced options
            'advanced_mode_enabled': getattr(self, 'advanced_mode', widgets.Checkbox(value=False)).value,
            'advanced_optimizer': getattr(self, 'advanced_optimizer', type('obj', (object,), {'value': 'standard'})).value,
            'advanced_scheduler': getattr(self, 'advanced_scheduler', type('obj', (object,), {'value': 'auto'})).value,
            'fused_back_pass': getattr(self, 'fused_back_pass', widgets.Checkbox(value=False)).value,
            'lycoris_method': getattr(self, 'lycoris_method', type('obj', (object,), {'value': 'none'})).value,
            # Loss configuration
            # HUBER LOSS DELETED - was corrupting safetensors files 
            # 'loss_type': self.loss_type.value,
            # 'huber_c': self.huber_c.value,
            # 'huber_schedule': self.huber_schedule.value,
        }

    def _convert_to_structured_config(self, flat_config):
        """Convert flat widget config to structured TOML format"""
        
        # 🔍 DEBUG: What's actually in the flat config?
        print("🚨 === FLAT CONFIG DEBUG (BEFORE CONVERSION) ===")
        print(f"📊 All flat config keys: {list(flat_config.keys())}")
        print(f"📊 network_dim: {repr(flat_config.get('network_dim'))}")
        print(f"📊 network_alpha: {repr(flat_config.get('network_alpha'))}")
        print(f"📊 unet_lr: {repr(flat_config.get('unet_lr'))}")
        print(f"📊 model_path: {repr(flat_config.get('model_path'))}")
        print(f"📊 epochs: {repr(flat_config.get('epochs'))}")
        print(f"📊 optimizer: {repr(flat_config.get('optimizer'))}")
        print("🚨 === END FLAT CONFIG DEBUG ===")
        
        # Handle resolution formatting like our KohyaTrainingManager does
        resolution = flat_config.get('resolution')
        if isinstance(resolution, (int, str)):
            formatted_resolution = f"{resolution},{resolution}"
        else:
            formatted_resolution = "512,512"

        # Determine network module based on LoRA type
        lora_type = flat_config.get('lora_type', 'LoRA')
        if lora_type in ['LoCon', 'LoKR', 'DyLoRA', 'DoRA (Weight Decomposition)', 
                        'LoHa (Hadamard Product)', '(IA)³ (Few Parameters)', 
                        'GLoRA (Generalized LoRA)', 'GLoKr (Generalized LoKR)',
                        'Native Fine-Tuning (Full)', 'Diag-OFT (Orthogonal Fine-Tuning)',
                        'BOFT (Butterfly Transform)']:
            network_module = "lycoris.kohya"
        else:
            network_module = "networks.lora"

        # Build network arguments (clean None values)
        network_args = {
            "network_dim": flat_config.get('network_dim'),
            "network_alpha": flat_config.get('network_alpha'),
            "network_module": network_module,
        }
        
        # Add new network parameters if specified
        if flat_config.get('dim_from_weights'):
            network_args["dim_from_weights"] = flat_config.get('dim_from_weights')
        if flat_config.get('network_dropout', 0.0) > 0.0:
            network_args["network_dropout"] = flat_config.get('network_dropout')
        
        # Add conv layers if specified
        if flat_config.get('conv_dim'):
            network_args["conv_dim"] = flat_config.get('conv_dim')
        if flat_config.get('conv_alpha'):
            network_args["conv_alpha"] = flat_config.get('conv_alpha')
            
        # Add advanced LoRA configuration (only if not empty)
        for param in ['down_lr_weight', 'mid_lr_weight', 'up_lr_weight', 'block_lr_zero_threshold',
                     'block_dims', 'block_alphas', 'conv_block_dims', 'conv_block_alphas']:
            value = flat_config.get(param)
            if value and str(value).strip():
                network_args[param] = value

        # Build optimizer arguments
        optimizer_args = {
            "learning_rate": flat_config.get('unet_lr'),
            "text_encoder_lr": flat_config.get('text_encoder_lr'),
            "lr_scheduler": flat_config.get('lr_scheduler'),
            "optimizer_type": flat_config.get('optimizer'),
            "max_grad_norm": flat_config.get('max_grad_norm'),
        }
        
        # Add optional optimizer settings (only if not default/empty)
        if flat_config.get('lr_scheduler_number', 3) != 3:
            optimizer_args["lr_scheduler_num_cycles"] = flat_config.get('lr_scheduler_number')
        if flat_config.get('lr_warmup_ratio') and float(flat_config.get('lr_warmup_ratio', 0)) > 0:
            optimizer_args["lr_warmup_ratio"] = flat_config.get('lr_warmup_ratio')
        if flat_config.get('lr_warmup_steps', 0) > 0:
            optimizer_args["lr_warmup_steps"] = flat_config.get('lr_warmup_steps')
        if flat_config.get('lr_power') and flat_config.get('lr_power') != '1.0':
            optimizer_args["lr_power"] = flat_config.get('lr_power')
        if flat_config.get('weight_decay') and flat_config.get('weight_decay') != '0.01':
            optimizer_args["weight_decay"] = flat_config.get('weight_decay')

        # Add network_args section for LyCORIS
        network_args_section = {}
        if network_module == "lycoris.kohya":
            # Map LoRA type to algo
            lora_algo_map = {
                'LoCon': 'locon',
                'LoKR': 'lokr', 
                'DyLoRA': 'dylora',
                'DoRA (Weight Decomposition)': 'dora',
                'LoHa (Hadamard Product)': 'loha',
                '(IA)³ (Few Parameters)': 'ia3',
                'GLoRA (Generalized LoRA)': 'glora',
                'Native Fine-Tuning (Full)': 'full',
                'Diag-OFT (Orthogonal Fine-Tuning)': 'diag-oft',
                'BOFT (Butterfly Transform)': 'boft'
            }
            if lora_type in lora_algo_map:
                network_args_section["algo"] = lora_algo_map[lora_type]
            
            # Add LyCORIS advanced parameters
            if flat_config.get('factor', -1) != -1:
                network_args_section["factor"] = flat_config.get('factor')
            if flat_config.get('train_norm'):
                network_args_section["train_norm"] = True
        
        # Build optimizer_args section from optimizer_args string
        optimizer_args_section = {}
        if flat_config.get('optimizer_args') and flat_config.get('optimizer_args').strip():
            # Parse "betas=0.9,0.99,0.999" format
            args_str = flat_config.get('optimizer_args')
            if 'betas=' in args_str:
                betas_part = args_str.split('betas=')[1].split(',')[0:3]  # Take up to 3 values
                if len(betas_part) >= 2:
                    optimizer_args_section["betas"] = f"{betas_part[0]},{betas_part[1]}" + (f",{betas_part[2]}" if len(betas_part) > 2 else "")
        
        # HUBER LOSS SECTION DELETED - was corrupting safetensors files during training
        # loss_arguments_section = {}
        # if flat_config.get('loss_type') and flat_config.get('loss_type') != 'l2':
        #     loss_arguments_section["loss_type"] = flat_config.get('loss_type')
        #     if flat_config.get('loss_type') == 'huber':
        #         if flat_config.get('huber_c'):
        #             loss_arguments_section["huber_c"] = flat_config.get('huber_c')
        #         if flat_config.get('huber_schedule'):
        #             loss_arguments_section["huber_schedule"] = flat_config.get('huber_schedule')

        structured_config = {
            "network_arguments": network_args,
            "optimizer_arguments": optimizer_args,
            "training_arguments": {
                "pretrained_model_name_or_path": flat_config.get('model_path'),
                "vae": flat_config.get('vae_path') if flat_config.get('vae_path') else None,
                "max_train_epochs": flat_config.get('epochs'),
                "max_train_steps": flat_config.get('max_train_steps') if flat_config.get('max_train_steps', 0) > 0 else None,
                "train_batch_size": flat_config.get('train_batch_size'),
                "save_every_n_epochs": flat_config.get('save_every_n_epochs'),
                "save_every_n_steps": flat_config.get('save_every_n_steps') if flat_config.get('save_every_n_steps', 0) > 0 else None,
                "keep_only_last_n_epochs": flat_config.get('keep_only_last_n_epochs'),
                "save_state": flat_config.get('save_state') if flat_config.get('save_state') else None,
                "save_last_n_steps_state": flat_config.get('save_last_n_steps_state') if flat_config.get('save_last_n_steps_state', 0) > 0 else None,
                "mixed_precision": flat_config.get('precision'),
                "output_dir": "output",
                "output_name": flat_config.get('project_name', 'lora'),
                "clip_skip": flat_config.get('clip_skip', 2),
                "save_model_as": "safetensors",
                "seed": flat_config.get('seed', 42),
                # New parameters from widgets
                "max_token_length": flat_config.get('max_token_length') if flat_config.get('max_token_length', 75) != 75 else None,
                "prior_loss_weight": flat_config.get('prior_loss_weight') if flat_config.get('prior_loss_weight', '1.0') != '1.0' else None,
                "persistent_data_loader_workers": flat_config.get('persistent_data_loader_workers') if flat_config.get('persistent_data_loader_workers', 0) > 0 else None,
                "no_token_padding": flat_config.get('no_token_padding') if flat_config.get('no_token_padding') else None,
                "weighted_captions": flat_config.get('weighted_captions') if flat_config.get('weighted_captions') else None,
                "color_aug": flat_config.get('color_aug') if flat_config.get('color_aug') else None,
                # Add critical missing training settings
                "gradient_checkpointing": flat_config.get('gradient_checkpointing'),
                "gradient_accumulation_steps": flat_config.get('gradient_accumulation_steps'),
                "cache_latents": flat_config.get('cache_latents'),
                "cache_latents_to_disk": flat_config.get('cache_latents_to_disk'),
                "cache_text_encoder_outputs": flat_config.get('cache_text_encoder_outputs'),
                # V-parameterization and model variant settings
                "v2": flat_config.get('v2'),
                "v_parameterization": flat_config.get('v_parameterization'),
                "zero_terminal_snr": flat_config.get('zero_terminal_snr'),
                # Cross attention and precision settings
                "xformers": flat_config.get('cross_attention') == 'xformers',
                "sdpa": flat_config.get('cross_attention') == 'sdpa',
                "fp8_base": flat_config.get('fp8_base'),
                "full_fp16": flat_config.get('full_fp16'),
                # Noise and training stability
                "noise_offset": flat_config.get('noise_offset'),
                "min_snr_gamma": flat_config.get('min_snr_gamma') if flat_config.get('min_snr_gamma_enabled') else None,
                "ip_noise_gamma": flat_config.get('ip_noise_gamma') if flat_config.get('ip_noise_gamma_enabled') else None,
                "multires_noise_iterations": 6 if flat_config.get('multinoise') else None,
                "adaptive_noise_scale": flat_config.get('adaptive_noise_scale') if flat_config.get('adaptive_noise_scale', 0) > 0 else None,
                # Caption handling
                "caption_dropout_rate": flat_config.get('caption_dropout_rate'),
                "caption_tag_dropout_rate": flat_config.get('caption_tag_dropout_rate'),
                "keep_tokens": flat_config.get('keep_tokens'),
                # SDXL specific optimizations
                "network_train_unet_only": flat_config.get('network_train_unet_only'),
                # VAE settings
                "vae_batch_size": flat_config.get('vae_batch_size'),
                "no_half_vae": flat_config.get('no_half_vae'),
                # Data augmentation
                "random_crop": flat_config.get('random_crop'),
            },
            "datasets": [{
                "subsets": [{
                    "image_dir": flat_config.get('dataset_path'),
                    "num_repeats": flat_config.get('num_repeats'),
                }]
            }],
            "general": {
                "resolution": formatted_resolution,
                "shuffle_caption": flat_config.get('shuffle_caption'),
                "flip_aug": flat_config.get('flip_aug'),
                "caption_extension": ".txt",
                # Bucketing settings
                "enable_bucket": flat_config.get('enable_bucket'),
                "bucket_no_upscale": flat_config.get('bucket_no_upscale'),
                "bucket_reso_steps": flat_config.get('bucket_reso_steps', 64),
                "min_bucket_reso": flat_config.get('min_bucket_reso'),
                "max_bucket_reso": flat_config.get('max_bucket_reso'),
            },
        }
        
        # Add optional sections only if they have content
        if network_args_section:
            structured_config["network_args"] = network_args_section
        if optimizer_args_section:
            structured_config["optimizer_args"] = optimizer_args_section  
        # HUBER LOSS SECTION DELETED - was corrupting safetensors files
        # if loss_arguments_section:
        #     structured_config["loss_arguments"] = loss_arguments_section
        
        # Include widget-only fields for monitor widget compatibility
        structured_config.update({
            "sample_prompt": flat_config.get('sample_prompt'),
            "sample_num_images": flat_config.get('sample_num_images'),
            "sample_resolution": flat_config.get('sample_resolution'),
            "sample_seed": flat_config.get('sample_seed'),
            "dataset_size": flat_config.get('dataset_size'),  # For step calculation
            "epochs": flat_config.get('epochs'),  # For total epochs calculation
        })
        
        # Clean None values from all sections
        def clean_none_values(obj):
            if isinstance(obj, dict):
                return {k: clean_none_values(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [clean_none_values(item) for item in obj if item is not None]
            return obj
            
        structured_config = clean_none_values(structured_config)
        return structured_config

    # Advanced sections removed - functionality simplified

    def _refresh_model_list(self):
        """Scan pretrained_model directory and populate dropdown"""
        try:
            import glob

            # Look for models in common locations
            search_paths = [
                "pretrained_model/*.safetensors",
                "pretrained_model/*.ckpt",
                "pretrained_model/*.pth",
                "models/*.safetensors",
                "models/*.ckpt",
                "*/pretrained_model/*.safetensors",  # Check subdirectories
            ]

            found_models = []
            for pattern in search_paths:
                found_models.extend(glob.glob(pattern))

            # Remove duplicates and sort
            found_models = sorted(list(set(found_models)))

            if found_models:
                # Create dropdown options with friendly names
                options = [('Select a model...', '')]
                for model_path in found_models:
                    model_name = os.path.basename(model_path)
                    # Truncate long names for dropdown display
                    display_name = model_name if len(model_name) <= 50 else model_name[:47] + "..."
                    options.append((display_name, model_path))

                self.model_dropdown.options = options
                print(f"✅ Found {len(found_models)} models in pretrained_model directory")
            else:
                self.model_dropdown.options = [('No models found - use custom path below', '')]
                print("📁 No models found in pretrained_model/ directory")
                print("💡 Place your .safetensors/.ckpt files in pretrained_model/ folder or use custom path")

        except Exception as e:
            print(f"⚠️ Error scanning for models: {e}")
            self.model_dropdown.options = [('Error scanning - use custom path', '')]

    def display(self):
        display(self.widget_box)

        # Make this training widget globally available for the monitor widget
        import __main__
        __main__.training_widget = self
