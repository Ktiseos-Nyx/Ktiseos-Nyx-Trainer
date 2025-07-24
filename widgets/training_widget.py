# widgets/training_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.training_manager import TrainingManager

class TrainingWidget:
    def __init__(self):
        self.manager = TrainingManager()
        self.create_widgets()
    
    def _parse_learning_rate(self, lr_text):
        """Parse learning rate from text input supporting both scientific notation and decimals"""
        try:
            # Handle both scientific notation (5e-4) and decimal (0.0005)
            return float(lr_text)
        except ValueError:
            print(f"⚠️ Invalid learning rate format: {lr_text}. Using default 1e-4")
            return 1e-4

    def create_widgets(self):
        header_icon = "⭐"
        header_main = widgets.HTML(f"<h2>{header_icon} 3. Training Configuration</h2>")

        # --- Project Settings ---
        project_desc = widgets.HTML("<h3>▶️ Project Settings</h3><p>Define your project name, the path to your base model, and your dataset directory. You can also specify an existing LoRA to continue training from and your Weights & Biases API key for logging.</p>")
        self.project_name = widgets.Text(description="Project Name:", placeholder="e.g., my-awesome-lora (no spaces or special characters)", layout=widgets.Layout(width='99%'))
        self.model_path = widgets.Text(description="Model Path:", placeholder="Absolute path to your base model (e.g., /path/to/model.safetensors)", layout=widgets.Layout(width='99%'))
        self.dataset_dir = widgets.Text(description="Dataset Dir:", placeholder="Absolute path to your dataset directory (e.g., /path/to/my_dataset)", layout=widgets.Layout(width='99%'))
        self.continue_from_lora = widgets.Text(description="Continue from LoRA:", placeholder="Absolute path to an existing LoRA to continue training (optional)", layout=widgets.Layout(width='99%'))
        self.wandb_key = widgets.Text(description="WandB API Key:", placeholder="Your Weights & Biases API key (optional, for logging)", layout=widgets.Layout(width='99%'))
        project_box = widgets.VBox([project_desc, self.project_name, self.model_path, self.dataset_dir, self.continue_from_lora, self.wandb_key])

        # --- Basic Settings ---
        basic_desc = widgets.HTML("""<h3>▶️ Basic Settings</h3>
        <p>Configure fundamental training parameters. <strong>Target 250-1000 total steps</strong> using this formula:</p>
        <p><code>Images × Repeats × Epochs ÷ Batch Size = Total Steps</code></p>
        <p><strong>Examples:</strong><br>
        • 10 images × 10 repeats × 10 epochs ÷ 2 batch = 500 steps<br>
        • 20 images × 5 repeats × 10 epochs ÷ 4 batch = 250 steps<br>
        • 100 images × 1 repeat × 10 epochs ÷ 4 batch = 250 steps</p>""")
        self.resolution = widgets.IntText(value=1024, description='Resolution:', style={'description_width': 'initial'})
        self.num_repeats = widgets.IntText(value=10, description='Num Repeats:', style={'description_width': 'initial'})
        self.epochs = widgets.IntText(value=10, description='Epochs:', style={'description_width': 'initial'})
        self.train_batch_size = widgets.IntText(value=4, description='Train Batch Size:', style={'description_width': 'initial'})
        self.flip_aug = widgets.Checkbox(value=False, description="Flip Augmentation (recommended for small datasets)", indent=False)
        self.shuffle_caption = widgets.Checkbox(value=True, description="Shuffle Captions (improves variety, incompatible with text encoder caching)", indent=False)
        
        # Live step calculator
        self.dataset_size = widgets.IntText(value=20, description="Dataset Size:", style={'description_width': 'initial'})
        self.step_calculator = widgets.HTML()
        
        def update_step_calculation(*args):
            images = self.dataset_size.value
            repeats = self.num_repeats.value 
            epochs = self.epochs.value
            batch_size = self.train_batch_size.value
            
            if batch_size > 0:
                total_steps = (images * repeats * epochs) // batch_size
                
                if total_steps < 250:
                    color = "#dc3545"  # Red
                    status = "⚠️ Too few steps - may be undercooked"
                elif total_steps <= 3000:
                    color = "#28a745"  # Green
                    status = "✅ Good step count"
                elif total_steps <= 5000:
                    color = "#ff8c00"  # Orange
                    status = "🟡 High steps - should work fine for SDXL"
                else:
                    color = "#ffc107"  # Yellow  
                    status = "🔴 Very high steps - consider reducing repeats"
                
                self.step_calculator.value = f"""
                <div style='background: {color}20; padding: 10px; border-left: 4px solid {color}; margin: 5px 0;'>
                <strong>📊 Total Steps: {total_steps}</strong><br>
                {images} images × {repeats} repeats × {epochs} epochs ÷ {batch_size} batch = {total_steps} steps<br>
                <em>{status}</em>
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
            warnings = []
            
            # Check text encoder caching vs shuffle caption conflict
            if self.cache_text_encoder_outputs.value and self.shuffle_caption.value:
                warnings.append("⚠️ Cannot use Caption Shuffling with Text Encoder Caching")
            
            # Check text encoder caching vs text encoder training conflict  
            if self.cache_text_encoder_outputs.value and float(self.text_encoder_lr.value) > 0:
                warnings.append("⚠️ Cannot cache Text Encoder while training it (set Text LR to 0)")
                
            if warnings:
                warning_html = "<div style='background: #fff3cd; padding: 10px; border-left: 4px solid #856404; margin: 5px 0;'>"
                warning_html += "<br>".join(warnings)
                warning_html += "<br><em>💡 Fix these conflicts to enable training</em></div>"
                self.config_warnings.value = warning_html
            else:
                self.config_warnings.value = ""
        
        # Note: Observers will be attached after all widgets are created
        
        basic_box = widgets.VBox([basic_desc, self.dataset_size, self.resolution, self.num_repeats, self.epochs, self.train_batch_size, self.step_calculator, self.flip_aug, self.shuffle_caption, self.config_warnings])

        # --- Learning Rate ---
        lr_desc = widgets.HTML("<h3>▶️ Learning Rate</h3><p>Adjust learning rates for the UNet and Text Encoder. Experiment with different schedulers and warmup ratios. Min SNR Gamma and IP Noise Gamma can improve results.</p>")
        self.unet_lr = widgets.Text(value='5e-4', placeholder='e.g., 5e-4 or 0.0005', description='🧠 UNet LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.text_encoder_lr = widgets.Text(value='1e-4', placeholder='e.g., 1e-4 or 0.0001', description='📝 Text LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.lr_scheduler = widgets.Dropdown(options=['cosine', 'cosine_with_restarts', 'constant', 'linear', 'polynomial', 'rex'], value='cosine', description='Scheduler:', style={'description_width': 'initial'})
        self.lr_scheduler_number = widgets.IntSlider(value=3, min=1, max=10, description='Scheduler Num (for restarts/polynomial):', style={'description_width': 'initial'}, continuous_update=False)
        self.lr_warmup_ratio = widgets.FloatSlider(value=0.05, min=0.0, max=0.5, step=0.01, description='Warmup Ratio:', style={'description_width': 'initial'}, continuous_update=False)
        self.min_snr_gamma_enabled = widgets.Checkbox(value=True, description="Enable Min SNR Gamma (recommended for better results)", indent=False)
        self.min_snr_gamma = widgets.FloatSlider(value=5.0, min=0.0, max=10.0, step=0.1, description='Min SNR Gamma:', style={'description_width': 'initial'}, continuous_update=False)
        self.ip_noise_gamma_enabled = widgets.Checkbox(value=False, description="Enable IP Noise Gamma", indent=False)
        self.ip_noise_gamma = widgets.FloatSlider(value=0.05, min=0.0, max=0.1, step=0.01, description='IP Noise Gamma:', style={'description_width': 'initial'}, continuous_update=False)
        self.multinoise = widgets.Checkbox(value=False, description="Multi-noise (can help with color balance)", indent=False)
        learning_box = widgets.VBox([
            lr_desc, self.unet_lr, self.text_encoder_lr, self.lr_scheduler, self.lr_scheduler_number, self.lr_warmup_ratio,
            self.min_snr_gamma_enabled, self.min_snr_gamma, self.ip_noise_gamma_enabled, self.ip_noise_gamma, self.multinoise
        ])

        # --- LoRA Structure ---
        lora_struct_desc = widgets.HTML("""
        <h3>▶️ LoRA Structure</h3>
        <p>Choose your LoRA type and define its dimensions. <strong>8 dim/4 alpha works great for characters (~50MB)</strong>. Higher dimensions capture more detail but create larger files.</p>
        
        <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <h4>🧩 Conv Layers Explained:</h4>
        <p><strong>Conv Dim/Alpha</strong> control additional convolutional learning layers (for LoCon, LoKR, DyLoRA, etc.):</p>
        <ul>
        <li><strong>Conv layers help with:</strong> Fine details, textures, spatial features, artistic styles</li>
        <li><strong>When to use:</strong> Style LoRAs, complex characters, detailed concepts</li>
        <li><strong>Recommended:</strong> Start with same values as Network Dim/Alpha (8/4)</li>
        <li><strong>Higher values:</strong> More detail capture but larger file size and slower training</li>
        <li><strong>Skip for:</strong> Simple character LoRAs where standard LoRA works fine</li>
        </ul>
        </div>
        """)
        self.lora_type = widgets.Dropdown(
            options=[
                'LoRA', 'LoCon', 'LoKR', 'DyLoRA', 
                'DoRA (Weight Decomposition)', 
                'LoHa (Hadamard Product)', 
                '(IA)³ (Few Parameters)', 
                'GLoRA (Generalized LoRA)'
            ], 
            value='LoRA', 
            description='LoRA Type:', 
            style={'description_width': 'initial'}
        )
        self.network_dim = widgets.IntSlider(value=8, min=1, max=128, step=1, description='Network Dim:', style={'description_width': 'initial'}, continuous_update=False)
        self.network_alpha = widgets.IntSlider(value=4, min=1, max=128, step=1, description='Network Alpha:', style={'description_width': 'initial'}, continuous_update=False)
        self.conv_dim = widgets.IntSlider(value=8, min=1, max=128, step=1, description='🧩 Conv Dim (for textures/details):', style={'description_width': 'initial'}, continuous_update=False)
        self.conv_alpha = widgets.IntSlider(value=4, min=1, max=128, step=1, description='🧩 Conv Alpha (conv learning rate):', style={'description_width': 'initial'}, continuous_update=False)
        lora_box = widgets.VBox([lora_struct_desc, self.lora_type, self.network_dim, self.network_alpha, self.conv_dim, self.conv_alpha])

        # --- Training Options ---
        train_opt_desc = widgets.HTML("""<h3>▶️ Training Options</h3>
        <p>Select your optimizer, cross-attention mechanism, and precision. Caching latents can save memory. Enable V-Parameterization for SDXL v-pred models.</p>
        
        <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #856404;'>
        <strong>⚠️ Optimizer Compatibility Notes:</strong><br>
        • <strong>AdamW:</strong> Safe, reliable, works everywhere (recommended)<br>
        • <strong>AdamW8bit:</strong> May have compatibility issues with some environments<br>
        • <strong>Prodigy:</strong> Adaptive learning rate, very good results<br>
        • <strong>Lion:</strong> Fast and memory efficient<br>
        <em>If you get import errors, try switching to AdamW or Prodigy</em>
        </div>""")
        self.optimizer = widgets.Dropdown(options=['AdamW', 'AdamW8bit', 'Prodigy', 'DAdaptation', 'DadaptAdam', 'DadaptLion', 'Lion', 'SGDNesterov', 'SGDNesterov8bit', 'AdaFactor', 'Came'], value='AdamW', description='Optimizer:', style={'description_width': 'initial'})
        self.cross_attention = widgets.Dropdown(options=['sdpa', 'xformers'], value='sdpa', description='Cross Attention:', style={'description_width': 'initial'})
        self.precision = widgets.Dropdown(options=['fp16', 'bf16', 'float'], value='fp16', description='Precision:', style={'description_width': 'initial'})
        self.cache_latents = widgets.Checkbox(value=True, description="Cache Latents (saves memory)", indent=False)
        self.cache_latents_to_disk = widgets.Checkbox(value=True, description="Cache Latents to Disk (uses disk space, saves more memory)", indent=False)
        self.cache_text_encoder_outputs = widgets.Checkbox(value=False, description="Cache Text Encoder Outputs (disables text encoder training)", indent=False)
        self.v_parameterization = widgets.Checkbox(value=False, description="V-Parameterization (enable for SDXL v-pred models)", indent=False)
        training_options_box = widgets.VBox([
            train_opt_desc, self.optimizer, self.cross_attention, self.precision, 
            self.cache_latents, self.cache_latents_to_disk, self.cache_text_encoder_outputs,
            self.v_parameterization
        ])

        # --- Advanced Training Options ---
        advanced_train_desc = widgets.HTML("""
        <h3>▶️ Advanced Training Options</h3>
        <p>Fine-tune caption handling, noise settings, and training stability options. These control how the model learns from your dataset.</p>
        
        <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <h4>📚 Caption Controls Explained:</h4>
        <ul>
        <li><strong>Caption Dropout:</strong> Randomly removes entire captions to improve unconditional generation</li>
        <li><strong>Tag Dropout:</strong> Randomly removes individual tags to prevent overfitting to specific combinations</li>
        <li><strong>Keep Tokens:</strong> Always keeps the first N tokens (useful for trigger words)</li>
        </ul>
        
        <h4>🔊 Noise & Stability:</h4>
        <ul>
        <li><strong>Noise Offset:</strong> Adds brightness variation to improve dark/light image generation</li>
        <li><strong>Zero Terminal SNR:</strong> Removes bias in noise scheduler (recommended for SDXL)</li>
        <li><strong>Clip Skip:</strong> How many CLIP layers to skip (1-2 for anime, 2 for realistic)</li>
        </ul>
        </div>
        """)
        
        # Caption dropout controls
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
        
        self.keep_tokens = widgets.IntSlider(
            value=0, min=0, max=5, step=1,
            description='Keep Tokens (preserve first N):',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # Noise and stability controls
        self.noise_offset = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.2, step=0.01,
            description='Noise Offset:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.adaptive_noise_scale = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.02, step=0.001,
            description='Adaptive Noise Scale:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.zero_terminal_snr = widgets.Checkbox(
            value=False,
            description="Zero Terminal SNR (recommended for SDXL)",
            indent=False
        )
        
        self.clip_skip = widgets.IntSlider(
            value=2, min=1, max=12, step=1,
            description='Clip Skip (1-2 anime, 2 realistic):',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # VAE and performance options
        self.vae_batch_size = widgets.IntSlider(
            value=1, min=1, max=8, step=1,
            description='VAE Batch Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.no_half_vae = widgets.Checkbox(
            value=False,
            description="No Half VAE (fixes some VAE issues, uses more VRAM)",
            indent=False
        )
        
        # Dataset bucketing controls
        self.bucket_reso_steps = widgets.IntSlider(
            value=64, min=32, max=128, step=32,
            description='Bucket Resolution Steps:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.min_bucket_reso = widgets.IntSlider(
            value=256, min=128, max=512, step=64,
            description='Min Bucket Resolution:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.max_bucket_reso = widgets.IntSlider(
            value=2048, min=1024, max=4096, step=512,
            description='Max Bucket Resolution:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.bucket_no_upscale = widgets.Checkbox(
            value=False,
            description="No Bucket Upscale (prevent upscaling small images)",
            indent=False
        )
        
        advanced_training_box = widgets.VBox([
            advanced_train_desc,
            widgets.HTML("<h4>📚 Caption Controls</h4>"),
            self.caption_dropout_rate,
            self.caption_tag_dropout_rate, 
            self.keep_tokens,
            widgets.HTML("<h4>🔊 Noise & Stability</h4>"),
            self.noise_offset,
            self.adaptive_noise_scale,
            self.zero_terminal_snr,
            self.clip_skip,
            widgets.HTML("<h4>🎨 VAE & Performance</h4>"),
            self.vae_batch_size,
            self.no_half_vae,
            widgets.HTML("<h4>📐 Dataset Bucketing</h4>"),
            self.bucket_reso_steps,
            self.min_bucket_reso,
            self.max_bucket_reso,
            self.bucket_no_upscale
        ])

        # --- Saving Options ---
        saving_desc = widgets.HTML("<h3>▶️ Saving Options</h3><p>Control how often your LoRA is saved during training and how many recent epochs to keep. Saving more frequently allows for better progress tracking.</p>")
        self.save_every_n_epochs = widgets.IntSlider(value=1, min=1, max=10, description='Save Every N Epochs:', style={'description_width': 'initial'}, continuous_update=False)
        self.keep_only_last_n_epochs = widgets.IntSlider(value=5, min=1, max=10, description='Keep Last N Epochs:', style={'description_width': 'initial'}, continuous_update=False)
        saving_box = widgets.VBox([saving_desc, self.save_every_n_epochs, self.keep_only_last_n_epochs])

        # --- Advanced Mode Section ---
        advanced_box = self._create_advanced_section()

        # --- Accordion ---
        accordion = widgets.Accordion(children=[
            project_box,
            basic_box,
            learning_box,
            lora_box,
            training_options_box,
            advanced_training_box,
            saving_box,
            advanced_box
        ])
        accordion.set_title(0, "▶️ Project Settings")
        accordion.set_title(1, "▶️ Basic Settings")
        accordion.set_title(2, "▶️ Learning Rate")
        accordion.set_title(3, "▶️ LoRA Structure")
        accordion.set_title(4, "▶️ Training Options")
        accordion.set_title(5, "▶️ Advanced Training Options")
        accordion.set_title(6, "▶️ Saving Options")
        accordion.set_title(7, "🧪 Advanced Mode (Experimental)")

        # --- Training Button ---
        self.start_button = widgets.Button(description="Start Training", button_style='success')
        
        # --- Status Summary (stays visible) ---
        self.status_bar = widgets.HTML(value="<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>📊 Status:</strong> Ready to start training</div>")
        
        # --- Detailed Training Log (scrollable) ---
        self.training_output = widgets.Output(layout=widgets.Layout(height='400px', overflow='scroll', border='1px solid #ddd', margin='10px 0'))
        
        # --- Progress Summary ---
        progress_desc = widgets.HTML("<h3>📊 Training Progress</h3><p>Status updates appear above, detailed logs below.</p>")
        
        self.start_button.on_click(self.run_training)

        # Attach observers for real-time validation (after all widgets are created)
        self.cache_text_encoder_outputs.observe(check_config_conflicts, names='value')
        self.shuffle_caption.observe(check_config_conflicts, names='value')
        self.text_encoder_lr.observe(check_config_conflicts, names='value')

        self.widget_box = widgets.VBox([
            header_main, 
            accordion, 
            self.start_button,
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
        
        self.status_bar.value = f"<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid {color};'><strong>📊 Status:</strong> {message}</div>"

    def run_training(self, b):
        self._update_status("Preparing training configuration...", "info")
        
        with self.training_output:
            self.training_output.clear_output()
            # Gather all the settings
            config = {
                'project_name': self.project_name.value,
                'model_path': self.model_path.value,
                'dataset_dir': self.dataset_dir.value,
                'continue_from_lora': self.continue_from_lora.value,
                'wandb_key': self.wandb_key.value,
                'resolution': self.resolution.value,
                'num_repeats': self.num_repeats.value,
                'epochs': self.epochs.value,
                'train_batch_size': self.train_batch_size.value,
                'flip_aug': self.flip_aug.value,
                'unet_lr': self._parse_learning_rate(self.unet_lr.value),
                'text_encoder_lr': self._parse_learning_rate(self.text_encoder_lr.value),
                'lr_scheduler': self.lr_scheduler.value,
                'lr_scheduler_number': self.lr_scheduler_number.value,
                'lr_warmup_ratio': self.lr_warmup_ratio.value,
                'min_snr_gamma_enabled': self.min_snr_gamma_enabled.value,
                'min_snr_gamma': self.min_snr_gamma.value,
                'ip_noise_gamma_enabled': self.ip_noise_gamma_enabled.value,
                'ip_noise_gamma': self.ip_noise_gamma.value,
                'multinoise': self.multinoise.value,
                'lora_type': self.lora_type.value,
                'network_dim': self.network_dim.value,
                'network_alpha': self.network_alpha.value,
                'conv_dim': self.conv_dim.value,
                'conv_alpha': self.conv_alpha.value,
                'optimizer': self.optimizer.value,
                'cross_attention': self.cross_attention.value,
                'precision': self.precision.value,
                'cache_latents': self.cache_latents.value,
                'cache_latents_to_disk': self.cache_latents_to_disk.value,
                'cache_text_encoder_outputs': self.cache_text_encoder_outputs.value,
                'shuffle_caption': self.shuffle_caption.value,
                'v_parameterization': self.v_parameterization.value,
                'save_every_n_epochs': self.save_every_n_epochs.value,
                'keep_only_last_n_epochs': self.keep_only_last_n_epochs.value,
                # Advanced training options
                'caption_dropout_rate': self.caption_dropout_rate.value,
                'caption_tag_dropout_rate': self.caption_tag_dropout_rate.value,
                'keep_tokens': self.keep_tokens.value,
                'noise_offset': self.noise_offset.value,
                'adaptive_noise_scale': self.adaptive_noise_scale.value,
                'zero_terminal_snr': self.zero_terminal_snr.value,
                'clip_skip': self.clip_skip.value,
                'vae_batch_size': self.vae_batch_size.value,
                'no_half_vae': self.no_half_vae.value,
                'bucket_reso_steps': self.bucket_reso_steps.value,
                'min_bucket_reso': self.min_bucket_reso.value,
                'max_bucket_reso': self.max_bucket_reso.value,
                'bucket_no_upscale': self.bucket_no_upscale.value,
                # Advanced options
                'advanced_mode_enabled': getattr(self, 'advanced_mode', widgets.Checkbox(value=False)).value,
                'advanced_optimizer': getattr(self, 'advanced_optimizer', type('obj', (object,), {'value': 'standard'})).value,
                'advanced_scheduler': getattr(self, 'advanced_scheduler', type('obj', (object,), {'value': 'auto'})).value,
                'fused_back_pass': getattr(self, 'fused_back_pass', widgets.Checkbox(value=False)).value,
                'lycoris_method': getattr(self, 'lycoris_method', type('obj', (object,), {'value': 'none'})).value,
                'experimental_features': self._get_experimental_features(),
            }
            self.manager.start_training(config)

    def _create_advanced_section(self):
        """Creates the Advanced Mode section with educational explanations"""
        
        # Advanced Mode Toggle
        advanced_header = widgets.HTML("""
        <h3>🧪 Advanced Training Mode</h3>
        <p><strong>⚠️ For experienced users only!</strong> These features are experimental and may require VastAI or high-end hardware.</p>
        """)
        
        self.advanced_mode = widgets.Checkbox(
            value=False,
            description="🚀 Enable Advanced Training Options",
            style={'description_width': 'initial'}
        )
        
        # Advanced options container (initially hidden)
        self.advanced_container = widgets.VBox([
            self._create_advanced_optimizer_section(),
            self._create_memory_optimization_section(),
            self._create_lycoris_advanced_section(),
            self._create_experimental_section()
        ])
        
        # Initially hide advanced options
        self.advanced_container.layout.display = 'none'
        
        # Show/hide based on toggle
        def toggle_advanced_mode(change):
            if change['new']:
                self.advanced_container.layout.display = 'block'
                self._show_advanced_warning()
            else:
                self.advanced_container.layout.display = 'none'
        
        self.advanced_mode.observe(toggle_advanced_mode, names='value')
        
        return widgets.VBox([
            advanced_header,
            self.advanced_mode,
            self.advanced_container
        ])
    
    def _create_advanced_optimizer_section(self):
        """Advanced optimizers with educational explanations"""
        
        optimizer_info = widgets.HTML("""
        <h4>🚀 Advanced Optimizers</h4>
        <p><strong>Choose your optimization algorithm:</strong></p>
        """)
        
        self.advanced_optimizer = widgets.Dropdown(
            options=[
                ('Standard (Use basic options)', 'standard'),
                ('CAME - Memory Efficient', 'came'),
                ('Prodigy Plus - Schedule Free', 'prodigy_plus'),
                ('StableAdamW - Experimental', 'stable_adamw'),
                ('ADOPT - Research Grade', 'adopt')
            ],
            value='standard',
            description='Optimizer:',
            style={'description_width': 'initial'}
        )
        
        # Dynamic explanation based on selection
        self.optimizer_explanation = widgets.HTML()
        
        def update_optimizer_explanation(change):
            explanations = {
                'standard': """
                <div style='background: #f0f8ff; padding: 10px; border-left: 4px solid #007acc;'>
                <strong>Standard Mode:</strong> Uses your basic optimizer selection above.<br>
                ✅ Safe and well-tested<br>
                ✅ Good for beginners
                </div>
                """,
                'came': """
                <div style='background: #f0fff0; padding: 10px; border-left: 4px solid #28a745;'>
                <strong>CAME (Derrian's Advanced):</strong> Memory-efficient optimizer from Derrian Distro.<br>
                ✅ Uses 30-40% less VRAM than AdamW<br>
                ✅ Often produces high-quality results<br>
                ✅ Auto-pairs with REX scheduler + Huber loss<br>
                ❌ Newer, less community testing<br>
                <em>🎯 Best for: VRAM-constrained training (8GB cards)</em>
                </div>
                """,
                'prodigy_plus': """
                <div style='background: #fff8dc; padding: 10px; border-left: 4px solid #ffc107;'>
                <strong>Prodigy Plus (OneTrainer):</strong> Learning rate AND schedule free!<br>
                ✅ No learning rate tuning needed<br>
                ✅ No scheduler needed<br>
                ✅ Memory optimizations included<br>
                ❌ Very new, experimental<br>
                <em>🎯 Best for: Users who hate hyperparameter tuning</em>
                </div>
                """,
                'stable_adamw': """
                <div style='background: #fff3cd; padding: 10px; border-left: 4px solid #856404;'>
                <strong>StableAdamW (Experimental):</strong> Research-grade stability improvements.<br>
                ✅ Better convergence stability<br>
                ✅ Handles difficult datasets better<br>
                ❌ Very experimental<br>
                ❌ May not work with all models<br>
                <em>⚠️ For research and experimentation only</em>
                </div>
                """,
                'adopt': """
                <div style='background: #f8d7da; padding: 10px; border-left: 4px solid #dc3545;'>
                <strong>ADOPT (Bleeding Edge):</strong> Adaptive gradient clipping research.<br>
                ✅ Potential for breakthrough results<br>
                ❌ Highly experimental<br>
                ❌ May crash or fail<br>
                ❌ No guarantees<br>
                <em>🔬 For AI researchers and risk-takers only!</em>
                </div>
                """
            }
            self.optimizer_explanation.value = explanations.get(change['new'], '')
            
            # Auto-update scheduler recommendations
            self._update_scheduler_recommendations(change['new'])
        
        self.advanced_optimizer.observe(update_optimizer_explanation, names='value')
        
        return widgets.VBox([
            optimizer_info,
            self.advanced_optimizer,
            self.optimizer_explanation
        ])
    
    def _create_memory_optimization_section(self):
        """Memory optimization techniques"""
        
        memory_info = widgets.HTML("""
        <h4>💾 Memory Wizardry</h4>
        <p><strong>Advanced VRAM reduction techniques:</strong></p>
        """)
        
        self.fused_back_pass = widgets.Checkbox(
            value=False,
            description="🚧 Fused Back Pass (Requires OneTrainer - Coming Soon)",
            style={'description_width': 'initial'},
            disabled=True  # Disable until OneTrainer integration
        )
        
        self.gradient_checkpointing = widgets.Checkbox(
            value=False,
            description="Aggressive Gradient Checkpointing",
            style={'description_width': 'initial'}
        )
        
        fused_explanation = widgets.HTML("""
        <div style='background: #fff3cd; padding: 10px; border-left: 4px solid #856404;'>
        <strong>🚧 Fused Back Pass - OneTrainer Integration Required</strong><br><br>
        <strong>What it would do:</strong><br>
        📊 Calculate gradient → ⚡ Update immediately → 🗑️ Free VRAM → 🔄 Next layer<br><br>
        <strong>Why it's disabled:</strong><br>
        ❌ Requires OneTrainer's custom training loop implementation<br>
        ❌ Cannot be added as simple config flag to SD scripts<br>
        ❌ Needs fundamental changes to gradient handling<br><br>
        <strong>🔮 Future Plans:</strong><br>
        • Integrate OneTrainer as optional backend<br>
        • Add backend switcher (SD Scripts vs OneTrainer)<br>
        • Enable advanced memory optimizations<br><br>
        <em>🎯 For now: Use gradient checkpointing + cache settings for VRAM optimization</em>
        </div>
        """)
        
        return widgets.VBox([
            memory_info,
            self.fused_back_pass,
            self.gradient_checkpointing,
            fused_explanation
        ])
    
    def _create_lycoris_advanced_section(self):
        """Advanced LyCORIS methods"""
        
        lycoris_info = widgets.HTML("""
        <h4>🦄 LyCORIS Advanced Methods</h4>
        <p><strong>Beyond standard LoRA - cutting-edge adaptation techniques:</strong></p>
        """)
        
        self.lycoris_method = widgets.Dropdown(
            options=[
                ('None (Use Main LoRA Type)', 'none'),
                ('BOFT - Butterfly Transform', 'boft')
            ],
            value='none',
            description='Advanced LyCORIS:',
            style={'description_width': 'initial'}
        )
        
        self.lycoris_explanation = widgets.HTML()
        
        def update_lycoris_explanation(change):
            explanations = {
                'none': """
                <div style='background: #f0f8ff; padding: 10px; border-left: 4px solid #007acc;'>
                <strong>Standard LoRA:</strong> The classic, reliable choice.<br>
                ✅ Well-tested and stable<br>
                ✅ Fast training<br>
                ✅ Universal compatibility
                </div>
                """,
                'dora': """
                <div style='background: #f0fff0; padding: 10px; border-left: 4px solid #28a745;'>
                <strong>DoRA (Weight Decomposition):</strong> Trains like full fine-tune!<br>
                ✅ Much higher quality than standard LoRA<br>
                ✅ Better coherency and detail preservation<br>
                ✅ Especially good for faces and complex scenes<br>
                ❌ 2-3x slower training<br>
                ❌ More complex to tune<br>
                <em>🎯 Worth it for: High-quality character/style LoRAs</em>
                </div>
                """,
                'lokr': """
                <div style='background: #fff8dc; padding: 10px; border-left: 4px solid #ffc107;'>
                <strong>LoKr (Kronecker Product):</strong> Mathematical efficiency master.<br>
                ✅ Better parameter efficiency than standard LoRA<br>
                ✅ Can achieve same quality with smaller file sizes<br>
                ✅ Good for concept learning<br>
                ❌ More sensitive to hyperparameters<br>
                <em>🎯 Best for: Concept LoRAs and style transfer</em>
                </div>
                """,
                'ia3': """
                <div style='background: #fff3cd; padding: 10px; border-left: 4px solid #856404;'>
                <strong>(IA)³ (Implicit Attention):</strong> Attention-focused adaptation.<br>
                ✅ Very parameter efficient<br>
                ✅ Good for style and lighting changes<br>
                ✅ Fast training<br>
                ❌ Limited for complex content changes<br>
                <em>🎯 Perfect for: Style LoRAs and lighting adjustments</em>
                </div>
                """
            }
            self.lycoris_explanation.value = explanations.get(change['new'], '')
        
        self.lycoris_method.observe(update_lycoris_explanation, names='value')
        
        return widgets.VBox([
            lycoris_info,
            self.lycoris_method,
            self.lycoris_explanation
        ])
    
    def _create_experimental_section(self):
        """Experimental features section"""
        
        experimental_info = widgets.HTML("""
        <h4>🔬 Experimental Lab</h4>
        <p><strong>⚠️ Dragons be here! Use at your own risk:</strong></p>
        """)
        
        self.experimental_options = widgets.VBox([
            widgets.Checkbox(
                value=False,
                description="🧬 HakuLatent EQ-VAE (Future)",
                style={'description_width': 'initial'},
                disabled=True  # Not implemented yet
            ),
            widgets.Checkbox(
                value=False,
                description="⚡ Adversarial Loss (Research)",
                style={'description_width': 'initial'},
                disabled=True  # Not implemented yet
            ),
            widgets.Checkbox(
                value=False,
                description="🌊 Multi-Resolution Training",
                style={'description_width': 'initial'}
            ),
            widgets.HTML("""
            <div style='background: #fff3cd; padding: 8px; border-left: 4px solid #856404;'>
            <strong>🚧 Work in Progress:</strong><br>
            • EQ-VAE: HakuLatent's advanced latent regularization<br>
            • Adversarial Loss: GAN-style training improvements<br>
            • Multi-Res: Train on multiple resolutions simultaneously<br><br>
            <em>These will be enabled as they become stable!</em>
            </div>
            """)
        ])
        
        return widgets.VBox([
            experimental_info,
            self.experimental_options
        ])
    
    def _update_scheduler_recommendations(self, optimizer):
        """Update scheduler recommendations based on optimizer choice"""
        if hasattr(self, 'advanced_scheduler'):
            recommendations = {
                'came': 'rex',
                'prodigy_plus': 'constant',  # Schedule-free
                'standard': 'cosine'
            }
            
            recommended = recommendations.get(optimizer, 'cosine')
            if recommended in [option[1] for option in self.lr_scheduler.options]:
                self.lr_scheduler.value = recommended
    
    def _show_advanced_warning(self):
        """Show warning when advanced mode is enabled"""
        warning = widgets.HTML("""
        <div style='background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 10px 0;'>
        <strong>⚠️ ADVANCED MODE ACTIVATED!</strong><br><br>
        You've entered the experimental zone! These features are:
        <ul>
        <li>🔬 <strong>Cutting-edge:</strong> May be unstable or break</li>
        <li>🚀 <strong>VastAI optimized:</strong> Some need powerful hardware</li>
        <li>🧠 <strong>Research-grade:</strong> Results may vary wildly</li>
        <li>💀 <strong>No guarantees:</strong> Backup your work!</li>
        </ul>
        <em>"Either gonna work or blow up!" - You asked for it! 😄</em>
        </div>
        """)
        
        # Add warning to the container
        if len(self.advanced_container.children) == 4:  # Only add once
            self.advanced_container.children = [warning] + list(self.advanced_container.children)
    
    def _get_experimental_features(self):
        """Collect experimental feature settings"""
        if hasattr(self, 'experimental_options'):
            # Extract checkbox values from experimental options
            features = {}
            for i, child in enumerate(self.experimental_options.children):
                if hasattr(child, 'value') and hasattr(child, 'description'):
                    features[f'experimental_{i}'] = child.value
            return features
        return {}

    def display(self):
        display(self.widget_box)
