# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/training_monitor_widget.py
import os
import re
import threading
import time

import ipywidgets as widgets
from IPython.display import display

# Inference functionality removed
from shared_managers import get_config_manager


class TrainingMonitorWidget:
    def __init__(self, training_manager_instance):
        self.training_manager = training_manager_instance
        self.training_config = None # To store the config passed from TrainingWidget

# Sidecar functionality removed

        self.create_widgets()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.training_phase = "Initializing..."
        self._last_checkpoint_time = None
        self._is_final_epoch = False

        # Inference parameters - DISABLED FOR NOW
        # self.sample_prompt = ""
        # self.sample_num_images = 3  # Generate 3 images every epoch
        # self.sample_resolution = 512  # Will be auto-detected based on model type
        # self.sample_seed = 42

        # Use actual paths from training manager
        self.base_model_path = ""  # This will be set from config when training starts
        self.output_dir = self.training_manager.output_dir if hasattr(self.training_manager, 'output_dir') else ""

    def set_training_config(self, training_config: dict):
        """Explicitly set the training config and initialize the monitor's state."""
        self.training_config = training_config

        # Extract total epochs and steps from the config
        # Note: Actual total steps might be calculated differently, but this is a good starting point.
        self.total_epochs = self.training_config.get('max_train_epochs', 10) # Default to 10 if not found
        num_steps_per_epoch = self.training_config.get('num_steps_per_epoch', 0)
        self.total_steps = self.total_epochs * num_steps_per_epoch

        # Reset UI elements to their initial state with correct max values
        self.update_phase("Configuration loaded. Ready to start training.", "info")
        self.update_progress(epoch=0, total_epochs=self.total_epochs, step=0, total_steps=self.total_steps)
        self.log_message(f"Training configured for {self.total_epochs} epochs.")


    def create_widgets(self):
        """Create the training monitor interface with accordion structure"""

        # Header for the entire widget
        main_header = widgets.HTML("<h2>📈 Training Progress & Control</h2>")

        # Create accordion sections
        self.create_training_control_section()
        self.create_progress_monitoring_section()
# Sample viewing/inference section removed

        # Create accordion
        self.accordion = widgets.Accordion(children=[
            self.training_control_box,
            self.progress_monitoring_box,
        ])
        self.accordion.set_title(0, "🚀 Start Training")
        self.accordion.set_title(1, "📊 Live Progress Monitor")

        # Main widget container
        self.widget_box = widgets.VBox([main_header, self.accordion])

    def create_training_control_section(self):
        """Create the training control section with start button"""
        control_desc = widgets.HTML("""<h3>🚀 Training Control</h3>
        <p>Start your LoRA training here. Configure your settings in the Training Configuration section above first!</p>
        <div style='padding: 10px; border: 1px solid #007acc; border-radius: 5px; margin: 10px 0;'>
        <strong>📋 Before Starting:</strong><br>
        • Configure your training parameters in the <strong>Training Configuration</strong> section above<br>
        • Verify your dataset directory and model paths are correct<br>
        • Check your learning rates and training steps look reasonable<br>
        • Make sure you have enough disk space and VRAM
        </div>""")

        # Start training button
        self.start_training_button = widgets.Button(
            description="🚀 Start LoRA Training",
            button_style='success',
            layout=widgets.Layout(width='300px', height='50px')
        )

        self.stop_training_button = widgets.Button(
            description="🛑 Emergency Stop",
            button_style='danger',
            layout=widgets.Layout(width='300px', height='50px')
        )
        self.stop_training_button.on_click(self.stop_training_clicked)
        self.stop_training_button.disabled = True

        # Training status
        self.control_status = widgets.HTML(
            value="<div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 10px 0;'>"
                  "<strong>Status:</strong> Ready to start training. Click the button above when you're ready!</div>"
        )

        # Hook up the button (we'll connect this to the training widget later)
        self.start_training_button.on_click(self.start_training_clicked)

        self.training_control_box = widgets.VBox([
            control_desc,
            widgets.HBox([self.start_training_button, self.stop_training_button]),
            self.control_status
        ])

    def create_progress_monitoring_section(self):
        """Create the progress monitoring section"""

        # Progress header
        progress_desc = widgets.HTML("<h3>📊 Live Training Progress</h3><p>Real-time monitoring will appear here when training starts.</p>")

        # Training Phase Status (fix theme colors)
        self.phase_status = widgets.HTML(
            value="<div style='padding: 15px; border: 1px solid #007acc; border-radius: 8px;'>"
                  "<strong>📊 Phase:</strong> Waiting to start...</div>"
        )

        # Progress Bars
        self.epoch_progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Epoch:',
            bar_style='info',
            style={'bar_color': '#007acc', 'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )

        self.step_progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Step:',
            bar_style='success',
            style={'bar_color': '#28a745', 'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )

        # Progress Labels
        self.epoch_label = widgets.HTML(value="<strong>Epoch:</strong> Not started")
        self.step_label = widgets.HTML(value="<strong>Step:</strong> Not started")

        # Resource Monitoring (optional placeholders)
        self.resource_info = widgets.HTML(
            value="<p><strong>💾 Resources:</strong> Monitoring will begin when training starts</p>"
        )

        # Training Log Output
        self.training_log = widgets.Output(
            layout=widgets.Layout(
                height='400px',
                overflow='scroll',
                border='1px solid #ddd',
                margin='10px 0'
            )
        )

        # Sample generation button - REMOVED (inference disabled)

        # Auto-save status (fix theme colors)
        self.autosave_status = widgets.HTML(
            value="<div style='padding: 8px; border: 1px solid #28a745; border-radius: 5px;'>"
                  "<strong>💾 Auto-save:</strong> Enabled - checkpoints saved each epoch</div>"
        )

        # Create the progress monitoring box
        self.progress_monitoring_box = widgets.VBox([
            progress_desc,
            self.phase_status,
            self.epoch_label,
            self.epoch_progress,
            self.step_label,
            self.step_progress,
            self.resource_info,
            widgets.HTML("<h4>📋 Training Log</h4>"),
            self.training_log,
            self.autosave_status
        ])

    def start_training_clicked(self, b):
        """Handle start training button click - DEAD SIMPLE FILE HUNTING APPROACH"""
        try:
            # Create our brilliant file hunter
            config_mgr = get_config_manager()

            # Step 1: Hunt for TOML files like a boss
            if not config_mgr.files_ready():
                self.control_status.value = "<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>❌ Error:</strong> Config files not ready! Click 'Prepare Training Configuration' first!</div>"
                return

            # Step 2: Files found? LET'S FUCKING GOOOOO! 🚀
            self.control_status.value = "<div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px; margin: 10px 0;'><strong>Status:</strong> ✅ Config files found! Starting training! 🚀</div>"
            self.accordion.selected_index = 1  # Switch to progress tab

            # Step 3: Get the file paths and launch
            config_paths = config_mgr.get_config_paths()
            print(f"🔍 Found config files: {list(config_paths.keys())}")

            # Launch training using the file paths
            self.training_manager.launch_from_files(config_paths, monitor_widget=self)
            self.start_training_button.disabled = True
            self.stop_training_button.disabled = False

        except Exception as e:
            self.control_status.value = f"<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>💥 Error:</strong> {str(e)}</div>"
            print(f"💥 Training start error: {e}")

    def stop_training_clicked(self, b):
        """Handle stop training button click"""
        self.control_status.value = "<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>Status:</strong> 🛑 Terminating training...</div>"
        self.training_manager.stop_training()
        self.start_training_button.disabled = False
        self.stop_training_button.disabled = True

    def update_progress(self, epoch=None, total_epochs=None, step=None, total_steps=None, phase=None):
        """Update progress indicators"""
        if epoch is not None and total_epochs is not None:
            self.current_epoch = epoch
            self.total_epochs = total_epochs
            self.epoch_progress.value = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
            self.epoch_progress.max = total_epochs
            self.epoch_label.value = f"<strong>Epoch:</strong> {epoch}/{total_epochs}"

        if step is not None and total_steps is not None:
            self.current_step = step
            self.total_steps = total_steps
            self.step_progress.value = int((step / total_steps) * 100) if total_steps > 0 else 0
            self.step_progress.max = total_steps
            self.step_label.value = f"<strong>Step:</strong> {step}/{total_steps}"

        if phase is not None:
            self.training_phase = phase
            self.phase_status.value = (
                f"<div style='padding: 15px; border: 1px solid #007acc; border-radius: 8px;'>"
                f"<strong>📊 Phase:</strong> {phase}</div>"
            )


    def update_phase(self, phase_text, phase_type="info"):
        """Update the current training phase"""
        phase_colors = {
            "info": "#007acc",
            "warning": "#ffc107",
            "success": "#28a745",
            "error": "#dc3545"
        }
        color = phase_colors.get(phase_type, "#007acc")

        self.phase_status.value = (
            f"<div style='padding: 15px; border: 1px solid {color}; border-radius: 8px;'>"
            f"<strong>📊 Phase:</strong> {phase_text}</div>"
        )

    def parse_training_output(self, line: str):
        """Parse training output and dispatch to specialized handlers."""
        line = line.strip()
        if not line:
            return

        with self.training_log:
            print(line)

        # Dispatch to handlers - order can matter.
        # Step/epoch parsing is most common, try it first.
        try:
            if self._parse_epoch_and_step(line):
                return # Line was handled
        except Exception as e:
            logger.warning(f"Error parsing epoch/step from line '{line}': {e}")

        try:
            if self._parse_phase(line):
                return # Line was handled
        except Exception as e:
            logger.warning(f"Error parsing phase from line '{line}': {e}")

    def _parse_phase(self, line: str) -> bool:
        """Parse the line for keywords indicating a change in the training phase."""
        line_lower = line.lower()
        
        # More specific keywords first
        if "caching latents to disk" in line_lower:
            self.update_phase("Caching latents to disk...", "info")
            return True
        elif "caching latents" in line_lower: # General case
            self.update_phase("Caching latents...", "info")
            return True
        elif "prepare optimizer" in line_lower:
            self.update_phase("Preparing optimizer and data loader...", "info")
            return True
        elif "make buckets" in line_lower or "enable bucket" in line_lower:
            self.update_phase("Creating resolution buckets...", "info")
            return True
        elif "start training" in line_lower:
            self.update_phase("Training started!", "success")
            return True
            
        return False

    def _parse_epoch_and_step(self, line: str) -> bool:
        """Parse the line for epoch and step information using more robust regex."""
        # Regex to find all key-value pairs of numbers, handling formats like "key: 123" or "key=123"
        matches = re.findall(r'(\b\w+\b)\s*[:=]\s*(\d+)', line.lower())
        
        if not matches:
            return False

        # Convert matches to a dictionary for easy lookup
        data = {key: int(value) for key, value in matches}

        epoch = data.get('epoch')
        step = data.get('step')

        # If we found an epoch number, update our state
        if epoch is not None:
            # Check if we're starting the final epoch
            if epoch >= self.total_epochs and self.total_epochs > 0:
                self.update_phase(f"Starting final epoch ({epoch}/{self.total_epochs})!", "warning")
                self._is_final_epoch = True
            elif self.total_epochs > 0:
                self.update_phase(f"Starting epoch {epoch}/{self.total_epochs}", "info")

            self.update_progress(epoch=epoch, total_epochs=self.total_epochs)
            return True # Mark as handled

        # If we only found a step number, update step progress
        if step is not None:
            self.update_progress(step=step, total_steps=self.total_steps)
            return True # Mark as handled
            
        return False


    def clear_log(self):
        """Clear the training log"""
        with self.training_log:
            self.training_log.clear_output()

    def log_message(self, message):
        """Add a message to the training log"""
        with self.training_log:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def display(self):
        """Display the widget"""
        display(self.widget_box)
