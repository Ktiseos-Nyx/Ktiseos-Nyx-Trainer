# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/utilities_widget.py
import os
import time
import ipywidgets as widgets
from IPython.display import display

from core.utilities_manager import UtilitiesManager
import logging



class UtilitiesWidget:
    def __init__(self, utilities_manager=None):
        # Use dependency injection - accept manager instance or create default
        if utilities_manager is None:
            utilities_manager = UtilitiesManager()

        self.manager = utilities_manager
        
        # Initialize logging
        self.logger = logging.getLogger("utilities_widget")
        
        self.create_widgets()

    def create_widgets(self):
        header_icon = "🔧"
        header_main = widgets.HTML(f"<h2>{header_icon} 4. Utilities</h2>")

        # --- Enhanced Hugging Face Upload ---
        upload_desc = widgets.HTML("""<h4>🚀 Enhanced HuggingFace Upload</h4>
        <p><strong>📋 Features:</strong> Multi-file upload, progress tracking, repository management, pull request creation</p>""")
        
        # Repository configuration
        self.hf_token = widgets.Password(description="HF Write Token:", placeholder="HuggingFace Write API Token", layout=widgets.Layout(width='99%'))
        
        hf_token_help = widgets.HTML("""
        <div style='background: #fff3cd; padding: 8px; border-radius: 5px; margin: 5px 0;'>
        🔑 Get your <strong>WRITE</strong> token <a href='https://huggingface.co/settings/tokens' target='_blank'>here</a>
        </div>
        """)
        
        repo_config_box = widgets.HBox([
            widgets.VBox([
                widgets.Label("Owner:"),
                widgets.Text(placeholder="your-username", layout=widgets.Layout(width='200px'))
            ], layout=widgets.Layout(width='33%')),
            widgets.VBox([
                widgets.Label("Repository:"),
                widgets.Text(placeholder="my-awesome-loras", layout=widgets.Layout(width='200px'))
            ], layout=widgets.Layout(width='33%')),
            widgets.VBox([
                widgets.Label("Type:"),
                widgets.Dropdown(options=['model', 'dataset', 'space'], value='model', layout=widgets.Layout(width='120px'))
            ], layout=widgets.Layout(width='33%'))
        ])
        
        self.hf_owner = repo_config_box.children[0].children[1]
        self.hf_repo_name = repo_config_box.children[1].children[1] 
        self.hf_repo_type = repo_config_box.children[2].children[1]
        
        self.hf_remote_folder = widgets.Text(description="Remote Folder:", placeholder="Optional: e.g., models/v1", layout=widgets.Layout(width='99%'))
        
        # File selection
        file_select_box = widgets.HBox([
            widgets.VBox([
                widgets.Label("File Type:"),
                widgets.Dropdown(options=self.manager.file_types, value='safetensors', layout=widgets.Layout(width='150px'))
            ], layout=widgets.Layout(width='50%')),
            widgets.VBox([
                widgets.Label("Sort By:"),
                widgets.Dropdown(options=['name', 'date'], value='name', layout=widgets.Layout(width='100px'))
            ], layout=widgets.Layout(width='50%'))
        ])
        
        self.hf_file_type = file_select_box.children[0].children[1]
        self.hf_sort_by = file_select_box.children[1].children[1]
        
        directory_box = widgets.HBox([
            widgets.Text(value=self.manager.project_root, description="Source Directory:", layout=widgets.Layout(width='80%')),
            widgets.Button(description="🔄 List Files", button_style='info', layout=widgets.Layout(width='15%'))
        ])
        
        self.hf_source_directory = directory_box.children[0]
        self.hf_refresh_button = directory_box.children[1]
        
        self.hf_file_list = widgets.SelectMultiple(options=[], description="Files:", layout=widgets.Layout(width='99%', height='150px'))
        
        # Upload options
        self.hf_commit_message = widgets.Textarea(value="Upload via Enhanced Utilities Widget 🤗", placeholder="Commit message", description="Message:", layout=widgets.Layout(width='99%', height='60px'))
        
        upload_options_box = widgets.HBox([
            widgets.Checkbox(value=False, description="Create Pull Request"),
            widgets.Checkbox(value=True, description="Clear output after upload")
        ])
        
        self.hf_create_pr = upload_options_box.children[0]
        self.hf_clear_after = upload_options_box.children[1]
        
        # Upload controls
        upload_controls_box = widgets.HBox([
            widgets.Button(description="⬆️ Upload Selected Files", button_style='success', layout=widgets.Layout(width='60%')),
            widgets.Button(description="🧹 Clear Output", button_style='warning', layout=widgets.Layout(width='35%'))
        ])
        
        self.hf_upload_button = upload_controls_box.children[0]
        self.hf_clear_button = upload_controls_box.children[1]
        
        # Progress tracking
        self.hf_progress_box = widgets.VBox([
            widgets.HBox([widgets.Label("Current File:", layout=widgets.Layout(width='100px')), widgets.Label("N/A")]),
            widgets.HBox([widgets.Label("Progress:", layout=widgets.Layout(width='100px')), widgets.Label("0/0")]),
            widgets.FloatProgress(value=0, min=0, max=100, description='Overall:', layout=widgets.Layout(width='90%'))
        ], layout=widgets.Layout(visibility='hidden', padding='10px', border='1px solid #ddd'))
        
        self.hf_current_file_label = self.hf_progress_box.children[0].children[1]
        self.hf_progress_label = self.hf_progress_box.children[1].children[1] 
        self.hf_progress_bar = self.hf_progress_box.children[2]
        
        self.hf_upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll'))
        
        hf_upload_box = widgets.VBox([
            upload_desc,
            self.hf_token,
            hf_token_help,
            repo_config_box,
            self.hf_remote_folder,
            file_select_box,
            directory_box,
            self.hf_file_list,
            self.hf_commit_message,
            upload_options_box,
            upload_controls_box,
            self.hf_progress_box,
            self.hf_upload_output
        ])

        # --- LoRA Resizing ---
        self.lora_input_path = widgets.Text(description="Input LoRA Path:", placeholder="/path/to/input_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_output_path = widgets.Text(description="Output LoRA Path:", placeholder="/path/to/output_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_new_dim = widgets.IntSlider(value=4, min=1, max=128, step=1, description='New Dim:', style={'description_width': 'initial'})
        self.lora_new_alpha = widgets.IntSlider(value=1, min=1, max=128, step=1, description='New Alpha:', style={'description_width': 'initial'})
        self.resize_button = widgets.Button(description="Resize LoRA", button_style='primary')
        self.resize_status = widgets.HTML("<p><strong>📊 Status:</strong> Ready to resize</p>")
        self.resize_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll'))
        lora_resize_box = widgets.VBox([
            self.lora_input_path,
            self.lora_output_path,
            self.lora_new_dim,
            self.lora_new_alpha,
            self.resize_button,
            self.resize_status,
            self.resize_output
        ])

        # --- Dataset Image Optimization ---
        optimization_desc = widgets.HTML("""<h4>🖼️ Dataset Image Optimization</h4>
        <p><strong>📋 What this does:</strong><br>
        • Convert PNG/BMP/TIFF images to JPEG or WebP for smaller file sizes<br>
        • Standardize image formats across your dataset<br>
        • Handle transparency properly (white background for JPEG)<br>
        • Keep original image dimensions - no resizing!</p>
        <p><strong>⚠️ Important:</strong> This modifies your images permanently. Make a backup first if needed!</p>""")

        self.optimize_dataset_path = widgets.Text(
            description="Dataset Path:",
            placeholder="datasets/your_dataset_folder",
            layout=widgets.Layout(width='99%')
        )

        self.optimize_format = widgets.Dropdown(
            options=[('WebP (Best compression)', 'webp'), ('JPEG (Wide compatibility)', 'jpeg')],
            value='webp',
            description='Target Format:',
            style={'description_width': 'initial'}
        )

        self.optimize_quality = widgets.IntSlider(
            value=95,
            min=60,
            max=100,
            step=5,
            description='Quality:',
            style={'description_width': 'initial'}
        )

        optimization_help = widgets.HTML("""<p><strong>💡 Tips:</strong><br>
        • <strong>Format conversion only:</strong> Images keep their original dimensions<br>
        • <strong>Quality 95:</strong> Excellent quality, good compression<br>
        • <strong>Quality 85:</strong> Good quality, better compression<br>
        • <strong>WebP:</strong> Best compression, smaller files<br>
        • <strong>JPEG:</strong> Wide compatibility, good compression</p>""")

        self.optimize_button = widgets.Button(
            description="🖼️ Optimize Dataset Images",
            button_style='warning',
            layout=widgets.Layout(width='50%')
        )

        self.cancel_optimize_button = widgets.Button(
            description="🚫 Cancel Optimization",
            button_style='danger',
            disabled=True,
            layout=widgets.Layout(width='50%')
        )

        # Progress tracking for optimization
        self.optimize_progress = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='99%', visibility='hidden')
        )

        self.optimize_progress_label = widgets.Label(
            value="Ready to optimize images",
            layout=widgets.Layout(visibility='hidden')
        )

        # Button row and progress
        optimize_button_row = widgets.HBox([self.optimize_button, self.cancel_optimize_button])
        optimize_progress_box = widgets.VBox([
            self.optimize_progress,
            self.optimize_progress_label
        ], layout=widgets.Layout(margin='5px 0px'))

        self.optimize_status = widgets.HTML("<p><strong>📊 Status:</strong> Ready to optimize</p>")
        self.optimize_output = widgets.Output(layout=widgets.Layout(height='400px', overflow='scroll'))

        optimization_box = widgets.VBox([
            optimization_desc,
            self.optimize_dataset_path,
            self.optimize_format,
            self.optimize_quality,
            optimization_help,
            optimize_button_row,
            optimize_progress_box,
            self.optimize_status,
            self.optimize_output
        ])

        # Dataset counting has been moved to the Dataset Widget for better organization

        # --- Accordion ---
        accordion = widgets.Accordion(children=[
            hf_upload_box,
            lora_resize_box,
            optimization_box
        ])
        accordion.set_title(0, "▶️ Upload to Hugging Face")
        accordion.set_title(1, "▶️ LoRA Resizing")
        accordion.set_title(2, "🖼️ Dataset Image Optimization")

        self.widget_box = widgets.VBox([header_main, accordion])

        # --- Button Events ---
        self.hf_upload_button.on_click(self.run_enhanced_upload_to_hf)
        self.hf_clear_button.on_click(self.clear_upload_output)
        self.hf_refresh_button.on_click(self.refresh_file_list)
        self.hf_file_type.observe(self.on_file_type_change, names='value')
        self.hf_sort_by.observe(self.on_sort_change, names='value')
        self.resize_button.on_click(self._handle_async_resize)
        self.cancel_optimize_button.on_click(self.cancel_current_optimization)
        self.optimize_button.on_click(self._handle_async_optimize)
        
        # Initialize file list
        self.refresh_file_list(None)

    def clear_upload_output(self, b):
        """Clear the upload output area"""
        self.hf_upload_output.clear_output(wait=True)
        self.hf_progress_box.layout.visibility = 'hidden'

    def refresh_file_list(self, b):
        """Refresh the file list based on current directory and file type"""
        directory = self.hf_source_directory.value.strip()
        file_extension = self.hf_file_type.value
        sort_by = self.hf_sort_by.value
        
        self.logger.info(f"Refreshing file list: directory='{directory}', extension='{file_extension}', sort='{sort_by}'")
        
        if not directory or not os.path.isdir(directory):
            self.logger.warning(f"Invalid directory: {directory}")
            self.hf_file_list.options = []
            with self.hf_upload_output:
                print(f"⚠️ Invalid directory: {directory}")
            return
        
        files = self.manager.get_files_in_directory(directory, file_extension, sort_by)
        self.hf_file_list.options = files
        
        self.logger.info(f"File scan complete: found {len(files)} files")
        if files:
            self.logger.debug(f"Files found: {files}")
        
        with self.hf_upload_output:
            if files:
                print(f"✨ Found {len(files)} '.{file_extension}' files in '{directory}'")
            else:
                print(f"🤷 No '.{file_extension}' files found in '{directory}'")

    def on_file_type_change(self, change):
        """Handle file type dropdown changes"""
        self.refresh_file_list(None)

    def on_sort_change(self, change):
        """Handle sort order dropdown changes"""
        self.refresh_file_list(None)

    def progress_callback(self, current, total, filename):
        """Update progress display during upload"""
        self.hf_current_file_label.value = filename
        self.hf_progress_label.value = f"{current}/{total}"
        percentage = int((current / total) * 100)
        self.hf_progress_bar.value = percentage

    def run_enhanced_upload_to_hf(self, b):
        """Enhanced multi-file upload to HuggingFace"""
        self.hf_upload_output.clear_output(wait=True)
        self.hf_progress_box.layout.visibility = 'visible'
        
        # Collect form data
        hf_token = self.hf_token.value.strip()
        owner = self.hf_owner.value.strip()
        repo_name = self.hf_repo_name.value.strip()
        repo_type = self.hf_repo_type.value
        remote_folder = self.hf_remote_folder.value.strip()
        selected_files = list(self.hf_file_list.value)
        commit_message = self.hf_commit_message.value.strip()
        create_pr = self.hf_create_pr.value
        
        with self.hf_upload_output:
            print("🎯 Preparing enhanced upload...")
            
            # Validation
            if not hf_token:
                print("❌ Error: HuggingFace token is required")
                print("💡 Get your token from: https://huggingface.co/settings/tokens")
                return
            
            if not owner or not repo_name:
                print("❌ Error: Owner and repository name are required")
                return
            
            if not selected_files:
                print("❌ Error: No files selected for upload")
                print("💡 Select files from the list above")
                return
            
            print(f"📁 Repository: {owner}/{repo_name} (type: {repo_type})")
            if remote_folder:
                print(f"📂 Remote folder: {remote_folder}")
            print(f"📦 Files selected: {len(selected_files)}")
            if self.manager.check_hf_transfer_availability():
                print("🚀 HF_TRANSFER enabled for faster uploads")
            
            # Perform upload
            results = self.manager.upload_multiple_files_to_huggingface(
                hf_token=hf_token,
                owner=owner,
                repo_name=repo_name,
                repo_type=repo_type,
                selected_files=selected_files,
                remote_folder=remote_folder,
                commit_message=commit_message,
                create_pr=create_pr,
                progress_callback=self.progress_callback
            )
            
            # Display results
            if results["success"]:
                print(f"\n✨ Upload completed! {results['success_count']}/{results['total_files']} files uploaded")
                
                if results["uploaded_files"]:
                    print("\n✅ Successfully uploaded:")
                    for file_info in results["uploaded_files"]:
                        print(f"  • {file_info['file']} ({file_info['size']}) - {file_info['duration']}")
                        print(f"    └─ {file_info['url']}")
                
                if results["failed_files"]:
                    print("\n❌ Failed uploads:")
                    for file_info in results["failed_files"]:
                        print(f"  • {file_info['file']}: {file_info['error']}")
                
                if create_pr and results['success_count'] > 0:
                    print(f"\n🎉 Pull request created! Check: https://huggingface.co/{results['repo_id']}/pulls")
                elif results['success_count'] > 0:
                    repo_url = f"https://huggingface.co/{results['repo_id']}"
                    if remote_folder:
                        repo_url += f"/tree/main/{remote_folder}"
                    print(f"\n🎉 Files uploaded to: {repo_url}")
                
            else:
                print(f"❌ Upload failed: {results.get('error', 'Unknown error')}")
            
            # Auto-clear after delay if enabled
            if self.hf_clear_after.value and results["success"]:
                import threading
                def delayed_clear():
                    time.sleep(5)
                    self.hf_upload_output.clear_output(wait=True)
                    self.hf_progress_box.layout.visibility = 'hidden'
                threading.Thread(target=delayed_clear, daemon=True).start()

    def run_resize_lora(self, b):
        self.resize_output.clear_output()
        self.resize_status.value = "<p><strong>⚙️ Status:</strong> Resizing LoRA...</p>"
        with self.resize_output:
            success = self.manager.resize_lora(
                self.lora_input_path.value,
                self.lora_output_path.value,
                self.lora_new_dim.value,
                self.lora_new_alpha.value
            )
            if success:
                self.resize_status.value = "<p><strong>✅ Status:</strong> LoRA resized successfully!</p>"
            else:
                self.resize_status.value = "<p><strong>❌ Status:</strong> LoRA resize failed. Check logs.</p>"

    def run_optimize_dataset(self, b):
        """Handle dataset image optimization"""
        self.optimize_output.clear_output()
        self.optimize_status.value = "<p><strong>⚙️ Status:</strong> Optimizing dataset images...</p>"

        with self.optimize_output:
            dataset_path = self.optimize_dataset_path.value.strip()
            target_format = self.optimize_format.value
            quality = self.optimize_quality.value

            if not dataset_path:
                self.optimize_status.value = "<p><strong>❌ Status:</strong> Please enter a dataset path.</p>"
                print("❌ Please enter a dataset path.")
                return

            print("🖼️ Starting image optimization...")
            print(f"📁 Dataset: {dataset_path}")
            print(f"🎯 Target format: {target_format.upper()}")
            print(f"⚙️ Quality: {quality}")
            print("📐 No resizing - keeping original dimensions")

            # Call the optimization method with no resizing
            success = self.manager.optimize_dataset_images(
                dataset_path=dataset_path,
                target_format=target_format,
                max_file_size_mb=999,  # High value to disable size-based resizing
                quality=quality,
                max_dimension=None  # No dimension-based resizing
            )

            if success:
                self.optimize_status.value = "<p><strong>✅ Status:</strong> Dataset optimization complete!</p>"
            else:
                self.optimize_status.value = "<p><strong>❌ Status:</strong> Dataset optimization failed. Check logs.</p>"

    # Dataset counting has been moved to the Dataset Widget for better organization

    # ===== ASYNC IMPLEMENTATIONS =====

    def _handle_async_resize(self, b):
        """Wrapper to handle async LoRA resize function"""
        task = asyncio.ensure_future(self.run_async_resize_lora(b))
        self._current_resize_task = task
        task.add_done_callback(self._resize_task_done)

    def _handle_async_optimize(self, b):
        """Wrapper to handle async optimization function"""
        task = asyncio.ensure_future(self.run_async_optimize_dataset(b))
        self._current_optimization_task = task
        task.add_done_callback(self._optimization_task_done)

    def _resize_task_done(self, task):
        """Callback when resize task completes or fails"""
        try:
            if task.cancelled():
                with self.resize_output:
                    print("🚫 LoRA resize cancelled by user")
                self.resize_status.value = "<p><strong>🚫 Status:</strong> LoRA resize cancelled</p>"
            elif task.exception():
                with self.resize_output:
                    print(f"❌ LoRA resize failed: {task.exception()}")
                self.resize_status.value = "<p><strong>❌ Status:</strong> LoRA resize failed with error</p>"
        except Exception as e:
            self.logger.error(f"Error in resize task callback: {e}")
        finally:
            self._current_resize_task = None

    def _optimization_task_done(self, task):
        """Callback when optimization task completes or fails"""
        try:
            if task.cancelled():
                with self.optimize_output:
                    print("🚫 Optimization cancelled by user")
                self.optimize_status.value = "<p><strong>🚫 Status:</strong> Optimization cancelled</p>"
            elif task.exception():
                with self.optimize_output:
                    print(f"❌ Optimization failed: {task.exception()}")
                self.optimize_status.value = "<p><strong>❌ Status:</strong> Optimization failed with error</p>"
        except Exception as e:
            self.logger.error(f"Error in optimization task callback: {e}")
        finally:
            self._current_optimization_task = None
            self._update_optimization_button_states()

    def cancel_current_optimization(self, b):
        """Cancel the currently running optimization task"""
        if hasattr(self, '_current_optimization_task') and self._current_optimization_task and not self._current_optimization_task.done():
            self._current_optimization_task.cancel()
            self.logger.info("Optimization cancellation requested")
        else:
            with self.optimize_output:
                print("ℹ️ No active optimization to cancel")

    def _update_optimization_button_states(self):
        """Update optimization button states based on current conditions"""
        optimization_in_progress = (hasattr(self, '_current_optimization_task') and
                                   self._current_optimization_task and
                                   not self._current_optimization_task.done())

        self.optimize_button.disabled = optimization_in_progress
        self.cancel_optimize_button.disabled = not optimization_in_progress

    def _hide_optimization_progress_widgets(self):
        """Hide the optimization progress widgets"""
        if hasattr(self, 'optimize_progress'):
            self.optimize_progress.layout.visibility = 'hidden'
        if hasattr(self, 'optimize_progress_label'):
            self.optimize_progress_label.layout.visibility = 'hidden'

    async def run_async_resize_lora(self, b):
        """Async LoRA resizing with progress feedback"""
        self.resize_output.clear_output()
        self.resize_status.value = "<p><strong>⚙️ Status:</strong> Resizing LoRA...</p>"

        try:
            with self.resize_output:
                print("🔧 Starting LoRA resize operation...")

                # Run resize in thread executor to avoid blocking UI
                import concurrent.futures
                import functools

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Wrap the blocking resize operation
                    resize_func = functools.partial(
                        self.manager.resize_lora,
                        self.lora_input_path.value,
                        self.lora_output_path.value,
                        self.lora_new_dim.value,
                        self.lora_new_alpha.value
                    )

                    # Run in executor to avoid blocking
                    success = await loop.run_in_executor(executor, resize_func)

                if success:
                    self.resize_status.value = "<p><strong>✅ Status:</strong> LoRA resize complete</p>"
                    print("✅ LoRA resize completed successfully!")
                else:
                    self.resize_status.value = "<p><strong>❌ Status:</strong> LoRA resize failed. Check logs.</p>"

        except asyncio.CancelledError:
            print("\n🚫 LoRA resize was cancelled")
            self.resize_status.value = "<p><strong>🚫 Status:</strong> LoRA resize cancelled by user</p>"
            raise
        except Exception as e:
            print(f"\n❌ LoRA resize failed with error: {e}")
            self.resize_status.value = "<p><strong>❌ Status:</strong> LoRA resize failed with error</p>"
            self.logger.error(f"LoRA resize error: {e}")

    async def run_async_optimize_dataset(self, b):
        """Async dataset image optimization with progress tracking"""
        self.optimize_output.clear_output()

        # Show progress widgets
        self.optimize_progress.layout.visibility = 'visible'
        self.optimize_progress_label.layout.visibility = 'visible'
        self.optimize_progress.value = 0
        self.optimize_progress_label.value = "Initializing optimization..."

        # Update button states
        self._update_optimization_button_states()

        self.optimize_status.value = "<p><strong>⚙️ Status:</strong> Optimizing dataset images...</p>"

        try:
            with self.optimize_output:
                dataset_path = self.optimize_dataset_path.value.strip()
                if not dataset_path:
                    print("❌ Please specify a dataset path")
                    self.optimize_status.value = "<p><strong>❌ Status:</strong> Please specify a dataset path</p>"
                    self._hide_optimization_progress_widgets()
                    return

                if not os.path.exists(dataset_path):
                    print(f"❌ Dataset path not found: {dataset_path}")
                    self.optimize_status.value = "<p><strong>❌ Status:</strong> Dataset path not found</p>"
                    self._hide_optimization_progress_widgets()
                    return

                target_format = self.optimize_format.value
                quality = self.optimize_quality.value

                print(f"🖼️ Optimizing images in: {dataset_path}")
                print(f"📊 Target format: {target_format.upper()}")
                print(f"⚙️ Quality: {quality}")
                print("="*50)

                # Run optimization in thread executor with progress tracking
                import concurrent.futures
                import functools

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Create a partial function for the manager's optimization
                    optimize_func = functools.partial(
                        self.manager.optimize_dataset_images,
                        dataset_path=dataset_path,
                        target_format=target_format,
                        max_file_size_mb=999,  # High value to disable size-based resizing
                        quality=quality,
                        max_dimension=None  # No dimension-based resizing
                    )

                    # Show initial progress
                    self.optimize_progress.value = 10
                    self.optimize_progress_label.value = "Starting optimization process..."

                    # Run in executor to avoid blocking
                    success = await loop.run_in_executor(executor, optimize_func)

                    # Complete progress
                    self.optimize_progress.value = 100
                    self.optimize_progress_label.value = "Optimization complete!"

                if success:
                    self.optimize_status.value = "<p><strong>✅ Status:</strong> Dataset optimization complete</p>"
                    print("✅ Dataset optimization completed successfully!")
                else:
                    self.optimize_status.value = "<p><strong>❌ Status:</strong> Dataset optimization failed</p>"

        except asyncio.CancelledError:
            print("\n🚫 Optimization was cancelled")
            self.optimize_status.value = "<p><strong>🚫 Status:</strong> Optimization cancelled by user</p>"
            raise
        except Exception as e:
            print(f"\n❌ Optimization failed with error: {e}")
            self.optimize_status.value = "<p><strong>❌ Status:</strong> Optimization failed with error</p>"
            self.logger.error(f"Optimization error: {e}")
        finally:
            # Always hide progress widgets when done
            await asyncio.sleep(0.5)  # Brief delay to let user see completion
            self._hide_optimization_progress_widgets()
            self._update_optimization_button_states()

    def display(self):
        display(self.widget_box)
