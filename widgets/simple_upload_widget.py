#!/usr/bin/env python3
"""
Simple Local File Upload Widget
Based on the beautiful HF uploader pattern - clean, simple, and actually works!
"""

import glob
import os
import time
import shutil
from pathlib import Path
import math

from ipywidgets import (Text, Dropdown, Button, SelectMultiple, VBox, HBox,
                        Output, Layout, Checkbox, HTML, Textarea, Label,
                        FloatProgress)
from IPython.display import display, clear_output

class LocalFileUploader:
    """
    A Jupyter widget-based tool to upload files to local directories.
    Clean and simple - no threading madness, no polling chaos!
    """

    def __init__(self):
        self.file_types = [
            # Image Files 🎨 (most common for datasets)
            ('PNG Images', 'png'), ('JPEG Images', 'jpg'), ('JPEG Alt', 'jpeg'),
            ('WebP Images', 'webp'), ('GIF Images', 'gif'), ('BMP Images', 'bmp'),
            # Archive Files 📦
            ('ZIP Archives', 'zip'), ('TAR Files', 'tar'), ('GZ Archives', 'gz'),
            ('RAR Archives', 'rar'), ('7Z Archives', '7z'),
            # Text Files 📝
            ('Text Files', 'txt'), ('Caption Files', 'caption'), ('Tag Files', 'tags'),
            ('JSON Files', 'json'), ('YAML Files', 'yaml'), ('YAML Alt', 'yml'),
            ('CSV Files', 'csv'), ('Log Files', 'log'),
            # AI Model Files 🤖
            ('SafeTensors', 'safetensors'), ('PyTorch Models', 'pt'), ('PyTorch Legacy', 'pth'),
            ('ONNX Models', 'onnx'), ('Checkpoints', 'ckpt'), ('Binary Files', 'bin'),
            # All Files
            ('All Files', '*')
        ]
        self.current_directory = os.getcwd()
        self._create_widgets()
        self._bind_events()
        self._update_files(None)  # Initial file list update

    def _create_widgets(self):
        # --- Upload Destination ---
        self.dest_info_html = HTML(value="<b>📁 Upload Destination</b>")
        self.dest_folder_text = Text(
            value="./datasets/my_dataset", 
            placeholder='e.g., ./datasets/my_dataset', 
            description='Destination:', 
            style={'description_width': 'initial'},
            layout=Layout(width='auto', flex='1 1 auto')
        )
        self.create_folder_checkbox = Checkbox(
            value=True, 
            description='Create folder if missing', 
            indent=False
        )

        # --- File Selection ---
        self.file_section_html = HTML(value="<b>🗂️ File Selection & Source</b>")
        self.file_type_dropdown = Dropdown(
            options=self.file_types, 
            value='jpg',  # Default to images for datasets
            description='File Type:', 
            style={'description_width': 'initial'}
        )
        self.sort_by_dropdown = Dropdown(
            options=['name', 'date'], 
            value='name', 
            description='Sort By:', 
            style={'description_width': 'initial'}
        )
        
        self.directory_label = Label(value="Source Directory:", layout=Layout(width='auto'))
        self.directory_text = Text(
            value=self.current_directory,
            description="",
            style={'description_width': '0px'},
            layout=Layout(width="auto", flex='1 1 auto')
        )
        self.directory_update_btn = Button(
            description='🔄 List Files', 
            button_style='info', 
            tooltip='Change source directory and refresh file list', 
            layout=Layout(width='auto')
        )

        # --- Upload Settings ---
        self.upload_section_html = HTML(value="<b>🚀 Upload Settings</b>")
        self.overwrite_checkbox = Checkbox(
            value=False, 
            description='Overwrite existing files', 
            indent=False
        )
        self.preserve_structure_checkbox = Checkbox(
            value=False, 
            description='Preserve folder structure', 
            indent=False
        )
        self.clear_after_checkbox = Checkbox(
            value=True, 
            description='Clear output after upload', 
            indent=False
        )

        # --- Action Buttons ---
        self.clear_output_button = Button(
            description='🧹 Clear Cache & Reset', 
            button_style='warning', 
            tooltip='Clear widget cache and reset selections (preserves folder)', 
            layout=Layout(width='auto')
        )

        # --- File Picker & Output ---
        self.file_picker_selectmultiple = SelectMultiple(
            options=[], 
            description='Files:', 
            layout=Layout(width="98%", height="200px"), 
            style={'description_width': 'initial'}
        )
        self.output_area = Output(
            layout=Layout(
                padding='10px', 
                border='1px solid #ccc', 
                margin_top='10px', 
                width='98%', 
                max_height='400px', 
                overflow_y='auto'
            )
        )

        # --- Progress Display Area (initially hidden) ---
        self.current_file_label = Label(value="N/A")
        self.file_count_label = Label(value="File 0/0")
        self.progress_bar = FloatProgress(
            value=0, min=0, max=100, 
            description='Overall:', 
            bar_style='info', 
            layout=Layout(width='85%', margin='0 5px 0 5px')
        )
        self.progress_percent_label = Label(value="0%")

        self.progress_display_box = VBox([
            HBox([Label("Current File:", layout=Layout(width='100px')), self.current_file_label], 
                 layout=Layout(width='auto')),
            HBox([Label("File Count:", layout=Layout(width='100px')), self.file_count_label], 
                 layout=Layout(width='auto')),
            HBox([self.progress_bar, self.progress_percent_label], 
                 layout=Layout(align_items='center', width='auto'))
        ], layout=Layout(
            visibility='hidden', 
            margin='10px 0 10px 0', 
            padding='10px', 
            border='1px solid #ddd', 
            width='98%'
        ))

    def _bind_events(self):
        self.directory_update_btn.on_click(self._update_directory_and_files)
        self.clear_output_button.on_click(self._clear_cache_and_reset)
        self.file_type_dropdown.observe(self._update_files, names='value')
        self.sort_by_dropdown.observe(self._update_files, names='value')

    def _clear_cache_and_reset(self, _):
        """
        Cache cleaner that resets widget state without touching the destination folder.
        Fixes the 'upload once' limitation by clearing all widget caches and selections.
        """
        # Clear output area
        self.output_area.clear_output(wait=True)
        
        # Reset progress display
        self.progress_display_box.layout.visibility = 'hidden'
        self.progress_bar.value = 0
        self.progress_percent_label.value = "0%"
        self.current_file_label.value = "N/A"
        self.file_count_label.value = "File 0/0"
        
        # Clear file selections but keep directory
        self.file_picker_selectmultiple.value = ()  # Clear selections
        
        # Reset checkboxes to defaults (but preserve folder path!)
        self.overwrite_checkbox.value = False
        self.preserve_structure_checkbox.value = False
        self.clear_after_checkbox.value = True
        
        # Force refresh file list to clear any cached state
        self._update_files(None)
        
        # Provide feedback
        with self.output_area:
            print("🧹 Widget cache cleared! Ready for fresh uploads.")
            print(f"📁 Destination folder preserved: {self.dest_folder_text.value}")

    def _update_directory_and_files(self, _):
        new_dir = self.directory_text.value.strip()
        if not new_dir:
            with self.output_area:
                clear_output(wait=True)
                print(f"📂 Current directory remains: {self.current_directory}")
            self._update_files(None)
            return

        if os.path.isdir(new_dir):
            self.current_directory = os.path.abspath(new_dir)
            self.directory_text.value = self.current_directory
            self._update_files(None)
        else:
            with self.output_area:
                clear_output(wait=True)
                print(f"❌ Invalid Directory: {new_dir}")

    def _update_files(self, _):
        file_extension = self.file_type_dropdown.value
        self.output_area.clear_output(wait=True)
        try:
            if file_extension == '*':
                glob_pattern = "*"
            else:
                glob_pattern = f"*.{file_extension}"
            
            if not os.path.isdir(self.current_directory):
                with self.output_area:
                    print(f"⚠️ Source directory '{self.current_directory}' is not valid. Please set a valid path.")
                self.file_picker_selectmultiple.options = []
                return

            found_paths = list(Path(self.current_directory).glob(glob_pattern))
            
            valid_files_info = []
            for p in found_paths:
                if p.is_symlink():
                    continue
                if not p.is_file():
                    continue
                
                if self.sort_by_dropdown.value == 'date':
                    sort_key = p.stat().st_mtime
                else:
                    sort_key = p.name.lower()
                valid_files_info.append((str(p), sort_key))

            if self.sort_by_dropdown.value == 'date':
                valid_files_info.sort(key=lambda item: item[1], reverse=True)
            else:
                valid_files_info.sort(key=lambda item: item[1])
            
            sorted_file_paths = [item[0] for item in valid_files_info]
            self.file_picker_selectmultiple.options = sorted_file_paths
            
            with self.output_area:
                if not sorted_file_paths:
                    if file_extension == '*':
                        print(f"🤷 No files found in '{self.current_directory}'.")
                    else:
                        print(f"🤷 No '.{file_extension}' files found in '{self.current_directory}'.")
                else:
                    file_type_desc = "files" if file_extension == '*' else f"'.{file_extension}' files"
                    print(f"✨ Found {len(sorted_file_paths)} {file_type_desc} in '{self.current_directory}'. Select files to upload.")

        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"❌ Error listing files: {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()

    def _format_size(self, size_bytes):
        if size_bytes < 0: return "Invalid size"
        if size_bytes == 0: return "0 B"
        units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
        i = math.floor(math.log(size_bytes, 1024)) if size_bytes > 0 else 0
        if i >= len(units): i = len(units) - 1
        
        s = round(size_bytes / (1024 ** i), 2)
        return f"{s} {units[i]}"

    def _print_file_info(self, file_path_str, index, total_files):
        file_path = Path(file_path_str)
        try:
            file_size = file_path.stat().st_size
            self.output_area.append_stdout(
                f"📦 Uploading {index}/{total_files}: {file_path.name} ({self._format_size(file_size)})\n"
            )
        except FileNotFoundError:
            self.output_area.append_stdout(
                f"⚠️ File not found (may have been moved/deleted): {file_path_str}\n"
            )

    def _upload_files_handler(self, _):
        dest_folder = self.dest_folder_text.value.strip()
        
        if not dest_folder:
            with self.output_area:
                clear_output(wait=True)
                print("❗ Please specify a destination folder.")
            return

        selected_file_paths = list(self.file_picker_selectmultiple.value)

        if not selected_file_paths:
            with self.output_area:
                clear_output(wait=True)
                print("📝 Nothing selected for upload. Please select files from the list.")
            return

        self.output_area.clear_output(wait=True)
        self.output_area.append_stdout(f"🎯 Preparing to upload to: {os.path.abspath(dest_folder)}\n")
        
        # Create destination folder if needed
        if self.create_folder_checkbox.value:
            try:
                os.makedirs(dest_folder, exist_ok=True)
                self.output_area.append_stdout(f"📁 Created/verified destination folder: {dest_folder}\n")
            except Exception as e:
                self.output_area.append_stdout(f"❌ Could not create destination folder: {e}\n")
                return
        elif not os.path.exists(dest_folder):
            self.output_area.append_stdout(f"❌ Destination folder does not exist: {dest_folder}\n")
            return

        self.progress_display_box.layout.visibility = 'visible'
        self.progress_bar.value = 0
        self.progress_percent_label.value = "0%"
        self.current_file_label.value = "Initializing..."
        
        total_files = len(selected_file_paths)
        self.file_count_label.value = f"File 0/{total_files}"
        
        success_count = 0
        for idx, local_file_path_str in enumerate(selected_file_paths, 1):
            current_file_path = Path(local_file_path_str)
            self.current_file_label.value = current_file_path.name
            self.file_count_label.value = f"File {idx}/{total_files}"
            self._print_file_info(local_file_path_str, idx, total_files)
            
            start_time = time.time()
            try:
                if not current_file_path.exists():
                    self.output_area.append_stdout(f"❌ SKIPPED: File '{current_file_path.name}' not found.\n")
                    continue

                # Determine destination path
                if self.preserve_structure_checkbox.value:
                    # Keep relative path structure
                    relative_path = current_file_path.relative_to(self.current_directory)
                    dest_file_path = Path(dest_folder) / relative_path
                    # Create subdirectories if needed
                    dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Flat structure - just filename
                    dest_file_path = Path(dest_folder) / current_file_path.name

                # Check for existing file
                if dest_file_path.exists() and not self.overwrite_checkbox.value:
                    self.output_area.append_stdout(f"⚠️ SKIPPED: '{dest_file_path.name}' already exists (overwrite disabled)\n")
                    continue

                # Copy the file
                shutil.copy2(str(current_file_path), str(dest_file_path))
                
                duration = time.time() - start_time
                self.output_area.append_stdout(f"✅ Copied '{current_file_path.name}' in {duration:.1f}s\n")
                success_count += 1

                # Yield to UI every 5 files to prevent freezing
                if idx % 5 == 0:
                    time.sleep(0.1)

            except Exception as e:
                self.output_area.append_stdout(f"❌ Error copying {current_file_path.name}: {type(e).__name__} - {str(e)}\n")
                import traceback
                with self.output_area:
                    traceback.print_exc()
                self.output_area.append_stdout("\n")
            finally:
                percentage = int((idx / total_files) * 100)
                self.progress_bar.value = percentage
                self.progress_percent_label.value = f"{percentage}%"

        self.output_area.append_stdout(f"\n✨ Upload process completed. {success_count}/{total_files} files successfully copied. ✨\n")
        if success_count > 0:
            self.output_area.append_stdout(f"🎉 Files copied to: {os.path.abspath(dest_folder)}\n")

        self.current_file_label.value = "All tasks complete."
        if self.clear_after_checkbox.value:
            time.sleep(3)  # Shorter than HF uploader since local is faster
            self.output_area.clear_output(wait=True)
            self.progress_display_box.layout.visibility = 'hidden'

    def display(self):
        """Arranges and displays the widgets."""
        dest_box = HBox([
            self.dest_folder_text, 
            self.create_folder_checkbox
        ], layout=Layout(width='100%', align_items='center'))
        
        dir_select_box = HBox([
            self.directory_label, 
            self.directory_text, 
            self.directory_update_btn
        ], layout=Layout(width='100%', align_items='center'))

        file_opts_box = HBox([
            self.file_type_dropdown, 
            self.sort_by_dropdown
        ], layout=Layout(width='100%', justify_content='space-between'))
        
        upload_opts_box = HBox([
            self.overwrite_checkbox, 
            self.preserve_structure_checkbox,
            self.clear_after_checkbox
        ], layout=Layout(margin='5px 0'))
        
        action_buttons_box = HBox([
            self.upload_button, 
            self.clear_output_button
        ], layout=Layout(margin='10px 0 0 0', justify_content='flex-start'))

        main_layout = VBox([
            self.dest_info_html, dest_box,
            HTML("<hr>"),
            self.file_section_html, file_opts_box, dir_select_box,
            self.file_picker_selectmultiple,
            HTML("<hr>"),
            self.upload_section_html, upload_opts_box,
            action_buttons_box,
            self.progress_display_box,
            self.output_area
        ], layout=Layout(width='700px', padding='10px', border='1px solid lightgray'))
        
        display(main_layout)

# Usage:
# uploader = LocalFileUploader()
# uploader.display()