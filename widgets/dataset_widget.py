# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/dataset_widget.py
import asyncio
import glob
import logging
import os
import re
import zipfile
from datetime import datetime

import ipywidgets as widgets
from IPython.display import display

from core.dataset_manager import DatasetManager
from core.file_upload_manager import FileUploadManager
from core.logging_config import setup_file_logging

# Setup external logging for widget debugging
setup_file_logging()
logger = logging.getLogger(__name__)


class DatasetWidget:
    def __init__(self, dataset_manager=None, file_upload_manager=None):
        # Use dependency injection - accept manager instances or create defaults
        if dataset_manager is None:
            dataset_manager = DatasetManager()  # Uses lazy ModelManager loading now!
        if file_upload_manager is None:
            file_upload_manager = FileUploadManager()

        self.manager = dataset_manager
        self.file_manager = file_upload_manager
        # Race condition guard: prevent observer conflicts during programmatic changes
        self._programmatic_change = False
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Dataset Manager."""
        header_icon = "📊"
        header_main = widgets.HTML(f"<h2>{header_icon} 2. Dataset Manager</h2>")

        # --- Unified Dataset Setup Section ---
        dataset_setup_desc = widgets.HTML("""<h3>📁 Dataset Setup</h3>
        <p><strong>🎯 Choose your preferred method to set up your training dataset.</strong></p>
        <p><strong>📋 Available Methods:</strong></p>
        <ul>
        <li><strong>URL/ZIP Download:</strong> Download and extract from URLs or file paths</li>
        <li><strong>Direct Image Upload:</strong> Upload individual images directly (perfect for small datasets)</li>
        <li><strong>Gallery-DL Scraper:</strong> Download images from various sites using gallery-dl (anime/character datasets)</li>
        </ul>""")

        # Dataset method selection
        self.dataset_method = widgets.RadioButtons(
            options=[('📥 URL/ZIP Download', 'url'), ('📁 Direct Image Upload', 'upload'), ('🔍 Gallery-DL Scraper', 'gallery_dl')],
            description='Method:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        # Shared dataset directory (will be auto-populated)
        self.dataset_directory = widgets.Text(
            description="Dataset Directory:",
            placeholder="Will be set automatically or enter manually",
            layout=widgets.Layout(width='99%')
        )

        # URL/ZIP Download Section
        url_method_desc = widgets.HTML("<h4>📥 URL/ZIP Download Setup</h4>")

        self.project_name = widgets.Text(
            description="Project Name:",
            placeholder="e.g., my_awesome_character (no spaces or special chars)",
            layout=widgets.Layout(width='99%')
        )

        self.dataset_url = widgets.Text(
            description="Dataset URL/Path:",
            placeholder="/path/to/dataset.zip or HuggingFace URL",
            layout=widgets.Layout(width='99%')
        )

        self.url_download_button = widgets.Button(description="🚀 Download & Extract Dataset", button_style='success')

        self.url_method_box = widgets.VBox([
            url_method_desc,
            self.project_name,
            self.dataset_url,
            self.url_download_button
        ])

        # Direct Upload Section
        upload_method_desc = widgets.HTML("<h4>📁 Direct Image Upload Setup</h4>")

        self.folder_name = widgets.Text(
            description="Folder Name:",
            placeholder="e.g., character_name (repeat count will be added automatically)",
            layout=widgets.Layout(width='70%')
        )

        self.folder_repeats = widgets.IntText(
            description="Repeat Count:",
            value=10,
            min=1,
            max=100,
            layout=widgets.Layout(width='25%'),
            style={'description_width': 'initial'}
        )

        self.create_folder_button = widgets.Button(
            description="📁 Create Folder",
            button_style='info',
            layout=widgets.Layout(width='25%')
        )

        folder_creation_box = widgets.VBox([
            widgets.HBox([self.folder_name, self.folder_repeats]),
            widgets.HTML("<small><i>💡 Final folder will be: <strong>{repeat_count}_{folder_name}</strong> (Kohya format)</i></small>"),
            self.create_folder_button
        ])

        self.file_upload = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif,.zip', # Added .zip
            multiple=True,
            description='Select Images/ZIP:', # Updated description
            layout=widgets.Layout(width='99%'),
            # Add max file size to prevent memory issues
            max_size=10*1024*1024  # 10MB per file limit
        )

        self.upload_images_button = widgets.Button(
            description="🚀 Upload Images",
            button_style='success',
            disabled=True
        )

        self.upload_zip_button = widgets.Button(
            description="📦 Upload & Extract ZIP",
            button_style='primary',
            disabled=True
        )

        self.reset_upload_button = widgets.Button(
            description="🔄 Force Reset Upload",
            button_style='warning',
            tooltip="If uploads fail or the widget seems stuck, click this to reset it.",
            disabled=False
        )

        self.cancel_upload_button = widgets.Button(
            description="🚫 Cancel Upload",
            button_style='danger',
            disabled=True
        )

        # Progress tracking widgets
        self.upload_progress = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#007bff'},
            layout=widgets.Layout(width='99%', visibility='hidden')
        )

        self.upload_status_label = widgets.Label(
            value="Ready to upload",
            layout=widgets.Layout(visibility='hidden')
        )

        # Create button row for upload, cancel, and reset
        upload_button_row = widgets.HBox([
            self.upload_images_button,
            self.upload_zip_button,
            self.cancel_upload_button,
            self.reset_upload_button
        ])

        # Progress display
        progress_box = widgets.VBox([
            self.upload_progress,
            self.upload_status_label
        ], layout=widgets.Layout(margin='10px 0px'))

        self.upload_method_box = widgets.VBox([
            upload_method_desc,
            folder_creation_box,
            self.file_upload,
            upload_button_row,
            progress_box
        ])

        # Gelbooru Scraper Section
        # Gallery-DL Scraper Section
        <div style='background: var(--jp-info-color-light); color: var(--jp-info-color-dark); padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🌐 Supported Sites:</strong> Gelbooru, Danbooru, Pixiv, Twitter, etc. (see <a href="https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md" target="_blank">supported sites</a>)<br>
        <strong>💡 Tips:</strong> Use appropriate tags, specify limits, and consider subfolders for organization.
        </div>""")

        self.gallery_dl_site = widgets.Dropdown(
            options=[('Gelbooru', 'gelbooru'), ('Danbooru', 'danbooru'), ('Pixiv', 'pixiv'), ('Twitter', 'twitter'), ('Custom URL (enter below)', 'custom')],
            description='Site:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.gallery_dl_tags = widgets.Text(
            description="Tags:",
            placeholder="e.g., 1girl, blue_hair, long_hair (comma separated)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_limit_range = widgets.Text(
            value="1-100",
            description='Limit Range:',
            placeholder="e.g., 1-100 (downloads first 100 images)",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.gallery_dl_custom_url = widgets.Text(
            description="Custom URL:",
            placeholder="Full URL for custom site (if 'Custom URL' selected above)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_sub_folder = widgets.Text(
            description="Subfolder:",
            placeholder="e.g., my_character (creates a subfolder within dataset dir)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_additional_args = widgets.Text(
            description="Additional Args:",
            placeholder="e.g., --no-part --write-info-json",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_write_tags = widgets.Checkbox(
            value=True,
            description="Write Tags (creates .txt files)",
            indent=False
        )

        self.gallery_dl_use_aria2c = widgets.Checkbox(
            value=True,
            description="Use aria2c (faster downloads)",
            indent=False
        )

        self.gallery_dl_button = widgets.Button(
            description="🔍 Start Gallery-DL Scrape",
            button_style='info',
            layout=widgets.Layout(width='25%')
        )

        self.gallery_dl_method_box = widgets.VBox([
            gallery_dl_method_desc,
            self.gallery_dl_site,
            self.gallery_dl_tags,
            self.gallery_dl_limit_range,
            self.gallery_dl_custom_url,
            self.gallery_dl_sub_folder,
            self.gallery_dl_additional_args,
            self.gallery_dl_write_tags,
            self.gallery_dl_use_aria2c,
            self.gallery_dl_button
        ])

        # Status and output (shared)
        self.dataset_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>Status:</strong> Select a method to begin</div>")
        self.dataset_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        # Show/hide method boxes based on selection
        def on_method_change(change):
            if change['new'] == 'url':
                self.url_method_box.layout.display = 'block'
                self.upload_method_box.layout.display = 'none'
                self.gallery_dl_method_box.layout.display = 'none'
            elif change['new'] == 'upload':
                self.url_method_box.layout.display = 'none'
                self.upload_method_box.layout.display = 'block'
                self.gallery_dl_method_box.layout.display = 'none'
            else:  # gallery_dl
                self.url_method_box.layout.display = 'none'
                self.upload_method_box.layout.display = 'none'
                self.gallery_dl_method_box.layout.display = 'block'

        self.dataset_method.observe(on_method_change, names='value')
        # Initialize with URL method visible
        self.upload_method_box.layout.display = 'none'
        self.gallery_dl_method_box.layout.display = 'none'

        dataset_setup_box = widgets.VBox([
            dataset_setup_desc,
            self.dataset_method,
            self.dataset_directory,
            self.url_method_box,
            self.upload_method_box,
            self.gallery_dl_method_box,
            self.dataset_status,
            self.dataset_output
        ])

        # FiftyOne integration buttons REMOVED - feature on pause

        # --- File Renaming Section ---
        rename_desc = widgets.HTML("""<h3>📝 File Renaming</h3>
        <p>Rename your dataset files to fix UTF-8 issues and create consistent naming. <strong>Caption files are automatically renamed too!</strong></p>
        <p><strong>📋 Naming Patterns:</strong></p>
        <ul>
        <li><strong>Simple Numbering:</strong> MyProject_001.jpg, MyProject_002.jpg</li>
        <li><strong>Sanitized Original:</strong> MyProject_CleanedName.jpg (removes special chars)</li>
        <li><strong>With Timestamp:</strong> MyProject_20240807_001.jpg</li>
        </ul>
        <p><em>💡 This fixes common training issues with special characters and UTF-8!</em></p>""")

        self.rename_project_name = widgets.Text(
            description="Project Name:",
            placeholder="e.g., MyCharacter (will become MyCharacter_001.jpg)",
            layout=widgets.Layout(width='99%')
        )

        self.rename_pattern = widgets.Dropdown(
            options=[
                ('Simple Numbering (MyProject_001.jpg)', 'numbered'),
                ('Sanitized Original (MyProject_CleanName.jpg)', 'sanitized'),
                ('With Timestamp (MyProject_20240807_001.jpg)', 'timestamp')
            ],
            value='numbered',
            description='Naming Pattern:'
        )

        self.rename_start_number = widgets.IntText(
            value=1,
            description="Start Number:",
            style={'description_width': 'initial'}
        )

        self.preview_rename_button = widgets.Button(
            description="👁️ Preview Changes",
            button_style='info'
        )

        self.rename_files_button = widgets.Button(
            description="📝 Rename Files",
            button_style='warning',
            disabled=True
        )

        self.rename_output = widgets.Output()

        rename_controls = widgets.HBox([
            self.rename_start_number,
            self.preview_rename_button,
            self.rename_files_button
        ])

        rename_box = widgets.VBox([
            rename_desc,
            self.rename_project_name,
            self.rename_pattern,
            rename_controls,
            self.rename_output
        ])

        # --- Tagging Section ---
        tagging_desc = widgets.HTML("""<h3>▶️ Image Tagging</h3>
        <p>Automatically generate captions for your images using AI taggers. <strong>Anime method uses SmilingWolf's WD14 taggers</strong>, Photo method uses BLIP captioning.</p>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>📋 Available SmilingWolf WD14 Models:</strong><br>
        <strong>V3 Models (Latest, Recommended):</strong><br>
        • <strong>EVA02 Large v3:</strong> Best quality, 315M params (default)<br>
        • <strong>ViT Large v3:</strong> High quality, updated training<br>
        • <strong>SwinV2 v3:</strong> Balanced performance, latest<br>
        • <strong>ConvNeXT v3:</strong> Good speed, updated<br>
        • <strong>ViT v3:</strong> Fast tagging, latest<br><br>
        <strong>V2 Models (Stable):</strong><br>
        • <strong>SwinV2 v2, MoAT v2, ConvNeXT v2:</strong> Proven stable options<br><br>
        <em>🔄 Models auto-download from HuggingFace on first use!</em>
        </div>""")

        # Dataset directory will be auto-populated from setup section

        self.tagging_method = widgets.Dropdown(
            options=['anime', 'photo'],
            description='Method:',
            style={'description_width': 'initial'}
        )

        # Enhanced tagger models with descriptions (mix of v2 and v3 models that actually exist)
        tagger_models = {
            "SmilingWolf/wd-eva02-large-tagger-v3": "EVA02 Large v3 (Best Quality, Newer)",
            "SmilingWolf/wd-vit-large-tagger-v3": "ViT Large v3 (High Quality, Updated)",
            "SmilingWolf/wd-swinv2-tagger-v3": "SwinV2 v3 (Balanced, Latest)",
            "SmilingWolf/wd-convnext-tagger-v3": "ConvNeXT v3 (Good Speed, Updated)",
            "SmilingWolf/wd-vit-tagger-v3": "ViT v3 (Fast, Latest)",
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2": "SwinV2 v2 (Stable)",
            "SmilingWolf/wd-v1-4-moat-tagger-v2": "MoAT v2 (Alternative)",
            "SmilingWolf/wd-v1-4-convnext-tagger-v2": "ConvNeXT v2 (Stable)"
        }

        self.tagger_model = widgets.Dropdown(
            options=list(tagger_models.keys()),
            value="SmilingWolf/wd-eva02-large-tagger-v3",
            description='Tagger Model:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        # Show model description
        self.tagger_desc = widgets.HTML()
        def update_tagger_desc(change):
            if change['new'] in tagger_models:
                self.tagger_desc.value = f"<small><i>{tagger_models[change['new']]}</i></small>"
        self.tagger_model.observe(update_tagger_desc, names='value')
        update_tagger_desc({'new': self.tagger_model.value})  # Initial desc

        self.tagging_threshold = widgets.FloatSlider(
            value=0.25, min=0.1, max=1.0, step=0.05,
            description='Threshold:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.blacklist_tags = widgets.Text(
            description="Blacklist Tags:",
            placeholder="e.g., 1girl,solo,standing (comma separated)",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.caption_extension = widgets.Dropdown(
            options=['.txt', '.caption'],
            value='.txt',
            description='Caption Extension:',
            style={'description_width': 'initial'}
        )

        self.tagging_button = widgets.Button(description="🏷️ Start Tagging", button_style='primary')

        self.cancel_tagging_button = widgets.Button(
            description="🚫 Cancel Tagging",
            button_style='danger',
            disabled=True
        )

        # Tagging progress widgets
        self.tagging_progress = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#007bff'},
            layout=widgets.Layout(width='99%', visibility='hidden')
        )

        self.tagging_progress_label = widgets.Label(
            value="Ready to tag images",
            layout=widgets.Layout(visibility='hidden')
        )

        # Button row for tagging and cancel
        tagging_button_row = widgets.HBox([
            self.tagging_button,
            self.cancel_tagging_button
        ])

        # Progress display
        tagging_progress_box = widgets.VBox([
            self.tagging_progress,
            self.tagging_progress_label
        ], layout=widgets.Layout(margin='10px 0px'))

        self.tagging_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>Status:</strong> Ready</div>")
        self.tagging_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        tagging_box = widgets.VBox([
            tagging_desc,
            self.tagging_method,
            self.tagger_model,
            self.tagger_desc,
            self.tagging_threshold,
            self.blacklist_tags,
            self.caption_extension,
            tagging_button_row,
            tagging_progress_box,
            self.tagging_status,
            self.tagging_output
        ])

        # --- Dataset Cleanup ---
        cleanup_desc = widgets.HTML("""<h3>▶️ Dataset Cleanup</h3>
        <p>Clean up old files when re-tagging datasets or starting fresh.</p>
        
        <p><strong>⚠️ What gets cleaned:</strong></p>
        <ul>
        <li><strong>.txt files:</strong> Caption files from previous tagging</li>
        <li><strong>.npz files:</strong> Cached latents from previous training</li>
        <li><strong>.caption files:</strong> Alternative caption format</li>
        <li><strong>Non-image files:</strong> Model files (.safetensors, .ckpt), configs (.json, .yaml), etc.</li>
        </ul>
        <p><em>🎯 Use this when you want to re-tag a dataset or clean up accidentally extracted files</em></p>""")

        # Dataset directory will be auto-populated from setup section

        # Cleanup options
        self.cleanup_text_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .txt caption files",
            indent=False
        )

        self.cleanup_npz_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .npz cached latents",
            indent=False
        )

        self.cleanup_caption_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .caption files",
            indent=False
        )

        self.cleanup_non_images = widgets.Checkbox(
            value=False,
            description="🗑️ Remove non-image files (.safetensors, .ckpt, .bin, .json, .yaml, etc.)",
            indent=False
        )

        self.cleanup_preview = widgets.Checkbox(
            value=True,
            description="👀 Preview files before deletion (recommended)",
            indent=False
        )

        self.cleanup_button = widgets.Button(description="🧹 Clean Dataset", button_style='warning')
        self.cleanup_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>Status:</strong> Ready</div>")
        self.cleanup_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        cleanup_box = widgets.VBox([
            cleanup_desc,
            self.cleanup_text_files,
            self.cleanup_npz_files,
            self.cleanup_caption_files,
            self.cleanup_non_images,
            self.cleanup_preview,
            self.cleanup_button,
            self.cleanup_status,
            self.cleanup_output
        ])

        # --- HuggingFace Upload Section ---
        upload_desc = widgets.HTML("""<h3>🤗 HuggingFace Dataset Upload</h3>
        <div style='background: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🚀 Share your dataset with the community!</strong><br>
        Upload your prepared dataset to HuggingFace Hub for sharing and backup.<br>
        <strong>📋 Requirements:</strong> HuggingFace account with WRITE token
        </div>""")

        self.upload_dataset_path = widgets.Text(
            description="Dataset Path:",
            placeholder="Path to your dataset folder (auto-filled from current dataset)",
            layout=widgets.Layout(width='99%')
        )

        self.upload_dataset_name = widgets.Text(
            description="Repository Name:",
            placeholder="e.g., my-character-dataset (lowercase, hyphens allowed)",
            layout=widgets.Layout(width='99%')
        )

        self.upload_hf_token = widgets.Password(
            description="HF Token:",
            placeholder="Your HuggingFace WRITE token",
            layout=widgets.Layout(width='99%')
        )

        token_help = widgets.HTML("""
        <div style='background: #fff3cd; padding: 8px; border-radius: 5px; margin: 5px 0;'>
        🔑 Get your <strong>WRITE</strong> token <a href='https://huggingface.co/settings/tokens' target='_blank'>here</a>
        </div>
        """)

        self.upload_orgs_name = widgets.Text(
            description="Organization:",
            placeholder="Leave empty to use personal account, or enter org name",
            layout=widgets.Layout(width='99%')
        )

        self.upload_description = widgets.Textarea(
            description="Description:",
            placeholder="Describe your dataset (optional but recommended)",
            layout=widgets.Layout(width='99%', height='80px')
        )

        self.upload_make_private = widgets.Checkbox(
            value=False,
            description="🔒 Make repository private",
            indent=False
        )

        self.upload_button = widgets.Button(description="🚀 Upload to HuggingFace", button_style='info')
        self.upload_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>Status:</strong> Ready</div>")
        self.upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        upload_box = widgets.VBox([
            upload_desc,
            self.upload_dataset_path,
            self.upload_dataset_name,
            self.upload_hf_token,
            token_help,
            self.upload_orgs_name,
            self.upload_description,
            self.upload_make_private,
            self.upload_button,
            self.upload_status,
            self.upload_output
        ])

        # --- Image Utilities Section ---
        image_utils_desc = widgets.HTML("""<h3>🖼️ Image Format Conversion</h3>
        <p>Convert image formats for better compatibility and storage optimization.</p>
        <p><strong>⚠️ Note:</strong> This only converts formats (WebP ↔ JPG ↔ PNG). No resizing or cropping is performed to preserve training data quality.</p>
        """)

        self.format_selector = widgets.Dropdown(
            options=[('Convert to JPG', 'jpg'), ('Convert to PNG', 'png'), ('Convert to WebP', 'webp')],
            value='jpg',
            description='Target Format:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        # Format conversion only - no resizing

        self.conversion_quality = widgets.IntSlider(
            value=95,
            min=85,
            max=100,
            step=5,
            description='Quality:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.convert_images_button = widgets.Button(
            description="🔄 Convert Image Formats",
            button_style='info'
        )
        self.convert_images_button.on_click(self.run_format_conversion)

        self.image_utils_output = widgets.Output()

        image_utils_box = widgets.VBox([
            image_utils_desc,
            self.format_selector,
            self.conversion_quality,
            self.convert_images_button,
            self.image_utils_output
        ])

        # Tag Curation functionality is integrated into Dataset Setup section

        # --- Advanced Caption Management ---
        caption_desc = widgets.HTML("""<h3>▶️ Advanced Caption Management</h3>
        <p>Professional tag curation tools for cleaning and organizing your caption files.</p>
        <p><strong>🎯 Available Tools:</strong></p>
        <ul>
        <li><strong>Trigger Words:</strong> Add activation tags to all captions</li>
        <li><strong>Tag Removal:</strong> Remove unwanted tags from all captions</li>
        <li><strong>Search & Replace:</strong> Advanced bulk tag replacement with AND/OR logic</li>
        <li><strong>Sort & Deduplicate:</strong> Organize tags alphabetically and remove duplicates</li>
        </ul>""")

        # Basic trigger word management
        basic_management_desc = widgets.HTML("<h4>🎯 Basic Tag Management</h4>")

        self.trigger_word = widgets.Text(
            description="Trigger Word:",
            placeholder="e.g., my_character, myart_style",
            layout=widgets.Layout(width='99%')
        )

        self.add_trigger_button = widgets.Button(description="➕ Add Trigger Word", button_style='success')

        self.remove_tags = widgets.Text(
            description="Remove Tags:",
            placeholder="e.g., 1girl,solo (comma separated)",
            layout=widgets.Layout(width='99%')
        )

        self.remove_tags_button = widgets.Button(description="➖ Remove Tags", button_style='warning')

        # Review/display tags
        self.review_tags_button = widgets.Button(description="👀 Review Tags", button_style='info')

        # Advanced search and replace
        advanced_management_desc = widgets.HTML("<h4>🔍 Advanced Search & Replace</h4>")

        self.search_tags = widgets.Text(
            description="Search Tags:",
            placeholder="e.g., 1girl solo standing (space or comma separated)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.replace_with = widgets.Text(
            description="Replace With:",
            placeholder="e.g., woman alone (leave empty to remove)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.search_mode = widgets.Dropdown(
            options=[('AND - all search tags must match', 'AND'), ('OR - any search tag matches', 'OR')],
            value='OR',
            description='Search Mode:',
            style={'description_width': 'initial'}
        )

        self.search_replace_button = widgets.Button(description="🔄 Search & Replace", button_style='info')

        # Sort and deduplicate
        organization_desc = widgets.HTML("<h4>📝 Tag Organization</h4>")

        self.sort_alphabetically = widgets.Checkbox(
            value=True,
            description="Sort tags alphabetically",
            indent=False
        )

        self.remove_duplicates = widgets.Checkbox(
            value=True,
            description="Remove duplicate tags",
            indent=False
        )

        self.organize_tags_button = widgets.Button(description="📋 Organize Tags", button_style='info')

        self.caption_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>Status:</strong> Ready</div>")
        self.caption_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        caption_box = widgets.VBox([
            caption_desc,
            basic_management_desc,
            self.trigger_word,
            self.add_trigger_button,
            self.remove_tags,
            widgets.HBox([self.remove_tags_button, self.review_tags_button]),
            advanced_management_desc,
            self.search_tags,
            self.replace_with,
            self.search_mode,
            self.search_replace_button,
            organization_desc,
            self.sort_alphabetically,
            self.remove_duplicates,
            self.organize_tags_button,
            self.caption_status,
            self.caption_output
        ])

        # --- Accordion ---
        self.accordion = widgets.Accordion(children=[
            dataset_setup_box,
            rename_box,
            tagging_box,
            caption_box,
            cleanup_box,
            upload_box,
            image_utils_box
        ])
        self.accordion.set_title(0, "📁 Dataset Setup")
        self.accordion.set_title(1, "📝 File Renaming")
        self.accordion.set_title(2, "🏷️ Image Tagging")
        self.accordion.set_title(3, "📝 Advanced Caption Management")
        self.accordion.set_title(4, "🧹 Dataset Cleanup")
        self.accordion.set_title(5, "🤗 HuggingFace Upload")
        self.accordion.set_title(6, "🖼️ Image Utilities")

        self.widget_box = widgets.VBox([header_main, self.accordion])

        # --- Button Events ---
        self.url_download_button.on_click(self.run_url_download)
        self.create_folder_button.on_click(self.run_create_folder)
        self.upload_images_button.on_click(self._handle_async_upload)
        self.upload_zip_button.on_click(self._handle_async_zip_upload)
        self.cancel_upload_button.on_click(self.cancel_current_upload)
        
        # Upload button click handlers attached
        # Gallery-dl scraper functionality integrated into URL download
        # self.gelbooru_button.on_click(self.run_gallery_dl_scraper)  # Button removed, functionality integrated
        self.preview_rename_button.on_click(self.run_preview_rename)
        self.rename_files_button.on_click(self.run_rename_files)
        self.tagging_button.on_click(self.run_tagging)
        # Removed cancel button - not needed for synchronous tagging
        self.cleanup_button.on_click(self.run_cleanup)
        self.add_trigger_button.on_click(self.run_add_trigger)
        self.remove_tags_button.on_click(self.run_remove_tags)
        self.review_tags_button.on_click(self.run_review_tags)
        self.search_replace_button.on_click(self.run_search_replace)
        self.organize_tags_button.on_click(self.run_organize_tags)
        self.upload_button.on_click(self.run_upload_to_huggingface)

        # --- File Upload Observer ---
        self.file_upload.observe(self.on_file_upload_change, names='value')

        # Reset upload button event handler (button created earlier)
        self.reset_upload_button.on_click(self.reset_upload_widget)

        # Auto-detect existing dataset directories on init
        self.auto_detect_existing_datasets()

    def on_file_upload_change(self, change):
        """Handle file upload selection changes"""
        try:
            # Race condition guard: ignore programmatic changes (like cache resets)
            if self._programmatic_change:
                logger.debug("Ignoring programmatic file upload change to prevent race condition")
                return
                
            new_files = change.get('new', [])
            
            if new_files:  # Files have been selected
                file_count = len(new_files)
                
                # --- Proactive Warning Logic ---
                warning_message = ""
                if file_count > 100:
                    warning_message = f"<div style='background: #fff3cd; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Warning:</strong> You have selected {file_count} files. Uploading many files at once can be unreliable on remote connections. Consider uploading in smaller batches of 50-100.</div>"

                large_files = [f for f in new_files if f['size'] > 50 * 1024 * 1024] # 50MB
                if large_files:
                    warning_message += f"<div style='background: #fff3cd; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107; margin-top: 5px;'><strong>⚠️ Warning:</strong> You have selected files larger than 50MB. This may fail. Please upload smaller files.</div>"

                if file_count > 0:
                    # Check if a folder path is specified
                    folder_path = self.dataset_directory.value.strip()
                    folder_exists = bool(folder_path) and os.path.exists(folder_path)

                    # Determine if any selected file is a ZIP (case-insensitive)
                    is_zip_selected = any(f['name'].lower().endswith('.zip') for f in new_files)

                    if folder_exists:
                        # Folder exists - ready to upload
                        self.upload_images_button.disabled = is_zip_selected # Disable image upload if zip is selected
                        self.upload_zip_button.disabled = not is_zip_selected # Enable zip upload only if zip is selected

                        if is_zip_selected:
                            status_message = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> {file_count} file(s) selected. Ready to upload and extract ZIP to '{folder_path}'.</div>"
                        else:
                            status_message = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> {file_count} file(s) selected. Ready to upload images to '{folder_path}'.</div>"
                        self.dataset_status.value = warning_message + status_message

                    elif folder_path:
                        # Folder path specified but doesn't exist - suggest creating it
                        self.upload_images_button.disabled = True
                        self.upload_zip_button.disabled = True
                        status_message = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>📁 Status:</strong> {file_count} file(s) selected. Folder '{folder_path}' doesn't exist - click 'Create Folder' first.</div>"
                        self.dataset_status.value = warning_message + status_message
                    else:
                        # No folder path specified - auto-suggest creating folder
                        self.upload_images_button.disabled = True
                        self.upload_zip_button.disabled = True
                        # Try to suggest a folder name based on the files
                        suggested_name = "uploaded_dataset"
                        if is_zip_selected:
                            zip_name = next(f['name'] for f in new_files if f['name'].lower().endswith('.zip'))
                            suggested_name = zip_name.replace('.zip', '_dataset')

                        if not self.folder_name.value.strip():
                            self.folder_name.value = suggested_name

                        status_message = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>📁 Status:</strong> {file_count} file(s) selected. Click 'Create Folder' first, then upload.</div>"
                        self.dataset_status.value = warning_message + status_message
                else:
                    self.upload_images_button.disabled = True
                    self.upload_zip_button.disabled = True
            else:
                # No files selected - reset everything
                self.upload_images_button.disabled = True
                self.upload_zip_button.disabled = True
                if not self.dataset_directory.value.strip():
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>Status:</strong> Select images/ZIP or create a folder to start uploading.</div>"
        except Exception as e:
            logger.error(f"ERROR in file upload observer: {e}")
            # Ensure buttons are disabled if error occurs
            self.upload_images_button.disabled = True
            self.upload_zip_button.disabled = True

    def run_url_download(self, b):
        """Handle URL/ZIP download method"""
        self.dataset_output.clear_output()
        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Downloading and extracting...</div>"
        with self.dataset_output:
            project_name = self.project_name.value.strip()
            dataset_url = self.dataset_url.value.strip()

            if not project_name:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a project name.</div>"
                logger.warning("URL download failed: no project name provided")
                return

            if not dataset_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a dataset URL.</div>"
                logger.warning("URL download failed: no dataset URL provided")
                return

            # Sanitize project name
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
            if clean_name != project_name:
                print(f"📝 Cleaned project name: '{project_name}' → '{clean_name}'")
                project_name = clean_name

            project_dir = f"datasets/{project_name}"

            # Create project directory
            os.makedirs(project_dir, exist_ok=True)

            success = self.manager.extract_dataset(dataset_url, project_dir)
            if success:
                # Update shared dataset directory
                self.dataset_directory.value = project_dir

                # Count images
                try:
                    from core.image_utils import count_images_in_directory
                    image_count = count_images_in_directory(project_dir)
                    print(f"📁 Found {image_count} images in {project_dir}")
                    print(f"📝 Dataset directory set: {project_dir}")

                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Downloaded {image_count} images to {project_dir}</div>"
                except Exception as e:
                    print(f"⚠️ Image counting error: {e}")
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Download complete.</div>"
            else:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Download and extraction failed. Check logs.</div>"

    def run_create_folder(self, b):
        """Create a new folder for image upload using Kohya's repeat count format"""
        
        self.dataset_output.clear_output()
        folder_name = self.folder_name.value.strip()
        repeat_count = self.folder_repeats.value

        if not folder_name:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a folder name.</div>"
            return

        # Sanitize folder name
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
        if clean_name != folder_name:
            folder_name = clean_name

        # Create folder in Kohya's expected format: {repeat_count}_{folder_name}
        kohya_folder_name = f"{repeat_count}_{folder_name}"
        folder_path = f"datasets/{kohya_folder_name}"

        with self.dataset_output:
            try:
                os.makedirs(folder_path, exist_ok=True)
                print(f"✅ Created Kohya-compatible folder: {folder_path}")
                print(f"📊 Repeat count: {repeat_count} (auto-detected by training)")

                # Update shared dataset directory
                self.dataset_directory.value = os.path.abspath(folder_path)

                # Check if files are already selected and enable appropriate buttons
                if self.file_upload.value:
                    file_count = len(self.file_upload.value)
                    is_zip_selected = any(f['name'].lower().endswith('.zip') for f in self.file_upload.value)

                    # Enable the right button based on file type
                    self.upload_images_button.disabled = is_zip_selected
                    self.upload_zip_button.disabled = not is_zip_selected

                    if is_zip_selected:
                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder created! {file_count} file(s) ready to upload (ZIP detected).</div>"
                    else:
                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder created! {file_count} file(s) ready to upload.</div>"
                else:
                    self.upload_images_button.disabled = True
                    self.upload_zip_button.disabled = True
                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder '{folder_path}' created! Now select images/ZIP to upload.</div>"

            except Exception as e:
                logger.error(f"Failed to create folder: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to create folder.</div>"

    def _set_upload_ui_state(self, is_uploading: bool):
        """Centralizes enabling/disabling of upload UI components."""
        self.upload_images_button.disabled = is_uploading
        self.upload_zip_button.disabled = is_uploading
        self.create_folder_button.disabled = is_uploading
        self.cancel_upload_button.disabled = not is_uploading
        # The reset button should always be available as an escape hatch
        # self.reset_upload_button.disabled = is_uploading

    def _handle_async_upload(self, b):
        """Wrapper to handle async upload function with proper error handling"""
        self._set_upload_ui_state(is_uploading=True)

        try:
            captured_files = []
            for file_info in self.file_upload.value:
                captured_files.append({
                    'name': file_info['name'],
                    'content': file_info['content'].tobytes() # Read into bytes NOW
                })
            
            if not captured_files:
                self.dataset_status.value = "❌ Status: Please select images to upload."
                self._update_upload_button_states() # Re-evaluate button states
                return
        except Exception as e:
            logger.error(f"FATAL: Failed to read file data from widget. State is likely corrupt. Error: {e}")
            self.dataset_status.value = "❌ Critical Error: Could not read files from the widget. Please click 'Force Reset Upload' and hard-refresh your browser tab (Cmd+Shift+R)."
            self._update_upload_button_states() # Re-evaluate button states
            return

        # Now, pass the captured data to the async function.
        task = asyncio.ensure_future(self.run_upload_images(b, captured_files=captured_files))
        # Store task reference to allow cancellation
        self._current_upload_task = task
        task.add_done_callback(self._upload_task_done)

    def _handle_async_zip_upload(self, b):
        """Wrapper to handle async ZIP upload function with proper error handling"""
        self._set_upload_ui_state(is_uploading=True)

        try:
            zip_file_data = None
            for file_info in self.file_upload.value:
                if file_info['name'].lower().endswith('.zip'):
                    zip_file_data = {
                        'name': file_info['name'],
                        'content': file_info['content'].tobytes() # Read into bytes NOW
                    }
                    break # Found the first zip, stop looking
            
            if not zip_file_data:
                self.dataset_status.value = "❌ Status: Please select a ZIP file to upload."
                self._update_upload_button_states() # Re-evaluate button states
                return
        except Exception as e:
            logger.error(f"FATAL: Failed to read file data from widget. State is likely corrupt. Error: {e}")
            self.dataset_status.value = "❌ Critical Error: Could not read files from the widget. Please click 'Force Reset Upload' and hard-refresh your browser tab (Cmd+Shift+R)."
            self._update_upload_button_states() # Re-evaluate button states
            return

        # Now, pass the captured data to the async function.
        task = asyncio.ensure_future(self.run_upload_zip(b, zip_file_data=zip_file_data))
        # Store task reference to allow cancellation
        self._current_upload_task = task
        task.add_done_callback(self._upload_task_done)

    def _upload_task_done(self, task):
        """Callback when upload task completes or fails"""
        try:
            if task.cancelled():
                with self.dataset_output:
                    print("🚫 Upload cancelled by user")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>🚫 Status:</strong> Upload cancelled</div>"
            elif task.exception():
                with self.dataset_output:
                    print(f"❌ Upload failed: {task.exception()}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload failed with error</div>"
        except Exception as e:
            logger.error(f"Error in upload task callback: {e}")
        finally:
            # Clear task reference
            self._current_upload_task = None
            # Aggressively reset the widget to prevent stale state
            self._programmatic_change = True
            try:
                self.file_upload.value = ()
            finally:
                self._programmatic_change = False
            # Re-enable upload buttons
            self._update_upload_button_states()

    def cancel_current_upload(self, b):
        """Cancel the currently running upload task"""
        if hasattr(self, '_current_upload_task') and self._current_upload_task and not self._current_upload_task.done():
            self._current_upload_task.cancel()
            logger.info("Upload cancellation requested")
        else:
            with self.dataset_output:
                print("ℹ️ No active upload to cancel")


    def _hide_tagging_progress_widgets(self):
        """Hide the tagging progress widgets"""
        if hasattr(self, 'tagging_progress'):
            self.tagging_progress.layout.visibility = 'hidden'
        if hasattr(self, 'tagging_progress_label'):
            self.tagging_progress_label.layout.visibility = 'hidden'

    def _update_upload_button_states(self):
        """Update upload button states based on current conditions"""
        # Check if upload is in progress
        upload_in_progress = (hasattr(self, '_current_upload_task') and
                            self._current_upload_task and
                            not self._current_upload_task.done())

        # Disable upload buttons during upload, enable cancel button
        if upload_in_progress:
            self.upload_images_button.disabled = True
            self.upload_zip_button.disabled = True
            if hasattr(self, 'cancel_upload_button'):
                self.cancel_upload_button.disabled = False
        else:
            # Re-enable based on file selection and folder existence
            if hasattr(self, 'file_upload') and self.file_upload.value:
                folder_path = self.dataset_directory.value.strip()
                folder_exists = bool(folder_path) and os.path.exists(folder_path)
                if folder_exists:
                    is_zip_selected = any(f['name'].lower().endswith('.zip') for f in self.file_upload.value)
                    self.upload_images_button.disabled = is_zip_selected
                    self.upload_zip_button.disabled = not is_zip_selected
                else:
                    self.upload_images_button.disabled = True
                    self.upload_zip_button.disabled = True
            else:
                self.upload_images_button.disabled = True
                self.upload_zip_button.disabled = True

            if hasattr(self, 'cancel_upload_button'):
                self.cancel_upload_button.disabled = True

    def _hide_progress_widgets(self):
        """Hide the upload progress widgets"""
        if hasattr(self, 'upload_progress'):
            self.upload_progress.layout.visibility = 'hidden'
        if hasattr(self, 'upload_status_label'):
            self.upload_status_label.layout.visibility = 'hidden'



    def _reset_upload_cache(self, b):
        """
        Clear the FileUpload widget cache to allow multiple uploads.
        Fixes the 'upload once' limitation by clearing file_upload.value.
        Uses race condition guard to prevent observer conflicts.
        """
        # Set guard flag to prevent observer race condition
        self._programmatic_change = True
        try:
            self.file_upload.value = ()  # Clear the cached files (won't trigger observer now)
            self.dataset_status.value = "🔄 Upload cache cleared! Ready for new uploads."
        finally:
            # Always clear the guard flag even if something goes wrong
            self._programmatic_change = False
        
        # Clear output area for fresh start
        self.dataset_output.clear_output()
        
        # Provide user feedback
        with self.dataset_output:
            print("🔄 Upload widget reset complete!")
            print("📁 You can now select and upload new files.")
            print("💡 The previous upload data has been cleared from memory.")

    async def run_upload_images(self, b, captured_files: list):
        """Upload multiple images to the created folder using FileUploadManager."""
        self.dataset_output.clear_output()
        self.upload_progress.layout.visibility = 'visible'
        self.upload_status_label.layout.visibility = 'visible'
        self.upload_progress.value = 0
        self.upload_status_label.value = "Initializing upload..."
        self._update_upload_button_states()

        upload_folder = self.dataset_directory.value
        if not upload_folder:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please create a folder first.</div>"
            self._hide_progress_widgets()
            return

        try:
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>⚙️ Status:</strong> Uploading {len(captured_files)} images...</div>"
            
            loop = asyncio.get_event_loop()
            # Use run_in_executor to run the synchronous manager method in a thread pool
            success, message = await loop.run_in_executor(
                None, # Use default executor
                self.file_manager.upload_images, 
                captured_files, 
                upload_folder
            )

            self.dataset_status.value = message # Display message from manager

            if success:
                # Clear the widget on success
                self._programmatic_change = True
                try:
                    self.file_upload.value = ()
                finally:
                    self._programmatic_change = False
                logger.info("Image upload successful, widget cache cleared.")

        except Exception as e:
            logger.error(f"Error during image upload via manager: {e}")
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> An unexpected error occurred: {e}</div>"
        finally:
            self._hide_progress_widgets()
            self._update_upload_button_states()

    async def run_upload_zip(self, b, zip_file_data: dict):
        """Upload and extract a ZIP file using FileUploadManager."""
        self.dataset_output.clear_output()
        self.upload_progress.layout.visibility = 'visible'
        self.upload_status_label.layout.visibility = 'visible'
        self.upload_progress.value = 0
        self.upload_status_label.value = "Initializing ZIP upload..."
        self._update_upload_button_states()

        upload_folder = self.dataset_directory.value
        if not upload_folder:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please create a folder first.</div>"
            self._hide_progress_widgets()
            return

        try:
            zip_filename = zip_file_data['name']
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>⚙️ Status:</strong> Uploading and extracting {zip_filename}...</div>"

            loop = asyncio.get_event_loop()
            # Use run_in_executor to run the synchronous manager method in a thread pool
            success, message = await loop.run_in_executor(
                None, # Use default executor
                self.file_manager.upload_and_extract_zip, 
                zip_file_data, 
                upload_folder
            )

            self.dataset_status.value = message # Display message from manager

            if success:
                # Clear the widget on success
                self._programmatic_change = True
                try:
                    self.file_upload.value = ()
                finally:
                    self._programmatic_change = False
                logger.info("ZIP upload successful, widget cache cleared.")

        except Exception as e:
            logger.error(f"Error during ZIP upload via manager: {e}")
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> An unexpected error occurred: {e}</div>"
        finally:
            self._hide_progress_widgets()
            self._update_upload_button_states()

    def reset_upload_widget(self, b):
        """Reset the file upload widget to clear any cached state"""
        # Clear the upload widget value using race condition guard
        self._programmatic_change = True
        try:
            self.file_upload.value = ()
        finally:
            self._programmatic_change = False

        # Reset button states
        self.upload_images_button.disabled = True
        self.upload_zip_button.disabled = True

        # Clear folder name if it was auto-suggested
        if self.folder_name.value in ['uploaded_dataset'] or '_dataset' in self.folder_name.value:
            self.folder_name.value = ""

        # Update status
        if self.dataset_directory.value.strip():
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔄 Status:</strong> Upload reset. Folder '{self.dataset_directory.value}' ready for new files.</div>"
        else:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔄 Status:</strong> Upload reset. Create a folder and select files to upload.</div>"

        logger.info("Upload widget reset")

    def auto_detect_existing_datasets(self):
        """Auto-detect existing dataset directories and set the first one as default"""

        # Look for existing datasets directories
        datasets_pattern = "datasets/*"
        existing_dirs = [d for d in glob.glob(datasets_pattern) if os.path.isdir(d)]

        if existing_dirs:
            # Sort by modification time (most recent first)
            existing_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            most_recent = existing_dirs[0]

            # Auto-populate if dataset directory is empty
            if not self.dataset_directory.value.strip():
                self.dataset_directory.value = most_recent
                logger.info(f"Auto-detected existing dataset: {most_recent}")

                # Update status to show detection
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔍 Status:</strong> Auto-detected dataset folder '{most_recent}'. Select files to upload here.</div>"

    # FiftyOne methods REMOVED - feature on pause
    # def launch_fiftyone_explorer() and def apply_fiftyone_curation() removed

    def run_gallery_dl_scraper(self, b):
        """Handle image scraping using gallery-dl"""
        self.dataset_output.clear_output()
        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Preparing gallery-dl scrape...</div>"

        with self.dataset_output:
            site = self.gallery_dl_site.value
            tags = self.gallery_dl_tags.value.strip()
            limit_range = self.gallery_dl_limit_range.value.strip()
            custom_url = self.gallery_dl_custom_url.value.strip()
            sub_folder = self.gallery_dl_sub_folder.value.strip()
            additional_args = self.gallery_dl_additional_args.value.strip()
            write_tags = self.gallery_dl_write_tags.value
            use_aria2c = self.gallery_dl_use_aria2c.value

            # Determine the final dataset directory
            # Use the folder_name and repeat_count from the direct upload section
            # This ensures Kohya-compatible folder naming for scraped datasets
            folder_name_base = self.folder_name.value.strip() if self.folder_name.value.strip() else "scraped_dataset"
            repeat_count = self.folder_repeats.value
            kohya_folder_name = f"{repeat_count}_{folder_name_base}"
            dataset_dir = f"datasets/{kohya_folder_name}"

            # If custom URL is selected, ensure it's provided
            if site == 'custom' and not custom_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please provide a Custom URL when 'Custom URL' site is selected.</div>"
                logger.warning("Gallery-DL scrape failed: custom URL not provided")
                return

            if not tags and not custom_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter tags or a Custom URL.</div>"
                logger.warning("Gallery-DL scrape failed: no tags or custom URL provided")
                return

            try:
                os.makedirs(dataset_dir, exist_ok=True)
                self.dataset_directory.value = dataset_dir # Update shared dataset directory

                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Scraping with gallery-dl...</div>"

                success = self.manager.scrape_with_gallery_dl(
                    site=site,
                    tags=tags,
                    dataset_dir=dataset_dir,
                    limit_range=limit_range,
                    write_tags=write_tags,
                    use_aria2c=use_aria2c,
                    custom_url=custom_url,
                    sub_folder=sub_folder,
                    additional_args=additional_args
                )

                if success:
                    try:
                        from core.image_utils import count_images_in_directory
                        image_count = count_images_in_directory(dataset_dir)
                        print(f"\n📁 Total images in dataset: {image_count}")
                        print(f"📝 Dataset directory set: {dataset_dir}")

                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Scrape complete. Found {image_count} images.</div>"
                    except Exception as e:
                        print(f"⚠️ Image counting error: {e}")
                        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Scrape complete.</div>"
                else:
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Gallery-DL scrape failed. Check logs.</div>"

            except Exception as e:
                logger.error(f"Failed to run gallery-dl scraper: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Scraper encountered an error.</div>"

    # Removed async tagging methods - using synchronous version only

    def run_tagging(self, b):
        """Fallback synchronous tagging method"""
        self.tagging_output.clear_output()
        self.tagging_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Starting {self.tagging_method.value} tagging...</div>"
        with self.tagging_output:
            if not self.dataset_directory.value:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Tagging failed: no dataset directory set")
                return

            print(f"🏷️ Starting {self.tagging_method.value} tagging with {self.tagger_model.value.split('/')[-1]}...")
            print(f"📁 Dataset: {self.dataset_directory.value}")

            # Enhanced tagging with more options
            success = self.manager.tag_images(
                self.dataset_directory.value,
                self.tagging_method.value,
                self.tagger_model.value,
                self.tagging_threshold.value,
                blacklist_tags=self.blacklist_tags.value,
                caption_extension=self.caption_extension.value
            )
            if success:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tagging complete.</div>"
            else:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Tagging failed. Check logs.</div>"

    def run_cleanup(self, b):
        """Clean up old caption and cache files from dataset directory"""
        self.cleanup_output.clear_output()
        self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Scanning for files to clean...</div>"

        with self.cleanup_output:
            dataset_dir = self.dataset_directory.value.strip()

            if not dataset_dir:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Cleanup failed: no dataset directory set")
                return

            print(f"🧩 Cleaning dataset: {dataset_dir}")

            if not os.path.exists(dataset_dir):
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Dataset directory does not exist.</div>"
                logger.error(f"Directory does not exist: {dataset_dir}")
                return

            # Find files to clean
            files_to_clean = []

            if self.cleanup_text_files.value:
                txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
                files_to_clean.extend([(f, "caption file") for f in txt_files])

            if self.cleanup_npz_files.value:
                npz_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
                files_to_clean.extend([(f, "cached latent") for f in npz_files])

            if self.cleanup_caption_files.value:
                caption_files = glob.glob(os.path.join(dataset_dir, "*.caption"))
                files_to_clean.extend([(f, "caption file") for f in caption_files])

            if self.cleanup_non_images.value:
                # Define common image extensions to preserve
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

                # Find all files in directory
                all_files = glob.glob(os.path.join(dataset_dir, "*"))

                for file_path in all_files:
                    if os.path.isfile(file_path):  # Skip directories
                        file_ext = os.path.splitext(file_path)[1].lower()

                        # If it's not an image extension and not already in our cleanup list
                        if file_ext not in image_extensions:
                            # Skip files we're already cleaning (txt, npz, caption)
                            if file_ext not in {'.txt', '.npz', '.caption'}:
                                # Identify file type for better reporting
                                if file_ext in {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}:
                                    file_type = "model file"
                                elif file_ext in {'.json', '.yaml', '.yml', '.toml', '.ini'}:
                                    file_type = "config file"
                                elif file_ext in {'.py', '.sh', '.bat', '.cmd'}:
                                    file_type = "script file"
                                elif file_ext in {'.zip', '.rar', '.7z', '.tar', '.gz'}:
                                    file_type = "archive file"
                                elif file_ext in {'.log', '.txt'}:
                                    file_type = "log file"
                                else:
                                    file_type = f"non-image file ({file_ext})"

                                files_to_clean.append((file_path, file_type))

            if not files_to_clean:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> No files found to clean. Directory is already clean!</div>"
                print("✅ No files found to clean. Directory is already clean!")
                return

            print(f"🧹 Found {len(files_to_clean)} files to clean in: {dataset_dir}")
            print("="*60)

            # Preview files
            if self.cleanup_preview.value:
                print("👀 PREVIEW - Files that will be deleted:")
                for file_path, file_type in files_to_clean:
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    size_mb = file_size / (1024 * 1024)
                    print(f"  🗑️ {file_name} ({file_type}, {size_mb:.2f} MB)")

                print("\n⚠️ PREVIEW MODE: No files were actually deleted.")
                print("💡 Uncheck 'Preview files before deletion' to actually delete these files.\n")

                self.cleanup_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>👀 Status:</strong> Preview complete - found {len(files_to_clean)} files to clean.</div>"
                return

            # Actually delete files
            deleted_count = 0
            total_size_freed = 0

            print("🗑️ DELETING FILES:")
            for file_path, file_type in files_to_clean:
                try:
                    if os.path.exists(file_path):
                        file_name = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_count += 1
                        total_size_freed += file_size
                        print(f"  ✅ Deleted: {file_name} ({file_type})")
                    else:
                        print(f"  ⚠️ File not found: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {os.path.basename(file_path)}: {e}")

            size_freed_mb = total_size_freed / (1024 * 1024)
            print("\n🎉 Cleanup complete!")
            print(f"📊 Deleted {deleted_count} files")
            print(f"💾 Freed {size_freed_mb:.2f} MB of disk space")
            print(f"📁 Dataset directory: {dataset_dir}")

            if deleted_count > 0:
                self.cleanup_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Cleaned {deleted_count} files ({size_freed_mb:.1f} MB freed).</div>"
            else:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> No files were deleted (may have been errors).</div>"

    def run_add_trigger(self, b):
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Adding trigger word...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Add trigger failed: no dataset directory set")
                return

            if not self.trigger_word.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a trigger word.</div>"
                logger.warning("Add trigger failed: no trigger word specified")
                return

            print(f"➕ Adding trigger word '{self.trigger_word.value}' to captions...")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            success = self.manager.add_trigger_word(self.dataset_directory.value, self.trigger_word.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Trigger word added.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to add trigger word. Check logs.</div>"

    def run_remove_tags(self, b):
        """Remove specified tags from all caption files"""
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Removing tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Remove tags failed: no dataset directory set")
                return

            if not self.remove_tags.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to remove.</div>"
                logger.warning("Remove tags failed: no tags specified")
                return

            print(f"➖ Removing tags '{self.remove_tags.value}' from captions...")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            success = self.manager.remove_tags(self.dataset_directory.value, self.remove_tags.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tags removed.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to remove tags. Check logs.</div>"

    def run_search_replace(self, b):
        """Advanced search and replace functionality for tags"""
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Processing search and replace...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Search replace failed: no dataset directory set")
                return

            if not self.search_tags.value.strip():
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to search for.</div>"
                logger.warning("Search replace failed: no search tags specified")
                return

            print("🔍 Search and replace operation:")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            print(f"🔍 Search tags: {self.search_tags.value}")
            print(f"🔄 Replace with: '{self.replace_with.value}' (empty = remove)")
            print(f"🎯 Search mode: {self.search_mode.value}")

            success = self.manager.search_and_replace_tags(
                dataset_dir=self.dataset_directory.value,
                search_tags=self.search_tags.value,
                replace_with=self.replace_with.value,
                search_mode=self.search_mode.value
            )

            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Search and replace complete.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Search and replace failed. Check logs.</div>"

    def run_organize_tags(self, b):
        """Sort tags alphabetically and remove duplicates"""
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Organizing tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                logger.warning("Tag organization failed: no dataset directory set")
                return

            operations = []
            if self.sort_alphabetically.value:
                operations.append("sort alphabetically")
            if self.remove_duplicates.value:
                operations.append("remove duplicates")

            if not operations:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> No operations selected.</div>"
                logger.warning("Tag organization failed: no operations selected")
                return

            print("📋 Tag organization:")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            print(f"🔧 Operations: {', '.join(operations)}")

            success = self.manager.sort_and_deduplicate_tags(
                dataset_dir=self.dataset_directory.value,
                sort_alphabetically=self.sort_alphabetically.value,
                remove_duplicates=self.remove_duplicates.value
            )

            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tag organization complete.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Tag organization failed. Check logs.</div>"

    def run_preview_rename(self, b):
        """Preview file renaming changes"""
        with self.rename_output:
            self.rename_output.clear_output()

            if not self.rename_project_name.value.strip():
                logger.warning("Preview rename failed: no project name provided")
                return

            if not self.dataset_directory.value.strip():
                logger.warning("Preview rename failed: no dataset directory set")
                return

            project_name = self.rename_project_name.value.strip()
            pattern = self.rename_pattern.value
            start_num = self.rename_start_number.value

            print(f"🔍 Preview renaming for: {self.dataset_directory.value}")
            print(f"📝 Pattern: {pattern}, Project: {project_name}, Start: {start_num}")

            preview_data = self.manager.preview_rename_files(
                self.dataset_directory.value,
                project_name,
                pattern,
                start_num
            )

            if preview_data:
                print(f"\n📋 Preview of {len(preview_data)} files to be renamed:")
                print("=" * 60)
                for i, item in enumerate(preview_data[:10]):  # Show first 10
                    status = "📸" if not item['has_caption'] else "📸📝"
                    print(f"{status} {item['old_name']} → {item['new_name']}")
                    if item['has_caption']:
                        print(f"    {item['old_caption']} → {item['new_caption']}")

                if len(preview_data) > 10:
                    print(f"... and {len(preview_data) - 10} more files")

                print("=" * 60)
                print("💡 If this looks good, click 'Rename Files' to proceed!")

                # Enable the rename button
                self.rename_files_button.disabled = False

                # Store preview data for actual rename
                self._current_preview = preview_data
            else:
                self.rename_files_button.disabled = True

    def run_rename_files(self, b):
        """Actually rename the files"""
        with self.rename_output:
            self.rename_output.clear_output()

            if not hasattr(self, '_current_preview') or not self._current_preview:
                logger.warning("Rename files failed: no preview data available")
                return

            project_name = self.rename_project_name.value.strip()
            pattern = self.rename_pattern.value

            print("📝 Renaming files...")
            print(f"🎯 Project: {project_name}, Pattern: {pattern}")

            success = self.manager.rename_dataset_files(
                self.dataset_directory.value,
                project_name,
                pattern,
                self.rename_start_number.value,
                self._current_preview
            )

            if success:
                print("✅ File renaming completed successfully!")
                print("💡 Caption files were automatically renamed to match!")
                # Clear preview data and disable button
                self._current_preview = None
                self.rename_files_button.disabled = True
            else:
                print("❌ File renaming failed. Check the logs above.")

    def run_upload_to_huggingface(self, b):
        """Upload dataset to HuggingFace Hub"""
        self.upload_output.clear_output()
        self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>🚀 Status:</strong> Starting upload...</div>"

        with self.upload_output:
            # Auto-fill dataset path if empty
            if not self.upload_dataset_path.value and self.dataset_directory.value:
                self.upload_dataset_path.value = self.dataset_directory.value
                print(f"📁 Auto-filled dataset path: {self.dataset_directory.value}")

            # Validation
            if not self.upload_dataset_path.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Dataset path required.</div>"
                logger.warning("HuggingFace upload failed: no dataset path specified")
                return

            if not self.upload_dataset_name.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Repository name required.</div>"
                logger.warning("HuggingFace upload failed: no repository name specified")
                return

            if not self.upload_hf_token.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> HuggingFace token required.</div>"
                logger.warning("HuggingFace upload failed: no token provided")
                return

            print("🤗 Starting HuggingFace dataset upload...")
            print("=" * 60)

            try:
                success = self.manager.upload_dataset_to_huggingface(
                    dataset_path=self.upload_dataset_path.value,
                    dataset_name=self.upload_dataset_name.value,
                    hf_token=self.upload_hf_token.value,
                    orgs_name=self.upload_orgs_name.value,
                    make_private=self.upload_make_private.value,
                    description=self.upload_description.value
                )

                if success:
                    self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Upload completed successfully!</div>"
                    print("\n🎉 Upload completed successfully!")
                    print("🤗 Your dataset is now available on HuggingFace Hub!")

                    # Create shareable info
                    repo_id = f"{self.upload_orgs_name.value.strip() or 'your-username'}/{self.upload_dataset_name.value}"
                    print("\n📋 Dataset Details:")
                    print(f"   🔗 URL: https://huggingface.co/datasets/{repo_id}")
                    print(f"   📝 Name: {self.upload_dataset_name.value}")
                    print(f"   🔒 Privacy: {'Private' if self.upload_make_private.value else 'Public'}")

                else:
                    self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload failed.</div>"
                    print("❌ Upload failed. Check the error messages above.")

            except Exception as e:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload error.</div>"
                print(f"💥 Unexpected error during upload: {str(e)}")
                print("💡 Please check your token permissions and try again.")

    def run_format_conversion(self, b):
        """Convert image formats in the current dataset directory."""
        self.image_utils_output.clear_output()
        dataset_path = self.dataset_directory.value.strip()

        target_format = self.format_selector.value
        quality = self.conversion_quality.value

        if not dataset_path:
            logger.warning("Format conversion failed: no dataset directory set")
            return

        if not os.path.exists(dataset_path):
            logger.error(f"Format conversion failed: dataset directory not found: {dataset_path}")
            return

        self.image_utils_output.append_stdout(f"🔄 Starting format conversion to {target_format.upper()} (quality: {quality})...\n")
        try:
            # Use the existing image conversion functionality (need to implement in manager)
            success = self.manager.convert_image_formats(dataset_path, target_format, quality)
            if success:
                self.image_utils_output.append_stdout("✅ Format conversion complete.\n")
            else:
                self.image_utils_output.append_stdout("❌ Format conversion failed. Check logs.\n")
        except Exception as e:
            self.image_utils_output.append_stdout(f"❌ An error occurred during format conversion: {e}\n")

    def run_review_tags(self, b):
        """Display the generated tags for review"""
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #e2e3e5; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>📋 Status:</strong> Displaying tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                logger.warning("Review tags failed: no dataset directory specified")
                self.caption_status.value = "<div style='background: #f8d7da; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> No dataset directory specified.</div>"
                return

            # Display the tags
            success = self.manager.display_dataset_tags(self.dataset_directory.value)
            if success:
                self.caption_status.value = "<div style='background: #d4edda; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tags displayed successfully.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8d7da; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to display tags.</div>"



    def display(self):
        display(self.widget_box)
