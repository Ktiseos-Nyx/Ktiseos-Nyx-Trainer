# widgets/dataset_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.dataset_manager import DatasetManager
from core.managers import ModelManager

class DatasetWidget:
    def __init__(self):
        # Note: This is not ideal, but for now we'll instantiate a ModelManager
        # to pass to the DatasetManager. A better approach would be to use a single
        # manager instance for the whole application.
        self.manager = DatasetManager(ModelManager())
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Dataset Manager."""
        header_icon = "📊"
        header_main = widgets.HTML(f"<h2>{header_icon} 2. Dataset Manager</h2>")

        # --- Project Setup Section ---
        project_desc = widgets.HTML("""<h3>▶️ Project Setup</h3>
        <p><strong>🎯 One-stop project creation!</strong> Enter your project name and dataset URL - we'll create the folder, download, extract, and calculate training parameters automatically!</p>
        """)
        
        self.project_name = widgets.Text(
            description="Project Name:", 
            placeholder="e.g., my_awesome_character (no spaces or special chars)", 
            layout=widgets.Layout(width='99%')
        )
        
        self.project_dataset_url = widgets.Text(
            description="Dataset URL:", 
            placeholder="/path/to/dataset.zip or HuggingFace URL", 
            layout=widgets.Layout(width='99%')
        )
        
        self.create_project_button = widgets.Button(description="🚀 Create Project & Setup Dataset", button_style='success')
        self.project_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>Status:</strong> Ready</div>")
        self.project_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        project_box = widgets.VBox([
            project_desc,
            self.project_name,
            self.project_dataset_url, 
            self.create_project_button,
            self.project_status,
            self.project_output
        ])

        # --- Manual Upload Section ---
        upload_desc = widgets.HTML("<h3>▶️ Manual Upload & Extract</h3><p>For manual control over dataset extraction.</p>")
        self.upload_path = widgets.Text(description="Zip Path:", placeholder="/path/to/dataset.zip or HuggingFace URL", layout=widgets.Layout(width='99%'))
        self.extract_dir = widgets.Text(description="Extract to:", placeholder="e.g., my_dataset_folder", layout=widgets.Layout(width='99%'))
        self.upload_button = widgets.Button(description="Upload & Extract", button_style='primary')
        self.upload_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>Status:</strong> Ready</div>")
        self.upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        upload_box = widgets.VBox([upload_desc, self.upload_path, self.extract_dir, self.upload_button, self.upload_status, self.upload_output])

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
        
        self.tagging_dataset_dir = widgets.Text(
            description="Dataset Dir:", 
            placeholder="e.g., my_dataset_folder", 
            layout=widgets.Layout(width='99%')
        )
        
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
        self.tagging_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>Status:</strong> Ready</div>")
        self.tagging_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        tagging_box = widgets.VBox([
            tagging_desc,
            self.tagging_dataset_dir, 
            self.tagging_method, 
            self.tagger_model, 
            self.tagger_desc,
            self.tagging_threshold, 
            self.blacklist_tags,
            self.caption_extension,
            self.tagging_button, 
            self.tagging_status,
            self.tagging_output
        ])

        # --- Caption Management ---
        caption_desc = widgets.HTML("<h3>▶️ Caption Management</h3><p>Add trigger words to activate your LoRA, or clean up captions by removing unwanted tags.</p>")
        
        self.caption_dataset_dir = widgets.Text(
            description="Dataset Dir:", 
            placeholder="e.g., my_dataset_folder", 
            layout=widgets.Layout(width='99%')
        )
        
        # Trigger word management
        self.trigger_word = widgets.Text(
            description="Trigger Word:", 
            placeholder="e.g., my_character, myart_style", 
            layout=widgets.Layout(width='99%')
        )
        
        self.add_trigger_button = widgets.Button(description="➕ Add Trigger Word", button_style='success')
        
        # Tag removal
        self.remove_tags = widgets.Text(
            description="Remove Tags:",
            placeholder="e.g., 1girl,solo (comma separated)",
            layout=widgets.Layout(width='99%')
        )
        
        self.remove_tags_button = widgets.Button(description="➖ Remove Tags", button_style='warning')
        
        self.caption_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>Status:</strong> Ready</div>")
        self.caption_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        caption_box = widgets.VBox([
            caption_desc,
            self.caption_dataset_dir, 
            self.trigger_word, 
            self.add_trigger_button,
            self.remove_tags,
            self.remove_tags_button,
            self.caption_status,
            self.caption_output
        ])

        # --- Accordion ---
        self.accordion = widgets.Accordion(children=[
            project_box,
            upload_box,
            tagging_box,
            caption_box
        ])
        self.accordion.set_title(0, "🚀 Project Setup")
        self.accordion.set_title(1, "▶️ Manual Upload & Extract")
        self.accordion.set_title(2, "▶️ Image Tagging")
        self.accordion.set_title(3, "▶️ Caption Management")

        self.widget_box = widgets.VBox([header_main, self.accordion])

        # --- Button Events ---
        self.create_project_button.on_click(self.run_create_project)
        self.upload_button.on_click(self.run_upload)
        self.tagging_button.on_click(self.run_tagging)
        self.add_trigger_button.on_click(self.run_add_trigger)
        self.remove_tags_button.on_click(self.run_remove_tags)

    def run_upload(self, b):
        self.upload_output.clear_output()
        self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Uploading and extracting...</div>"
        with self.upload_output:
            success = self.manager.extract_dataset(self.upload_path.value, self.extract_dir.value)
            if success:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Upload and extraction complete.</div>"
            else:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload and extraction failed. Check logs.</div>"

    def run_tagging(self, b):
        self.tagging_output.clear_output()
        self.tagging_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Starting {self.tagging_method.value} tagging...</div>"
        with self.tagging_output:
            if not self.tagging_dataset_dir.value:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a dataset directory.</div>"
                print("❌ Please specify a dataset directory.")
                return
                
            print(f"🏷️ Starting {self.tagging_method.value} tagging with {self.tagger_model.value.split('/')[-1]}...")
            
            # Enhanced tagging with more options
            success = self.manager.tag_images(
                self.tagging_dataset_dir.value,
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

    def run_add_trigger(self, b):
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Adding trigger word...</div>"
        with self.caption_output:
            if not self.caption_dataset_dir.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a dataset directory.</div>"
                print("❌ Please specify a dataset directory.")
                return
                
            if not self.trigger_word.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a trigger word.</div>"
                print("❌ Please specify a trigger word.")
                return
                
            print(f"➕ Adding trigger word '{self.trigger_word.value}' to captions...")
            success = self.manager.add_trigger_word(self.caption_dataset_dir.value, self.trigger_word.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Trigger word added.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to add trigger word. Check logs.</div>"
    
    def run_remove_tags(self, b):
        """Remove specified tags from all caption files"""
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Removing tags...</div>"
        with self.caption_output:
            if not self.caption_dataset_dir.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a dataset directory.</div>"
                print("❌ Please specify a dataset directory.")
                return
                
            if not self.remove_tags.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to remove.</div>"
                print("❌ Please specify tags to remove.")
                return
                
            print(f"➖ Removing tags '{self.remove_tags.value}' from captions...")
            success = self.manager.remove_tags(self.caption_dataset_dir.value, self.remove_tags.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tags removed.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to remove tags. Check logs.</div>"

    def run_create_project(self, b):
        """🚀 One-stop project creation with calculator integration"""
        self.project_output.clear_output()
        self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Creating project...</div>"
        with self.project_output:
            project_name = self.project_name.value.strip()
            dataset_url = self.project_dataset_url.value.strip()
            
            if not project_name:
                self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a project name.</div>"
                print("❌ Please enter a project name.")
                return
                
            if not dataset_url:
                self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a dataset URL.</div>"
                print("❌ Please enter a dataset URL.")
                return
            
            # Sanitize project name
            import re
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
            if clean_name != project_name:
                print(f"📝 Cleaned project name: '{project_name}' → '{clean_name}'")
                project_name = clean_name
            
            project_dir = f"datasets/{project_name}"
            
            print(f"🚀 Creating project: {project_name}")
            print(f"📁 Project directory: {project_dir}")
            
            # Step 1: Create project directory
            import os
            os.makedirs(project_dir, exist_ok=True)
            print(f"✅ Created project directory: {project_dir}")
            
            # Step 2: Download and extract dataset
            print(f"📥 Downloading dataset from: {dataset_url}")
            success = self.manager.extract_dataset(dataset_url, project_dir)
            
            if success:
                # Step 3: Run personal calculator
                print(f"🧮 Running personal LoRA calculator...")
                try:
                    from personal_lora_calculator import count_images_in_directory
                    image_count = count_images_in_directory(project_dir)
                    
                    if image_count > 0:
                        self.project_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Project created successfully! Found {image_count} images.</div>"
                        print(f"📊 Found {image_count} images!")
                        print(f"")
                        print(f"🎯 TRAINING RECOMMENDATIONS:")
                        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        print(f"📁 Dataset Directory: {project_dir}")
                        print(f"📸 Image Count: {image_count}")
                        print(f"")
                        print(f"💡 Copy this path to your training widget:")
                        print(f"   {project_dir}")
                        print(f"")
                        print(f"🧮 Run the personal calculator notebook for detailed training parameters!")
                        
                    else:
                        self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> Project created, but no images found.</div>"
                        print("⚠️ No images found in extracted dataset. Check the extraction.")
                        
                except Exception as e:
                    self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> Project created, but calculator failed.</div>"
                    print(f"⚠️ Calculator error: {e}")
                    print(f"📊 Project created successfully, but calculator failed.")
                    print(f"💡 Use dataset directory: {project_dir}")
            else:
                self.project_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Dataset download/extraction failed.</div>"
                print(f"❌ Dataset download/extraction failed.")

    def display(self):
        display(self.widget_box)
