#!/usr/bin/env python3
"""
Image Curation Widget - Pre-tagging image curation interface using FiftyOne

Implements the corrected workflow: Upload → FiftyOne Curation → WD14 Tagging → Training
This is step 1.5 in the training pipeline - BEFORE tagging, not after.
"""

import os

import ipywidgets as widgets
from IPython.display import clear_output, display


class ImageCurationWidget:
    def __init__(self, shared_managers):
        self.shared_managers = shared_managers
        self.curation_manager = None
        self.current_dataset = None
        self.current_session = None
        self.curation_sidecar = None

        # Initialize sidecar for FiftyOne
        try:
            from sidecar import Sidecar
            self.curation_sidecar = Sidecar(
                title='🔍 FiftyOne Image Curation',
                anchor='split-right'
            )
            print("📺 Sidecar initialized for FiftyOne curation")
        except ImportError:
            print("⚠️ Sidecar not available - FiftyOne will open in browser tab")
            print("   Install with: pip install sidecar")
            self.curation_sidecar = None

        # UI state
        self.project_path = ""
        self.staging_dir = ""

        # Create the widget interface
        self._create_interface()

    def _create_interface(self):
        """Create the complete curation widget interface"""

        # Header
        self.header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; text-align: center;'>
                🖼️ Image Curation Manager
            </h2>
            <p style='color: white; margin: 5px 0 0 0; text-align: center; opacity: 0.9;'>
                Pre-tagging image curation with duplicate detection • Step 1.5 of LoRA Training Pipeline
            </p>
        </div>
        """)

        # Step 1: Upload Images
        self._create_upload_section()

        # Step 2: Curation Interface
        self._create_curation_section()

        # Step 3: Export Results
        self._create_export_section()

        # Status and Progress
        self._create_status_section()

        # Main container
        self.main_widget = widgets.VBox([
            self.header,
            self.upload_section,
            self.curation_section,
            self.export_section,
            self.status_section
        ])

    def _create_upload_section(self):
        """Create image upload interface"""

        # Project path input
        self.project_path_input = widgets.Text(
            placeholder="Enter project path (e.g., /path/to/my_lora_project)",
            description="Project Path:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )

        # Source paths for images
        self.source_paths_input = widgets.Textarea(
            placeholder="Enter image sources (one per line):\n/path/to/images/folder\n/path/to/image.jpg\n/path/to/archive.zip",
            description="Image Sources:",
            rows=4,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )

        # Upload button
        self.upload_button = widgets.Button(
            description="📤 Upload Images to Staging",
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        self.upload_button.on_click(self._on_upload_click)

        # Upload output
        self.upload_output = widgets.Output()

        self.upload_section = widgets.VBox([
            widgets.HTML("<h3>📁 Step 1: Upload Images</h3>"),
            widgets.HTML("<p>Upload raw images to staging directory for curation.</p>"),
            self.project_path_input,
            self.source_paths_input,
            self.upload_button,
            self.upload_output
        ])

    def _create_curation_section(self):
        """Create curation interface"""

        # Create dataset button
        self.create_dataset_button = widgets.Button(
            description="🔍 Create Curation Dataset",
            button_style='info',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        self.create_dataset_button.on_click(self._on_create_dataset_click)

        # Launch FiftyOne button
        self.launch_fiftyone_button = widgets.Button(
            description="🖥️ Launch FiftyOne App",
            button_style='success',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        self.launch_fiftyone_button.on_click(self._on_launch_fiftyone_click)

        # Refresh stats button
        self.refresh_stats_button = widgets.Button(
            description="📊 Refresh Stats",
            button_style='',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        self.refresh_stats_button.on_click(self._on_refresh_stats_click)

        # Curation output
        self.curation_output = widgets.Output()

        self.curation_section = widgets.VBox([
            widgets.HTML("<h3>🔍 Step 2: Curate Images</h3>"),
            widgets.HTML("""
            <p>Visual inspection and duplicate removal using FiftyOne:</p>
            <ul>
                <li>🧠 CLIP-based duplicate detection</li>
                <li>👁️ Visual quality inspection</li>
                <li>❌ Remove unwanted images</li>
                <li>✅ Mark images for training</li>
            </ul>
            """),
            widgets.HBox([
                self.create_dataset_button,
                self.launch_fiftyone_button,
                self.refresh_stats_button
            ]),
            self.curation_output
        ])

    def _create_export_section(self):
        """Create export interface"""

        # Export directory input
        self.export_dir_input = widgets.Text(
            placeholder="Leave empty to use project_path/dataset",
            description="Export Directory:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )

        # Export button
        self.export_button = widgets.Button(
            description="📦 Export Curated Dataset",
            button_style='warning',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        self.export_button.on_click(self._on_export_click)

        # Export output
        self.export_output = widgets.Output()

        self.export_section = widgets.VBox([
            widgets.HTML("<h3>📦 Step 3: Export Curated Dataset</h3>"),
            widgets.HTML("<p>Export curated images for WD14 tagging and training.</p>"),
            self.export_dir_input,
            self.export_button,
            self.export_output
        ])

    def _create_status_section(self):
        """Create status and progress display"""

        self.status_display = widgets.HTML("""
        <div style='background: #007acc; color: white; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h4>📋 Workflow Status</h4>
            <p>Ready to begin image curation workflow.</p>
        </div>
        """)

        self.status_section = widgets.VBox([self.status_display])

    def _on_upload_click(self, button):
        """Handle image upload"""

        with self.upload_output:
            clear_output()

            if not self.project_path_input.value.strip():
                print("❌ Please enter a project path")
                return

            if not self.source_paths_input.value.strip():
                print("❌ Please enter at least one image source")
                return

            try:
                # Initialize curation manager
                if not self.curation_manager:
                    from core.image_curation_manager import ImageCurationManager
                    self.curation_manager = ImageCurationManager()

                # Set project path
                self.project_path = self.project_path_input.value.strip()

                # Create staging directory
                project_name = os.path.basename(self.project_path.rstrip('/'))
                self.staging_dir = self.curation_manager.create_staging_directory(
                    self.project_path, project_name
                )

                # Parse source paths
                source_paths = [
                    path.strip() for path in self.source_paths_input.value.strip().split('\n')
                    if path.strip()
                ]

                # Upload images
                stats = self.curation_manager.upload_images_to_staging(source_paths, self.staging_dir)

                if stats['uploaded'] > 0:
                    print(f"✅ Successfully uploaded {stats['uploaded']} images")
                    self.create_dataset_button.disabled = False
                    self._update_status("Images uploaded to staging. Ready to create curation dataset.")
                else:
                    print("❌ No images uploaded. Check your source paths.")

            except Exception as e:
                print(f"❌ Upload failed: {e}")

    def _on_create_dataset_click(self, button):
        """Handle dataset creation"""

        with self.curation_output:
            clear_output()

            if not self.staging_dir:
                print("❌ No staging directory. Upload images first.")
                return

            try:
                print("🔍 Creating FiftyOne curation dataset...")

                # Create dataset with duplicate detection
                dataset_name = f"curation_{os.path.basename(self.project_path)}"
                self.current_dataset = self.curation_manager.create_curation_dataset(
                    self.staging_dir, dataset_name
                )

                if self.current_dataset:
                    print("✅ Curation dataset created successfully")

                    # Show initial stats
                    self.curation_manager.print_curation_stats(self.current_dataset)

                    self.launch_fiftyone_button.disabled = False
                    self.refresh_stats_button.disabled = False
                    self._update_status("Curation dataset ready. Launch FiftyOne to begin visual curation.")
                else:
                    print("❌ Failed to create curation dataset")

            except Exception as e:
                print(f"❌ Dataset creation failed: {e}")

    def _on_launch_fiftyone_click(self, button):
        """Handle FiftyOne app launch"""

        with self.curation_output:
            clear_output()

            if not self.current_dataset:
                print("❌ No dataset available. Create dataset first.")
                return

            try:
                print("🖥️ Launching FiftyOne curation app...")

                # Get server configuration
                server_config = self.curation_manager.get_server_config()

                # Launch FiftyOne app
                self.current_session = self.curation_manager.launch_curation_app(
                    self.current_dataset, server_config
                )

                if self.current_session:
                    # Integrate with sidecar if available
                    if self.curation_sidecar:
                        try:
                            print("🔗 Integrating FiftyOne with Jupyter sidecar...")

                            # Display FiftyOne in the sidecar
                            with self.curation_sidecar:
                                # Get the session's app URL
                                session_url = self.current_session.url

                                # Create an iframe to display FiftyOne in sidecar
                                iframe_html = f"""
                                <iframe src="{session_url}" 
                                        width="100%" 
                                        height="800px" 
                                        frameborder="0"
                                        title="FiftyOne Image Curation">
                                </iframe>
                                """

                                from IPython.display import HTML
                                display(HTML(iframe_html))

                            print("✅ FiftyOne launched in sidecar panel!")
                            print("📺 Use the curation panel on the right to inspect and curate images")

                        except Exception as e:
                            print(f"⚠️ Sidecar integration failed: {e}")
                            print(f"🌐 FiftyOne available at: {self.current_session.url}")
                    else:
                        print(f"🌐 FiftyOne available at: {self.current_session.url}")
                        print("💡 Install sidecar for integrated panel: pip install sidecar")

                    self.export_button.disabled = False
                    self._update_status("FiftyOne app launched. Perform visual curation, then export results.")
                else:
                    print("❌ Failed to launch FiftyOne app")

            except Exception as e:
                print(f"❌ FiftyOne launch failed: {e}")

    def _on_refresh_stats_click(self, button):
        """Handle stats refresh"""

        with self.curation_output:
            clear_output()

            if not self.current_dataset:
                print("❌ No dataset available")
                return

            try:
                print("📊 Refreshing curation statistics...")
                self.curation_manager.print_curation_stats(self.current_dataset)

            except Exception as e:
                print(f"❌ Stats refresh failed: {e}")

    def _on_export_click(self, button):
        """Handle dataset export"""

        with self.export_output:
            clear_output()

            if not self.current_dataset:
                print("❌ No dataset available. Create and curate dataset first.")
                return

            try:
                # Determine export directory
                if self.export_dir_input.value.strip():
                    export_dir = self.export_dir_input.value.strip()
                else:
                    export_dir = os.path.join(self.project_path, "dataset")

                print("📦 Exporting curated dataset...")

                # Export curated images
                success, stats = self.curation_manager.export_curated_dataset(
                    self.current_dataset, export_dir
                )

                if success:
                    self._update_status(f"Curation complete! {stats['exported']} images exported to {export_dir}")

                    # Show next steps
                    print("\\n🎯 NEXT WORKFLOW STEP:")
                    print(f"   Use WD14 Tagger widget on: {export_dir}")

                else:
                    print("❌ Export failed")

            except Exception as e:
                print(f"❌ Export failed: {e}")

    def _update_status(self, message: str):
        """Update the status display"""

        self.status_display.value = f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h4>📋 Workflow Status</h4>
            <p>{message}</p>
        </div>
        """

    def display(self):
        """Display the widget"""
        display(self.main_widget)

    def get_current_stats(self):
        """Get current curation statistics"""
        if self.current_dataset and self.curation_manager:
            return self.curation_manager.get_curation_progress(self.current_dataset)
        return None


def create_image_curation_widget(shared_managers):
    """Factory function to create the image curation widget"""
    return ImageCurationWidget(shared_managers)


def test_widget():
    """Test the widget in standalone mode"""

    print("🧪 Testing Image Curation Widget")

    # Mock shared managers for testing
    class MockSharedManagers:
        pass

    shared_managers = MockSharedManagers()

    # Create and display widget
    widget = ImageCurationWidget(shared_managers)
    widget.display()

    return widget


if __name__ == "__main__":
    test_widget()
