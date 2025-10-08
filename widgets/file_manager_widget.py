# widgets/file_manager_widget.py
import os
import sys

import ipywidgets as widgets
from IPython.display import clear_output


class FileManagerWidget:
    """Widget interface for IMjoy Elfinder file manager"""

    def __init__(self):
        self.file_manager = None
        self.output_area = widgets.Output()

        # Initialize file manager
        try:
            sys.path.append(os.path.join(os.getcwd(), 'core'))
            from file_manager import get_file_manager
            self.file_manager = get_file_manager()
        except ImportError:
            self.file_manager = None

    def create_file_manager_widget(self):
        """Create the file manager widget interface"""

        # Header
        header = widgets.HTML("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h2 style="margin: 0; display: flex; align-items: center;">
                📁 File Manager
                <span style="font-size: 14px; margin-left: 10px; opacity: 0.8;">IMjoy Elfinder</span>
            </h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Browse and manage project files with a modern web interface</p>
        </div>
        """)

        # Quick start buttons
        quick_start_header = widgets.HTML("<h3>🚀 Quick Start</h3>")

        project_btn = widgets.Button(
            description="📂 Browse Project",
            button_style="info",
            tooltip="Browse the entire project directory",
            layout=widgets.Layout(width="150px", margin="2px")
        )

        dataset_btn = widgets.Button(
            description="🖼️ Browse Datasets",
            button_style="success",
            tooltip="Browse dataset directories",
            layout=widgets.Layout(width="150px", margin="2px")
        )

        output_btn = widgets.Button(
            description="📦 Browse Output",
            button_style="warning",
            tooltip="Browse training output directory",
            layout=widgets.Layout(width="150px", margin="2px")
        )

        quick_buttons = widgets.HBox([project_btn, dataset_btn, output_btn])

        # Advanced options
        advanced_header = widgets.HTML("<h3>⚙️ Advanced Options</h3>")

        # Directory selection
        directory_input = widgets.Text(
            value=os.getcwd(),
            placeholder="Enter directory path",
            description="Directory:",
            layout=widgets.Layout(width="400px")
        )

        # Port selection
        port_input = widgets.IntText(
            value=8765,
            description="Port:",
            min=8000,
            max=9999,
            layout=widgets.Layout(width="150px")
        )

        # Display mode
        display_mode = widgets.RadioButtons(
            options=[
                ("Embedded (in notebook)", False),
                ("New tab (recommended)", True)
            ],
            value=True,
            description="Display:"
        )

        # Start button
        start_btn = widgets.Button(
            description="🚀 Start File Manager",
            button_style="primary",
            layout=widgets.Layout(width="200px")
        )

        # Control buttons
        control_header = widgets.HTML("<h3>🎛️ Control</h3>")

        list_btn = widgets.Button(
            description="📋 List Active",
            button_style="",
            layout=widgets.Layout(width="150px", margin="2px")
        )

        stop_all_btn = widgets.Button(
            description="🛑 Stop All",
            button_style="danger",
            layout=widgets.Layout(width="150px", margin="2px")
        )

        control_buttons = widgets.HBox([list_btn, stop_all_btn])

        # Event handlers
        def on_project_click(b):
            self._start_with_output("🚀 Starting project browser...")
            if self.file_manager:
                self.file_manager.start_project_browser(open_in_new_tab=True)
            else:
                print("❌ File manager not available")

        def on_dataset_click(b):
            self._start_with_output("🚀 Starting dataset browser...")
            if self.file_manager:
                # Try common dataset locations
                for dataset_path in ['dataset', 'datasets', '.']:
                    full_path = os.path.join(os.getcwd(), dataset_path)
                    if os.path.exists(full_path):
                        self.file_manager.start_file_manager(full_path, open_in_new_tab=True)
                        return
                print("📁 No dataset directory found, using project root")
                self.file_manager.start_project_browser(open_in_new_tab=True)
            else:
                print("❌ File manager not available")

        def on_output_click(b):
            self._start_with_output("🚀 Starting output browser...")
            if self.file_manager:
                self.file_manager.start_output_browser(open_in_new_tab=True)
            else:
                print("❌ File manager not available")

        def on_start_click(b):
            self._start_with_output(f"🚀 Starting file manager for {directory_input.value}...")
            if self.file_manager:
                self.file_manager.start_file_manager(
                    root_dir=directory_input.value,
                    open_in_new_tab=display_mode.value,
                    port=port_input.value
                )
            else:
                print("❌ File manager not available")

        def on_list_click(b):
            self._start_with_output("📋 Listing active file managers...")
            if self.file_manager:
                self.file_manager.list_active_managers()
            else:
                print("❌ File manager not available")

        def on_stop_all_click(b):
            self._start_with_output("🛑 Stopping all file managers...")
            if self.file_manager:
                self.file_manager.stop_all_managers()
            else:
                print("❌ File manager not available")

        # Bind events
        project_btn.on_click(on_project_click)
        dataset_btn.on_click(on_dataset_click)
        output_btn.on_click(on_output_click)
        start_btn.on_click(on_start_click)
        list_btn.on_click(on_list_click)
        stop_all_btn.on_click(on_stop_all_click)

        # Layout
        advanced_section = widgets.VBox([
            advanced_header,
            widgets.HBox([directory_input, port_input]),
            display_mode,
            start_btn
        ])

        control_section = widgets.VBox([
            control_header,
            control_buttons
        ])

        # Installation check
        installation_status = self._check_installation()

        main_widget = widgets.VBox([
            header,
            installation_status,
            quick_start_header,
            quick_buttons,
            advanced_section,
            control_section,
            self.output_area
        ])

        return main_widget

    def _check_installation(self):
        """Check if IMjoy Elfinder is installed"""
        try:
            import imjoy_elfinder
            return widgets.HTML("""
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; 
                        padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                ✅ IMjoy Elfinder is installed and ready to use
            </div>
            """)
        except ImportError:
            return widgets.HTML("""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; 
                        padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                ❌ IMjoy Elfinder not installed<br>
                <strong>Install with:</strong> <code>pip install imjoy-elfinder</code>
            </div>
            """)

    def _start_with_output(self, message):
        """Clear output and show starting message"""
        with self.output_area:
            clear_output(wait=True)
            print(message)

def create_file_manager_widget():
    """Factory function to create file manager widget"""
    widget = FileManagerWidget()
    return widget.create_file_manager_widget()
