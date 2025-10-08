# widgets/finetuning_widget.py
import ipywidgets as widgets
from IPython.display import display

class FinetuningWidget:
    def __init__(self):
        # We will add a manager class here later
        # self.manager = FinetuningManager()
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Finetuning Widget."""
        header = widgets.HTML("<h2>🧬 Finetuning Controls</h2>")
        
        # Note about supported models, as requested
        supported_models_note = widgets.HTML(
            "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'>"
            "<strong>Note:</strong> This finetuning module is currently focused on <strong>SD1.5</strong> and <strong>SDXL</strong> base models. "
            "Support for other architectures is not yet implemented.</div>"
        )

        # Placeholder for finetuning-specific widgets
        placeholder_text = widgets.Label("Finetuning settings (like learning rate, optimizer, etc.) will go here.")

        self.widget_box = widgets.VBox([
            header,
            supported_models_note,
            placeholder_text
        ])

    def display(self):
        """Displays the widget."""
        display(self.widget_box)
