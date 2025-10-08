import os

import ipywidgets as widgets
from IPython.display import Image as IPImage
from IPython.display import display


class InferenceWidget:
    def __init__(self):
        try:
            from sidecar import Sidecar
            self.inference_sidecar = Sidecar(title='🎨 LoRA Inference Preview', anchor='split-bottom')
        except ImportError:
            print("⚠️ Sidecar not available. Inference will display in main notebook.")
            self.inference_sidecar = None

        # Lazy import inference manager to avoid startup issues
        from shared_managers import get_inference_manager
        self.inference_manager = get_inference_manager()
        self.create_widgets()

    def create_widgets(self):
        # LoRA selection
        self.lora_dropdown = widgets.Dropdown(
            description="LoRA Model:",
            options=self._get_lora_options(),  # Populate from output directory
            layout=widgets.Layout(width='99%')
        )

        # Base Model Path
        self.base_model_path = widgets.Text(
            description="Base Model:",
            placeholder="Path to base SD model (e.g., /pretrained_model/sdxl.safetensors)",
            layout=widgets.Layout(width='99%')
        )
        self.load_base_model_button = widgets.Button(
            description="Load Base Model",
            button_style='info'
        )
        self.load_base_model_button.on_click(self._on_load_base_model)

        # Prompt inputs
        self.prompt_text = widgets.Textarea(
            description="Prompt:",
            placeholder="your_trigger_word, 1girl, standing...",
            layout=widgets.Layout(width='99%', height='100px')
        )

        self.negative_prompt_text = widgets.Textarea(
            description="Negative:",
            placeholder="worst quality, low quality...",
            layout=widgets.Layout(width='99%', height='60px')
        )

        # Generation controls
        self.lora_strength = widgets.FloatSlider(
            description="LoRA Strength:",
            min=0.0, max=2.0, step=0.1, value=1.0
        )

        self.cfg_scale = widgets.FloatSlider(
            description="CFG Scale:",
            min=1.0, max=20.0, step=0.5, value=7.0
        )

        self.steps_slider = widgets.IntSlider(
            description="Steps:",
            min=10, max=50, step=5, value=20
        )

        self.generate_button = widgets.Button(
            description="🎨 Generate Preview",
            button_style='primary'
        )
        self.generate_button.on_click(self._on_generate)

        # Results display
        self.result_output = widgets.Output()

        self.widget_box = widgets.VBox([
            self.base_model_path,
            self.load_base_model_button,
            self.lora_dropdown,
            self.lora_strength,
            self.prompt_text,
            self.negative_prompt_text,
            self.cfg_scale,
            self.steps_slider,
            self.generate_button,
            self.result_output
        ])

    def _get_lora_options(self):
        # Scan the output directory for .safetensors files
        lora_files = []
        output_dir = os.path.join(os.getcwd(), "output") # Assuming 'output' is where LoRAs are saved
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".safetensors"):
                        lora_files.append(os.path.join(root, file))
        return [("None", None)] + [(os.path.basename(f), f) for f in lora_files]

    def _on_load_base_model(self, b):
        with self.result_output:
            self.result_output.clear_output()
            model_path = self.base_model_path.value.strip()
            if not model_path or not os.path.exists(model_path):
                print("❌ Please provide a valid path to your base model.")
                return
            try:
                self.inference_manager.load_base_model(model_path)
                print("✅ Base model loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load base model: {e}")

    def _on_generate(self, b):
        with self.result_output:
            self.result_output.clear_output()
            if self.inference_manager.pipeline is None:
                print("❌ Please load a base model first.")
                return

            lora_path = self.lora_dropdown.value
            lora_strength = self.lora_strength.value
            prompt = self.prompt_text.value
            negative_prompt = self.negative_prompt_text.value
            steps = self.steps_slider.value
            cfg = self.cfg_scale.value

            try:
                if lora_path:
                    self.inference_manager.load_lora(lora_path, lora_strength)

                image = self.inference_manager.generate_preview(prompt, negative_prompt, steps, cfg)
                if image:
                    display(IPImage(image.tobytes(), format='png'))
                    print("✅ Preview generated!")
                else:
                    print("❌ Image generation failed.")
            except Exception as e:
                print(f"❌ Error during image generation: {e}")

    def display(self):
        if self.inference_sidecar:
            with self.inference_sidecar:
                display(self.widget_box)
        else:
            # Fallback to main notebook display
            display(self.widget_box)
