import torch
from diffusers import StableDiffusionPipeline


class LoRAInferenceManager:
    def __init__(self):
        self.pipeline = None
        self.current_lora = None
        self.base_model = None

    def load_base_model(self, model_path):
        """Load base SD model"""
        print(f"🎨 Loading base model: {model_path}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.base_model = model_path
        print("✅ Base model loaded.")

    def load_lora(self, lora_path, strength=1.0):
        """Load and apply LoRA"""
        if self.pipeline:
            print(f"✨ Loading LoRA: {lora_path} with strength {strength}")
            self.pipeline.load_lora_weights(lora_path)
            self.current_lora = lora_path
            print("✅ LoRA loaded.")
        else:
            print("❌ Base model not loaded. Cannot load LoRA.")

    def generate_preview(self, prompt, negative_prompt="", steps=20, cfg=7.0):
        """Generate preview image"""
        if not self.pipeline:
            print("❌ Pipeline not initialized. Load base model first.")
            return None

        print(f"Generating preview for: {prompt}")
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg
        ).images[0]

        print("✅ Preview image generated.")
        return image
