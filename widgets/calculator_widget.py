# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/calculator_widget.py
import glob
import os
from pathlib import Path

import ipywidgets as widgets
from IPython.display import Image, display


class CalculatorWidget:
    def __init__(self):
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the LoRA Step Calculator."""
        header_icon = "🧮"
        header_main = widgets.HTML(f"<h2>{header_icon} LoRA Step Calculator - Kohya Compatible</h2>")

        # --- Input Section ---
        input_desc = widgets.HTML("""<h3>▶️ Training Parameters</h3>
        <p>Select your dataset folder to automatically detect repeat counts (Kohya format).</p>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🎯 How it works:</strong><br>
        • Scans your dataset folder for Kohya-format subdirectories<br>
        • Auto-detects repeat counts from folder names (e.g., "10_character_name" → 10 repeats)<br>
        • Calculates accurate training steps: <strong>(Images × Repeats × Epochs) ÷ Batch Size</strong>
        </div>""")

        self.dataset_path = widgets.Text(
            description="Dataset Path:",
            placeholder="datasets/10_character_name or browse...",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.browse_button = widgets.Button(description="📁 Browse", button_style='info', layout=widgets.Layout(width='150px'))

        self.epochs = widgets.IntText(
            description="Epochs:",
            value=10,
            style={'description_width': 'initial'}
        )

        self.batch_size = widgets.IntText(
            description="Batch Size:",
            value=1,
            style={'description_width': 'initial'}
        )

        self.calculate_button = widgets.Button(description="Calculate Steps", button_style='success')

        dataset_input_box = widgets.HBox([self.dataset_path, self.browse_button])

        input_box = widgets.VBox([
            input_desc,
            dataset_input_box,
            self.epochs,
            self.batch_size,
            self.calculate_button
        ])

        # --- Output Section ---
        self.output_area = widgets.Output(layout=widgets.Layout(height='auto', overflow='auto', border='1px solid #ddd'))

        # --- Main Widget Box ---
        self.widget_box = widgets.VBox([header_main, input_box, self.output_area])

        # --- Button Events ---
        self.calculate_button.on_click(self.run_calculation)
        self.browse_button.on_click(self.browse_datasets)

    def extract_kohya_params(self, folder_name: str):
        """Extract repeat count from Kohya-format folder name using the same logic as config_util.py"""
        tokens = folder_name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError:
            # No repeat count in folder name - default to 1 like Kohya does
            return 1, folder_name
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def count_images_in_directory(self, directory):
        """Count image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        image_count = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_count += 1
        return image_count

    def browse_datasets(self, b):
        """Auto-populate with detected dataset directories"""
        with self.output_area:
            self.output_area.clear_output()

            # Look for dataset directories
            datasets_pattern = "datasets/*"
            existing_dirs = [d for d in glob.glob(datasets_pattern) if os.path.isdir(d)]

            if existing_dirs:
                # Sort by modification time (most recent first)
                existing_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                print("📁 Available datasets:")
                print("=" * 40)

                for i, dataset_dir in enumerate(existing_dirs[:5]):  # Show top 5
                    folder_name = os.path.basename(dataset_dir)
                    repeats, caption = self.extract_kohya_params(folder_name)
                    image_count = self.count_images_in_directory(dataset_dir)

                    print(f"{i+1}. {dataset_dir}")
                    print(f"   📊 {image_count} images, {repeats} repeats")
                    if caption:
                        print(f"   🏷️ Caption: {caption}")
                    print()

                # Auto-populate with most recent
                most_recent = existing_dirs[0]
                self.dataset_path.value = most_recent
                print(f"📌 Auto-selected: {most_recent}")

            else:
                print("📂 No datasets found in 'datasets/' directory")
                print("💡 Create a dataset using the Dataset Maker first!")

    def run_calculation(self, b):
        self.output_area.clear_output()

        with self.output_area:
            dataset_path = self.dataset_path.value.strip()
            epochs = self.epochs.value
            batch_size = self.batch_size.value

            if not dataset_path:
                print("❌ Please specify a dataset path.")
                print("💡 Click 'Browse' to see available datasets")
                return

            if batch_size <= 0:
                print("❌ Batch size must be greater than zero.")
                return

            if not os.path.exists(dataset_path):
                print(f"❌ Dataset path does not exist: {dataset_path}")
                return

            # Extract Kohya parameters from folder name
            folder_name = os.path.basename(dataset_path)
            repeats, caption = self.extract_kohya_params(folder_name)

            # Count images in dataset
            images = self.count_images_in_directory(dataset_path)

            if images == 0:
                print(f"❌ No images found in: {dataset_path}")
                return

            # Calculate total steps using Kohya's exact logic
            total_steps = (images * repeats * epochs) // batch_size

            # --- Display Results ---
            print("🧮 LoRA Step Calculator - Kohya Compatible")
            print("=" * 50)
            print(f"📁 Dataset Path:      {dataset_path}")
            if caption:
                print(f"🏷️ Detected Caption:  {caption}")
            print(f"📸 Images Found:      {images}")
            print(f"🔄 Repeat Count:      {repeats} (auto-detected from folder name)")
            print(f"📅 Epochs:            {epochs}")
            print(f"📦 Batch Size:        {batch_size}")
            print("=" * 50)
            print(f"⚡ Total Steps:       {total_steps}")

            # Training time estimation
            if total_steps > 0:
                print("\n⏱️ Time Estimates (approximate):")
                print(f"   • GPU rental:     {total_steps * 2 / 60:.1f}-{total_steps * 4 / 60:.1f} minutes")
                print(f"   • Home GPU:       {total_steps * 3 / 60:.1f}-{total_steps * 6 / 60:.1f} minutes")

            # Recommendations
            print("\n🎯 Training Analysis:")
            if total_steps < 500:
                print("⚠️ Low step count - may underfit. Consider more epochs or repeats.")
            elif total_steps > 5000:
                print("⚠️ High step count - may overfit. Consider fewer epochs.")
            else:
                print("✅ Good step count for most LoRA training scenarios.")

            print(f"\n💡 Formula: ({images} images × {repeats} repeats × {epochs} epochs) ÷ {batch_size} batch = {total_steps} steps")

            # --- Display Doro Image ---
            image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'doro.png')
            if os.path.exists(image_path):
                display(Image(filename=image_path))
            else:
                print(f"\n(Doro image not found at '{image_path}')")

    def display(self):
        display(self.widget_box)
