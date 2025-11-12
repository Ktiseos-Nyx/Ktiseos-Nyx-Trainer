"""
Gradio UI Mockup - Custom Themed LoRA Trainer
This is a DEMO to show what's possible with Gradio theming.
No actual training happens - just UI exploration!
"""

import gradio as gr
import time

# ============================================================================
# CUSTOM THEME
# ============================================================================
custom_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "monospace"],
).set(
    # Button styling
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_shadow="*shadow_drop_lg",
    button_large_padding="12px 24px",
    button_large_text_size="16px",

    # Input styling
    input_background_fill="*neutral_50",
    input_border_color="*neutral_300",
    input_shadow="*shadow_inset",

    # Panel styling
    panel_background_fill="white",
    panel_border_color="*neutral_200",
    panel_border_width="1px",

    # Spacing
    spacing_lg="16px",
    spacing_xl="24px",

    # Border radius
    radius_lg="12px",
    radius_xl="16px",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
custom_css = """
/* Container styling */
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

/* Header styling */
.app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
}

.app-header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
}

.app-header p {
    margin: 8px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
}

/* Tab styling */
.tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
}

/* Card-like sections */
.section-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Status display */
.status-box {
    background: #f0fdf4;
    border: 2px solid #86efac;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
}

.status-box.error {
    background: #fef2f2;
    border-color: #fca5a5;
}

/* Progress styling */
.progress-container {
    margin: 24px 0;
}

/* Better accordion styling */
.accordion {
    border-radius: 8px !important;
}

/* Form improvements */
label {
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 8px !important;
}

/* Output text styling */
.output-text {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
"""

# ============================================================================
# MOCK FUNCTIONS
# ============================================================================

def mock_train(
    model_name,
    dataset_path,
    output_name,
    rank,
    alpha,
    learning_rate,
    epochs,
    batch_size,
    optimizer,
    progress=gr.Progress()
):
    """Mock training function - doesn't actually train!"""

    yield "🚀 Initializing training...\n"
    time.sleep(1)

    yield f"📦 Model: {model_name}\n"
    yield f"📁 Dataset: {dataset_path or 'Not specified'}\n"
    yield f"💾 Output: {output_name}\n\n"

    yield "⚙️ Training Configuration:\n"
    yield f"  • Rank: {rank}\n"
    yield f"  • Alpha: {alpha}\n"
    yield f"  • Learning Rate: {learning_rate}\n"
    yield f"  • Epochs: {epochs}\n"
    yield f"  • Batch Size: {batch_size}\n"
    yield f"  • Optimizer: {optimizer}\n\n"

    # Simulate training epochs
    for epoch in range(1, epochs + 1):
        progress(epoch / epochs, desc=f"Epoch {epoch}/{epochs}")
        yield f"📈 Epoch {epoch}/{epochs} - Loss: {1.5 - (epoch * 0.1):.4f}\n"
        time.sleep(0.5)

    yield "\n✅ Training completed successfully!\n"
    yield f"💾 Model saved to: output/{output_name}/\n"

def mock_status():
    """Mock status check"""
    return """
🟢 **System Status: Ready**

**GPU:** Not connected (mockup mode)
**Memory:** Available
**Training:** No active jobs

This is a UI mockup - no actual training will occur.
"""

def mock_validate_dataset(file):
    """Mock dataset validation"""
    if file is None:
        return "⚠️ No dataset uploaded"
    return f"✅ Dataset valid: {file.name}\n📊 Images: ~150 (mock)\n🏷️ Captions: Found"

# ============================================================================
# BUILD THE UI
# ============================================================================

with gr.Blocks(theme=custom_theme, css=custom_css, title="LoRA Trainer") as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
            <h1>🎨 Ktiseos Nyx LoRA Trainer</h1>
            <p>Custom Themed Gradio Interface - Mockup Version</p>
        </div>
    """)

    # Main Tabs
    with gr.Tabs():

        # ====================================================================
        # TRAINING TAB
        # ====================================================================
        with gr.Tab("🚀 Training", id="train"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Model Configuration")

                    model_name = gr.Dropdown(
                        choices=[
                            "stabilityai/stable-diffusion-xl-base-1.0",
                            "runwayml/stable-diffusion-v1-5",
                            "stabilityai/stable-diffusion-2-1",
                        ],
                        value="stabilityai/stable-diffusion-xl-base-1.0",
                        label="Base Model",
                        info="Select your base model"
                    )

                    output_name = gr.Textbox(
                        value="my-lora",
                        label="Output Name",
                        placeholder="my-awesome-lora"
                    )

                    gr.Markdown("### 📁 Dataset")

                    dataset_path = gr.Textbox(
                        label="Dataset Path",
                        placeholder="/workspace/datasets/my-dataset",
                        info="Path to your prepared dataset"
                    )

                    dataset_file = gr.File(
                        label="Or Upload Dataset",
                        file_types=[".zip"],
                        type="filepath"
                    )

                    dataset_status = gr.Textbox(
                        label="Dataset Status",
                        interactive=False,
                        lines=3
                    )

                    validate_btn = gr.Button("🔍 Validate Dataset", variant="secondary")

                    gr.Markdown("### ⚙️ LoRA Parameters")

                    with gr.Accordion("Basic Settings", open=True):
                        rank = gr.Slider(4, 128, value=32, step=4, label="Rank (Dimension)")
                        alpha = gr.Slider(1, 128, value=32, step=1, label="Alpha")
                        learning_rate = gr.Number(value=1e-4, label="Learning Rate")

                    with gr.Accordion("Training Settings", open=True):
                        epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                        batch_size = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        optimizer = gr.Dropdown(
                            choices=["AdamW", "AdamW8bit", "Lion", "Prodigy"],
                            value="AdamW8bit",
                            label="Optimizer"
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        gr.Markdown("*Additional parameters would go here*")
                        mixed_precision = gr.Radio(
                            choices=["no", "fp16", "bf16"],
                            value="fp16",
                            label="Mixed Precision"
                        )
                        gradient_accumulation = gr.Slider(1, 8, value=1, step=1, label="Gradient Accumulation Steps")

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Training Output")

                    train_output = gr.Textbox(
                        label="Console Output",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        elem_classes=["output-text"]
                    )

                    with gr.Row():
                        train_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### 💡 Tips")
                    gr.Markdown("""
                    - **Rank**: Higher = more capacity, slower training
                    - **Alpha**: Usually same as rank for balanced learning
                    - **Learning Rate**: Start with 1e-4, adjust if needed
                    - **Epochs**: 10-20 for most cases, monitor for overfitting
                    """)

            # Wire up mock functions
            validate_btn.click(
                fn=mock_validate_dataset,
                inputs=[dataset_file],
                outputs=[dataset_status]
            )

            train_btn.click(
                fn=mock_train,
                inputs=[
                    model_name, dataset_path, output_name,
                    rank, alpha, learning_rate,
                    epochs, batch_size, optimizer
                ],
                outputs=[train_output]
            )

        # ====================================================================
        # STATUS TAB
        # ====================================================================
        with gr.Tab("📊 Status", id="status"):
            gr.Markdown("## System Status")

            status_output = gr.Markdown(value=mock_status())

            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            refresh_btn.click(fn=mock_status, outputs=[status_output])

            gr.Markdown("---")
            gr.Markdown("## Recent Jobs")
            gr.Markdown("*No jobs yet - start training to see history*")

        # ====================================================================
        # SETTINGS TAB
        # ====================================================================
        with gr.Tab("⚙️ Settings", id="settings"):
            gr.Markdown("## Application Settings")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎨 Appearance")
                    theme_choice = gr.Radio(
                        choices=["Light", "Dark", "Auto"],
                        value="Light",
                        label="Theme"
                    )

                with gr.Column():
                    gr.Markdown("### 🔧 Advanced")
                    auto_save = gr.Checkbox(label="Auto-save configuration", value=True)
                    notifications = gr.Checkbox(label="Enable notifications", value=True)

            gr.Button("💾 Save Settings", variant="primary")

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
