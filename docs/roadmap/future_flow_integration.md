# Plan for Integrating FLOW Training with Chroma

## 1. Goal
Integrate the "FLOW Matching Trainer" (`lodestone-rock/flow`) into the existing FastAPI backend and Next.js frontend, specifically enabling training of image generation models using "Chroma" base models. This integration aims to provide an alternative or supplementary training workflow to the existing Kohya SS methods.

## 2. Overview of FLOW Training
FLOW is a powerful training toolkit for image generation models using the Flow Matching technique. Key features include:
*   **Streaming Dataloader:** Works directly from S3 or local drives.
*   **Flexible Configuration:** JSON-based configuration files (`training_config.json`).
*   **Multi-GPU Training:** Automatic device detection.
*   **Inference During Training:** Configurable inference runs.
*   **Wandb and Hugging Face Integration.**
*   **Parameter-Efficient Training:** Supports layer rotation and offloading.
*   **Data Format:** Uses JSONL files for dataset metadata, specifying `filename` (S3 URL or local path), `caption_or_tags`, dimensions, etc.
*   **Chroma:** Refers to the base model checkpoint (e.g., `chroma-8.9b.safetensors`) from which FLOW training resumes.

## 3. Integration Strategy
The integration will involve extending the existing backend and frontend to support FLOW's distinct configuration, data formats, and training lifecycle, while leveraging existing infrastructure where possible.

## 4. Key Integration Areas

### 4.1. Dependency Management
*   **Action:** Add `flow` library to `requirements.txt`.
*   **Consideration:** Identify and add any new direct or indirect dependencies required by `flow` to `requirements.txt`.

### 4.2. Backend (FastAPI) Modifications

#### 4.2.1. New API Routes
*   **Action:** Create new API endpoints (e.g., `/api/flow/config`, `/api/flow/train/start`, `/api/flow/train/status`, `/api/flow/models`) to manage FLOW-specific training.
*   **Consideration:** Design routes to mirror existing Kohya SS training patterns for consistency.

#### 4.2.2. Data Preparation & Management
*   **Action:** Develop a utility or service (e.g., `flow_dataset_preparer.py`) to convert existing dataset metadata or allow users to create new metadata in FLOW's required JSONL format. This might involve:
    *   Reading image paths and captions from existing datasets.
    *   Generating `width`, `height`, `is_tag_based`, `is_url_based` fields.
    *   Handling `filename` as either local paths or S3 URLs.
*   **Integration:** Adapt existing `dataset_manager.py` or create a new `flow_dataset_manager` to handle FLOW-specific data loading (S3/local).

#### 4.2.3. FLOW Training Orchestration
*   **Action:** Create a `flow_training_manager.py` (similar to `kohya_training_manager.py`) responsible for:
    *   Receiving JSON configuration from the API.
    *   Dynamically generating or modifying `training_config.json` for FLOW.
    *   Executing FLOW's training script (`train_mp.py`) as a subprocess.
    *   Managing multi-GPU invocation and environment setup.
    *   Monitoring the subprocess and capturing logs/status.
*   **Consideration:** Ensure proper handling of `wandb_key`, `hf_repo_id`, `hf_token` as configured by the user.

#### 4.2.4. Model & Checkpoint Management
*   **Action:** Implement logic to allow users to specify paths for `chroma_path`, `vae_path`, `t5_path`, `t5_config_path`, `t5_tokenizer_path` via the API.
*   **Integration:** Ensure these paths correctly reference downloaded/managed models within the project's `pretrained_model` or `output` directories.

### 4.3. Frontend (Next.js) Modifications

#### 4.3.1. New UI Pages/Components
*   **Action:** Develop new React components and pages under `frontend/src/app/flow-training/` for:
    *   **Configuration Form:** A comprehensive form mirroring FLOW's `training`, `inference`, `dataloader`, and `model` JSON configuration sections.
    *   **Dataset Selection:** UI elements to select or configure the JSONL metadata file and image sources (local/S3).
    *   **Training Initiation & Monitoring:** Buttons to start/stop training, and a real-time log/status display.
    *   **Inference Results Viewer:** Displaying images generated during FLOW's `inference` runs.
*   **Consideration:** Replicate the existing UI/UX patterns for consistency.

#### 4.3.2. API Integration
*   **Action:** Connect frontend components to the new FastAPI FLOW API routes for data submission, status polling, and result retrieval.

### 4.4. Data Flow Considerations
*   **Existing Dataset -> FLOW:** Design a clear pathway for existing image datasets to be processed into the JSONL format required by FLOW. This might involve leveraging the existing tagging services or creating new ones.
*   **FLOW Output:** Integrate with FLOW's `save_folder` for checkpoints and `inference_folder` for generated images, making them accessible via the UI.

## 5. Model Management and Chroma
The core "Chroma" model will be treated as a base model for FLOW training.
*   **Selection:** Users will be able to select the `chroma_path` (a `.safetensors` file) from available base models.
*   **Version Control:** Consider how different versions of Chroma models (e.g., `chroma-8.9b.safetensors`) will be managed and presented to the user.

## 6. Testing Strategy
*   **Unit Tests:** Develop unit tests for new backend services (e.g., `flow_dataset_preparer`, `flow_training_manager`) and API endpoints.
*   **Integration Tests:** Test the full workflow from frontend configuration through backend execution and result retrieval.
*   **End-to-End Tests:** Verify that FLOW training runs successfully and produces expected outputs.

## 8. Future Expansion & Advanced Features (Inspired by SimpleTuner)
To ensure the project remains competitive and powerful, consider incorporating advanced features that align with the "easy for me" philosophy and future content support. These should be approached with a focus on compatibility and ease of integration, acknowledging licensing implications.

*   **Advanced Caching:** Implement disk-based caching for image, video, audio, and caption embeddings to accelerate training workflows. This complements FLOW's streaming dataloader.
*   **Aspect Bucketing:** Enhance the dataloader (both for FLOW and existing methods) to support varied image/video sizes and aspect ratios more robustly.
*   **Memory Optimization Techniques:** Investigate and integrate strategies like DeepSpeed & FSDP2, gradient checkpointing, and optimizer state offload to enable training larger models on more constrained hardware.
*   **S3/Cloud Storage Integration:** Further develop and abstract the S3/cloud storage capabilities for datasets, making it a first-class citizen for all training workflows (similar to FLOW's direct S3 dataloader). This will also need careful consideration of security and access management.
*   **EMA Support:** Integrate Exponential Moving Average (EMA) weights for improved model stability and quality during training.
*   **Layer Rotation & Offloading:** Expand on parameter-efficient training techniques by providing more flexible configuration for layer rotation and offloading (as seen in FLOW's `trained_single_blocks`, `trained_double_blocks`, `change_layer_every`).

## 9. Licensing Considerations
As new libraries and features are integrated, always prioritize reviewing their licenses (e.g., Apache 2.0, MIT, etc.) to ensure compatibility with the project's overall licensing and to avoid any legal impediments to distribution or modification.

## 10. Potential Challenges/Considerations
*   **JSON vs. TOML:** The project currently uses TOML for Kohya SS. FLOW uses JSON. Need to ensure a clear distinction and handling for both, possibly abstracting configuration storage.
*   **Overlap with Kohya SS:** Minimizing code duplication and clearly distinguishing between Kohya SS and FLOW workflows.
*   **Resource Management:** Carefully handle `offload_param_count` and GPU memory, especially with multi-GPU training.
*   **S3 Integration:** Robustly handle S3 credentials and secure access for the streaming dataloader.
*   **User Experience:** Ensure the UI for FLOW configuration is intuitive, given the many parameters.
*   **Log Parsing:** Adapt log parsing mechanisms to extract progress and status from FLOW's output for real-time UI updates.

This document serves as a high-level plan. Each section can be broken down into more detailed tasks as development progresses.
