# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/dataset_manager.py
import gc
import os
import platform
import subprocess
import sys
import zipfile

# Import subprocess environment utilities
try:
    from .managers import get_subprocess_environment
except ImportError:
    # Fallback if not available
    def get_subprocess_environment(project_root=None):
        return os.environ.copy()

# Simple emoji-based print functions (no colored terminal output)
def success(msg, **kwargs):
    print(f"✅ {msg}")
def error(msg, **kwargs):
    print(f"❌ {msg}")
def warning(msg, **kwargs):
    print(f"⚠️ {msg}")
def info(msg, **kwargs):
    print(f"ℹ️ {msg}")
def progress(msg, **kwargs):
    print(f"🚀 {msg}")
def print_header(text, **kwargs):
    print(f"\n=== {text} ===")
def cprint(*args, **kwargs):
    # Simple replacement for colored prints
    print(*args)

class DatasetManager:
    def __init__(self, model_manager=None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_manager = model_manager
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")

    def create_fiftyone_dataset(self, dataset_path):
        """Convert real WD14-tagged dataset to FiftyOne format for viewing"""
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError("FiftyOne is required for dataset exploration. Install with: pip install fiftyone")

        import os

        # Load images first
        dataset = fo.Dataset.from_images_dir(
            dataset_path,
            recursive=True
        )

        print(f"📁 Loaded {len(dataset)} images, integrating captions...")

        # Integrate real caption data from .txt files
        for sample in dataset:
            image_path = sample.filepath
            caption_path = os.path.splitext(image_path)[0] + '.txt'

            # Add folder metadata if in Kohya format (e.g., "10_character_name/")
            folder_name = os.path.basename(os.path.dirname(image_path))
            if '_' in folder_name and folder_name.split('_')[0].isdigit():
                parts = folder_name.split('_', 1)
                sample['repeats'] = int(parts[0])
                sample['concept'] = parts[1] if len(parts) > 1 else 'unknown'
            else:
                # Flat directory - no folder structure
                sample['repeats'] = 1
                sample['concept'] = os.path.basename(dataset_path)

            # Load caption file if it exists
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption_content = f.read().strip()

                    # Store raw caption
                    sample['caption_raw'] = caption_content

                    # Parse tags (real format: "trigger, tag1, tag2, tag3")
                    if caption_content:
                        # Split on comma+space (real WD14 format)
                        all_tags = [tag.strip() for tag in caption_content.split(', ') if tag.strip()]

                        # First tag is typically the trigger word
                        if all_tags:
                            sample['trigger_word'] = all_tags[0]
                            sample['wd14_tags'] = all_tags[1:] if len(all_tags) > 1 else []
                            sample['all_tags'] = all_tags
                            sample['tag_count'] = len(all_tags)
                        else:
                            sample['trigger_word'] = ''
                            sample['wd14_tags'] = []
                            sample['all_tags'] = []
                            sample['tag_count'] = 0
                    else:
                        # Empty caption file
                        sample['caption_raw'] = ''
                        sample['trigger_word'] = ''
                        sample['wd14_tags'] = []
                        sample['all_tags'] = []
                        sample['tag_count'] = 0

                except Exception as e:
                    print(f"⚠️ Error reading caption {caption_path}: {e}")
                    sample['caption_raw'] = ''
                    sample['trigger_word'] = ''
                    sample['wd14_tags'] = []
                    sample['all_tags'] = []
                    sample['tag_count'] = 0
            else:
                # No caption file found
                sample['caption_raw'] = ''
                sample['trigger_word'] = ''
                sample['wd14_tags'] = []
                sample['all_tags'] = []
                sample['tag_count'] = 0

            sample.save()

        print("✅ Caption integration complete")
        return dataset

    def launch_dataset_explorer(self, dataset_path):
        """Launch FiftyOne with server-optimized configuration"""
        try:
            import fiftyone as fo
        except ImportError:
            print("❌ FiftyOne dataset exploration requires FiftyOne:")
            print("   pip install fiftyone")
            return None

        # Use server-friendly configuration
        try:
            from .fiftyone_server_config import \
                create_server_friendly_dataset_launcher
            launcher = create_server_friendly_dataset_launcher()
            session, server_config = launcher(dataset_path)
        except ImportError:
            print("⚠️ Server config not available, using basic FiftyOne setup...")
            # Fallback to basic setup
            dataset = self.create_fiftyone_dataset(dataset_path)
            session = fo.launch_app(dataset, auto=False)
            server_config = {}

        # Add custom views and analysis for LoRA training
        dataset = session.dataset
        results = {}

        try:
            # Duplicate detection view (optional - can be resource intensive)
            print("🔍 Computing dataset similarity for duplicate detection...")
            duplicates_view = dataset.compute_similarity(brain_key="image_similarity")
            results['duplicates'] = duplicates_view
            print(f"✅ Duplicate detection computed. Found {len(duplicates_view.matches)} potential duplicates.")
        except Exception as e:
            print(f"⚠️ Duplicate detection failed: {e}")
            results['duplicates'] = None

        try:
            # Tag distribution analysis (using real field names)
            print("📊 Analyzing tag distribution...")

            # Count all tags across all samples
            all_tags_counter = {}
            trigger_words_counter = {}

            for sample in dataset:
                # Count individual WD14 tags
                if hasattr(sample, 'wd14_tags') and sample.wd14_tags:
                    for tag in sample.wd14_tags:
                        all_tags_counter[tag] = all_tags_counter.get(tag, 0) + 1

                # Count trigger words
                if hasattr(sample, 'trigger_word') and sample.trigger_word:
                    trigger_words_counter[sample.trigger_word] = trigger_words_counter.get(sample.trigger_word, 0) + 1

            results['tag_stats'] = all_tags_counter
            results['trigger_stats'] = trigger_words_counter

            print(f"✅ Tag analysis complete: {len(all_tags_counter)} unique tags, {len(trigger_words_counter)} trigger words")
        except Exception as e:
            print(f"⚠️ Tag analysis failed: {e}")
            results['tag_stats'] = {}
            results['trigger_stats'] = {}

        try:
            # Image quality analysis
            quality_metrics = self.analyze_image_quality(dataset)
            results['quality_metrics'] = quality_metrics
        except Exception as e:
            print(f"⚠️ Quality analysis failed: {e}")
            results['quality_metrics'] = None

        try:
            # LoRA-specific views
            lora_views = self.create_lora_specific_views(dataset)
            results['lora_views'] = lora_views
        except Exception as e:
            print(f"⚠️ LoRA views failed: {e}")
            results['lora_views'] = None

        # Print server access information
        if server_config.get('external_url'):
            print(f"\n🌍 Access FiftyOne at: {server_config['external_url']}")
        elif server_config.get('address') and server_config.get('port'):
            print(f"\n🔗 Access FiftyOne at: http://{server_config['address']}:{server_config['port']}")

        return session, results['duplicates'], results['tag_stats'], results['quality_metrics'], results['lora_views']

    def analyze_image_quality(self, dataset):
        """
        Performs basic image quality analysis on a FiftyOne dataset.
        Returns a dictionary of quality metrics.
        """
        try:
            import fiftyone as fo
        except ImportError:
            print("❌ Image quality analysis requires FiftyOne: pip install fiftyone")
            return None

        import numpy as np
        from PIL import Image

        print("📊 Analyzing image quality (blur, resolution, format consistency)...")

        quality_metrics = {
            "total_images": len(dataset),
            "blurred_images": 0,
            "min_resolution": {"width": float('inf'), "height": float('inf')},
            "max_resolution": {"width": 0, "height": 0},
            "common_formats": {},
            "avg_aspect_ratio": 0.0,
            "aspect_ratios": []
        }

        blur_threshold = 500.0 # Example threshold, can be adjusted

        for sample in dataset:
            try:
                # Resolution and aspect ratio
                img = Image.open(sample.filepath)
                width, height = img.size
                quality_metrics["min_resolution"]["width"] = min(quality_metrics["min_resolution"]["width"], width)
                quality_metrics["min_resolution"]["height"] = min(quality_metrics["min_resolution"]["height"], height)
                quality_metrics["max_resolution"]["width"] = max(quality_metrics["max_resolution"]["width"], width)
                quality_metrics["max_resolution"]["height"] = max(quality_metrics["max_resolution"]["height"], height)

                aspect_ratio = width / height
                quality_metrics["aspect_ratios"].append(aspect_ratio)

                # Format consistency
                img_format = img.format
                quality_metrics["common_formats"][img_format] = quality_metrics["common_formats"].get(img_format, 0) + 1

                # Blur detection (using a simple variance of Laplacian)
                # Requires OpenCV, which might not be a direct dependency of FiftyOne
                # For now, this will be a placeholder or require explicit cv2 import
                # try:
                #     import cv2
                #     gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                #     fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                #     if fm < blur_threshold:
                #         quality_metrics["blurred_images"] += 1
                # except ImportError:
                #     print("Warning: OpenCV not found, skipping blur detection.")
                #     pass # Skip blur detection if OpenCV is not installed

            except Exception as e:
                print(f"Warning: Could not analyze {sample.filepath}: {e}")

        if quality_metrics["aspect_ratios"]:
            quality_metrics["avg_aspect_ratio"] = np.mean(quality_metrics["aspect_ratios"])

        print("✅ Image quality analysis complete.")
        return quality_metrics

    def create_lora_specific_views(self, dataset):
        """
        Creates FiftyOne views for LoRA-specific analysis (repeat folders, concepts).
        """
        try:
            import fiftyone as fo
        except ImportError:
            print("❌ LoRA-specific views require FiftyOne: pip install fiftyone")
            return None

        lora_views = {}

        # Repeat Folder Analysis View
        if "repeats" in dataset.first().field_names:
            print("📊 Creating view for Repeat Folder Analysis...")
            # Group by 'repeats' and sort by repeat count
            lora_views["repeat_analysis_view"] = dataset.group_by("repeats").sort_by("repeats")
            print("✅ Repeat Folder Analysis view created.")

        # Concept Separation View
        if "concept" in dataset.first().field_names:
            print("📊 Creating view for Concept Separation...")
            # Group by 'concept'
            lora_views["concept_separation_view"] = dataset.group_by("concept")
            print("✅ Concept Separation view created.")

        return lora_views

    def apply_curation_to_dataset(self, dataset_path, fiftyone_dataset=None):
        """
        Applies changes made in FiftyOne (e.g., tag edits) back to the local dataset files.
        """
        try:
            import fiftyone as fo
        except ImportError:
            print("❌ Curation requires FiftyOne: pip install fiftyone")
            return False

        import os

        print(f"💾 Applying FiftyOne curation to {dataset_path}...")
        try:
            # Use provided FiftyOne dataset or load from directory
            if fiftyone_dataset is not None:
                dataset = fiftyone_dataset
                print("📊 Using active FiftyOne session dataset")
            else:
                print("📁 Loading dataset from directory...")
                # Load the FiftyOne dataset (assuming it's still active or can be reloaded)
                # For simplicity, we'll create a new dataset object from the directory
                # In a real scenario, you might want to get the active session's dataset
                dataset = fo.Dataset.from_images_dir(
                    dataset_path,
                    tags_field="wd14_tags",
                    recursive=True
                )

            files_updated = 0
            images_deleted = 0 # Placeholder for future deletion logic

            for sample in dataset:
                # Check for tag changes using real field structure
                caption_filepath = os.path.splitext(sample.filepath)[0] + ".txt"

                # Reconstruct caption from FiftyOne fields
                if hasattr(sample, 'all_tags') and sample.all_tags:
                    # Use all_tags if available (preserves order)
                    current_tags = ", ".join(sample.all_tags)
                elif hasattr(sample, 'trigger_word') or hasattr(sample, 'wd14_tags'):
                    # Reconstruct from trigger + wd14_tags
                    tags_list = []
                    if hasattr(sample, 'trigger_word') and sample.trigger_word:
                        tags_list.append(sample.trigger_word)
                    if hasattr(sample, 'wd14_tags') and sample.wd14_tags:
                        tags_list.extend(sample.wd14_tags)
                    current_tags = ", ".join(tags_list)
                elif hasattr(sample, 'caption_raw') and sample.caption_raw:
                    # Use raw caption if available
                    current_tags = sample.caption_raw
                else:
                    # No tags available
                    continue

                # Compare with existing caption file
                try:
                    with open(caption_filepath, 'r', encoding='utf-8') as f:
                        existing_tags = f.read().strip()
                except FileNotFoundError:
                    existing_tags = "" # No existing caption file

                # Update if changed
                if existing_tags != current_tags:
                    with open(caption_filepath, 'w', encoding='utf-8') as f:
                        f.write(current_tags)
                    print(f"  ✅ Updated tags for {os.path.basename(sample.filepath)}")
                    files_updated += 1

                # Future: Handle image deletion if a 'deleted' field is added by FiftyOne curation
                # if "deleted" in sample and sample.deleted:
                #     os.remove(sample.filepath)
                #     images_deleted += 1
                #     print(f"  Deleted image: {os.path.basename(sample.filepath)}")

            print(f"✅ Curation applied: {files_updated} caption files updated, {images_deleted} images deleted.")
            return True

        except Exception as e:
            print(f"❌ Error applying curation changes: {e}")
            return False

    def _detect_environment_type(self):
        """Detect the current environment to choose the best tagger strategy"""
        # Check for VastAI/rental GPU environments
        if (os.path.exists("/workspace") or
            "VAST_CONTAINERLABEL" in os.environ or
            "vastai" in platform.node().lower()):
            return "vastai_rental"

        # Check for general Jupyter
        if "jupyter" in sys.modules:
            return "jupyter_local"

        # Default to local environment
        return "local"

    def _get_tagger_script_path(self):
        """Get the appropriate tagger script based on environment"""
        env_type = self._detect_environment_type()

        if env_type == "vastai_rental":
            # Use robust custom tagger for unstable rental GPU environments
            custom_tagger = os.path.join(self.project_root, "custom", "tag_images_by_wd14_tagger.py")
            if os.path.exists(custom_tagger):
                print("🏷️ Using WD14 tagger")
                return custom_tagger

        elif env_type == "jupyter_local":
            # Local Jupyter - prefer custom tagger for stability
            custom_tagger = os.path.join(self.project_root, "custom", "tag_images_by_wd14_tagger.py")
            if os.path.exists(custom_tagger):
                print("📚 Using custom tagger for local Jupyter")
                return custom_tagger

        # Fallback to Derrian's backend (Kohya-based)
        derrian_tagger = os.path.join(self.sd_scripts_dir, "finetune", "tag_images_by_wd14_tagger.py")
        if os.path.exists(derrian_tagger):
            print("⚙️ Using Derrian's backend tagger (Kohya-based)")
            return derrian_tagger

        # Final fallback to custom if nothing else works
        custom_tagger = os.path.join(self.project_root, "custom", "tag_images_by_wd14_tagger.py")
        print("🔄 Falling back to custom tagger as last resort")
        return custom_tagger

    def extract_dataset(self, zip_path, extract_to_dir, hf_token=""):
        if not extract_to_dir:
            print("Error: Please specify a directory to extract to.")
            return False

        # Handle Hugging Face URL
        if zip_path.startswith("https://huggingface.co/"):
            print("Downloading dataset from Hugging Face...")
            # Lazy load ModelManager only when needed for downloads
            if not self.model_manager:
                from shared_managers import get_model_manager
                self.model_manager = get_model_manager()
            downloaded_zip_path = self.model_manager.download_file(zip_path, self.project_root, hf_token)
            if not downloaded_zip_path:
                print("Failed to download the dataset.")
                return False
            zip_path = downloaded_zip_path

        if not os.path.exists(zip_path):
            print(f"Error: Zip file not found at '{zip_path}'")
            return False

        extract_path = os.path.join(self.project_root, extract_to_dir)
        os.makedirs(extract_path, exist_ok=True)

        print(f"Extracting dataset to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(extract_path)
            print("Extraction complete.\n")
            return True
        except zipfile.BadZipFile:
            print("Error: The file is not a valid zip file.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            return False
        finally:
            # Clean up downloaded zip file if it was from HF
            if 'downloaded_zip_path' in locals() and os.path.exists(downloaded_zip_path):
                os.remove(downloaded_zip_path)
            # Force garbage collection after dataset processing
            gc.collect()

    def tag_images(self, dataset_dir, method, tagger_model, threshold, blacklist_tags="", caption_extension=".txt"):
        """Enhanced image tagging with more options"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if method == "anime":
            return self._tag_images_wd14(dataset_path, tagger_model, threshold, blacklist_tags, caption_extension)
        elif method == "photo":
            return self._tag_images_blip(dataset_path, caption_extension)
        else:
            print(f"❌ Unknown tagging method: {method}")
            return False

    def _tag_images_wd14(self, dataset_path, tagger_model, threshold, blacklist_tags, caption_extension):
        """Tag images using WD14 tagger"""
        # Known working tagger models (fallback list)
        working_models = [
            tagger_model,  # User's choice first
            "SmilingWolf/wd-vit-large-tagger-v3",  # Reliable fallback
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2"  # Stable older model
        ]

        # First ensure the tagger model is available
        self._ensure_tagger_model_available(tagger_model)

        # Always use current Python executable for environment-agnostic execution
        # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
        venv_python = sys.executable
        print(f"🐍 Using current Python environment: {venv_python}")

        tagger_script = self._get_tagger_script_path()

        # Check if tagger script exists
        if not os.path.exists(tagger_script):
            print(f"❌ Tagger script not found at: {tagger_script}")
            print("💡 This usually means the SD-scripts setup is incomplete")
            print("💡 Try running the setup widget first to install the training backend")
            return False

        # Try each model until one works
        for attempt, model_to_try in enumerate(working_models):
            print(f"🏷️ Attempting tagging with {model_to_try.split('/')[-1]} (threshold: {threshold})...")

            # Build command with enhanced options
            # Models are in wd14_tagger_model/ directory (default location)
            command = [
                venv_python, tagger_script,
                dataset_path,
                "--repo_id", model_to_try,
                "--model_dir", "wd14_tagger_model",  # Use relative path for cross-platform compatibility
                "--force_download",  # Force download to local model_dir instead of using HF cache
                "--thresh", str(threshold),
                "--batch_size", "8" if "v3" in model_to_try or "swinv2" in model_to_try else "1",
                "--max_data_loader_n_workers", "2",
                "--caption_extension", caption_extension,
                "--remove_underscore"  # Convert underscores to spaces
            ]

            # Try ONNX but don't force it - graceful fallback to PyTorch
            use_onnx = False
            try:
                # Simple import test first
                test_onnx_cmd = [venv_python, "-c", "import onnxruntime; print('OK')"]
                result = subprocess.run(test_onnx_cmd, check=True, capture_output=True, text=True, timeout=5)
                if result.stdout.strip() == 'OK':
                    use_onnx = True
                    if "v3" in model_to_try:
                        print("🚀 ONNX available - using for v3 model (faster inference)")
                    else:
                        print("✅ ONNX available - using accelerated inference")
            except Exception as e:
                print("⚠️ ONNX not working - using PyTorch inference (slower but stable)")
                print(f"   Reason: {str(e)[:80]}...")

            # Only add ONNX flag if we're confident it will work
            if use_onnx:
                command.append("--onnx")
            else:
                print("   📝 Note: WD14 tagger will use PyTorch backend")

            # Add blacklisted tags if provided
            if blacklist_tags:
                command.extend(["--undesired_tags", blacklist_tags.replace(" ", "")])
                print(f"🚫 Blacklisted tags: {blacklist_tags}")

            # Try running the command
            if self._run_subprocess(command, f"Image tagging (attempt {attempt + 1})"):
                print(f"✅ Successfully tagged with {model_to_try.split('/')[-1]}")
                # Force garbage collection after successful tagging
                gc.collect()
                return True
            else:
                print(f"❌ Failed with {model_to_try.split('/')[-1]}")
                if attempt < len(working_models) - 1:
                    print("🔄 Trying fallback model...")

        print("❌ All tagger models failed")
        # Force garbage collection after tagging attempts
        gc.collect()
        return False

    def _tag_images_blip(self, dataset_path, caption_extension):
        """Tag images using BLIP captioning"""
        # Always use current Python executable for environment-agnostic execution
        # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
        venv_python = sys.executable
        print(f"🐍 Using current Python environment: {venv_python}")

        blip_script = os.path.join(self.sd_scripts_dir, "finetune/make_captions.py")

        # Check if BLIP script exists
        if not os.path.exists(blip_script):
            print(f"❌ BLIP script not found at: {blip_script}")
            print("💡 This usually means the SD-scripts setup is incomplete")
            print("💡 Try running the setup widget first to install the training backend")
            return False

        command = [
            venv_python, blip_script,
            dataset_path,
            "--beam_search",
            "--max_data_loader_n_workers", "2",
            "--batch_size", "8",
            "--min_length", "10",
            "--max_length", "75",
            "--caption_extension", caption_extension
        ]

        print("📸 Generating BLIP captions...")
        return self._run_subprocess(command, "BLIP captioning")

    def _run_subprocess(self, command, task_name):
        """Helper to run subprocess with consistent error handling"""
        try:
            env = os.environ.copy()

            # NOTE: Removed custom_scheduler PYTHONPATH injection
            # The custom_scheduler should only be used for training, not for tagging operations
            # Adding it here was breaking HuggingFace model loading in tagger subprocesses

            # Add common CUDA/CuDNN paths to LD_LIBRARY_PATH (keep existing logic!)
            cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
            new_ld_library_path = f"{cuda_path}/lib64:{cuda_path}/extras/CUPTI/lib64"
            # Add common CuDNN paths (adjust if your CuDNN is elsewhere)
            cudnn_paths = [
                f"{cuda_path}/lib",  # Common for some CuDNN installations
                f"{cuda_path}/targets/x86_64-linux/lib", # Common for newer CUDA/CuDNN
                "/usr/lib/x86_64-linux-gnu", # System-wide CuDNN
            ]
            for p in cudnn_paths:
                if os.path.exists(p):
                    new_ld_library_path += f":{p}"

            if "LD_LIBRARY_PATH" in env:
                env["LD_LIBRARY_PATH"] = f"{new_ld_library_path}:{env['LD_LIBRARY_PATH']}"
            else:
                env["LD_LIBRARY_PATH"] = new_ld_library_path

            print(f"Setting LD_LIBRARY_PATH for tagger: {env['LD_LIBRARY_PATH']}")
            if "PYTHONPATH" in env:
                # The environment setup is reused, so we log the actual path being set
                if "PYTHONPATH" in env and env["PYTHONPATH"]:
                    print(f"ℹ️  Setting PYTHONPATH for subprocess: {env['PYTHONPATH']}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.project_root,
                env=env # Pass the modified environment
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')

            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            print(f"\n✅ {task_name} complete.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during {task_name.lower()}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during {task_name.lower()}: {e}")
            return False

    def add_trigger_word(self, dataset_dir, trigger_word):
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory not found at {dataset_path}")
            return False

        if not trigger_word:
            print("Error: Please specify a trigger word.")
            return False

        try:
            print(f"Adding trigger word '{trigger_word}' to caption files...")
            files_processed = 0
            for item in os.listdir(dataset_path):
                if item.endswith(".txt"):
                    file_path = os.path.join(dataset_path, item)
                    try:
                        # Read the original content first
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()

                        # Write the new content with trigger word prepended
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(f"{trigger_word}, {content}")

                        print(f"  Added to: {item}")
                        files_processed += 1
                    except Exception as e:
                        print(f"Error processing {item}: {e}")
            print(f"✅ Trigger word addition complete. Processed {files_processed} files.")
            return True
        except Exception as e:
            print(f"❌ Error adding trigger word: {e}")
            return False

    def remove_tags(self, dataset_dir, tags_to_remove):
        """Remove specified tags from all caption files"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if not tags_to_remove:
            print("❌ Error: Please specify tags to remove.")
            return False

        # Parse tags (handle both comma and space separated)
        tags_list = [tag.strip() for tag in tags_to_remove.replace(',', ' ').split() if tag.strip()]

        print(f"🗑️ Removing tags from caption files: {', '.join(tags_list)}...")
        files_processed = 0

        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    # Remove each tag
                    modified = False
                    for tag in tags_list:
                        # Remove tag with various separators
                        patterns = [f"{tag}, ", f", {tag}", f" {tag}, ", f", {tag} ", tag]
                        for pattern in patterns:
                            if pattern in content:
                                content = content.replace(pattern, "")
                                modified = True

                    # Clean up any double commas or extra spaces
                    content = ', '.join([t.strip() for t in content.split(',') if t.strip()])

                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content + "\n")
                        files_processed += 1
                        print(f"  ✓ Updated: {item}")

                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")

        print(f"✅ Tag removal complete. {files_processed} files updated.")
        return True

    def _ensure_tagger_model_available(self, tagger_model):
        """Ensure the tagger model is downloaded and available"""
        print(f"🔍 Checking tagger model availability: {tagger_model}")

        # Try to import huggingface_hub to check model availability
        try:
            import torch
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            print("⚠️ huggingface_hub or torch not available - model download may fail")
            print("💡 The tagger script will attempt to download automatically")
            return

        try:
            # Check if model is already cached
            model_path = snapshot_download(repo_id=tagger_model, allow_patterns=["*.onnx", "*.json"])
            print(f"✅ Tagger model found at: {model_path}")

        except Exception:
            print(f"📥 Downloading tagger model {tagger_model}...")
            try:
                # Download the model files we need
                files_to_download = [
                    "model.onnx",
                    "selected_tags.csv",
                    "config.json"
                ]

                for file in files_to_download:
                    try:
                        hf_hub_download(repo_id=tagger_model, filename=file)
                        print(f"  ✅ Downloaded: {file}")
                    except Exception as file_error:
                        print(f"  ⚠️ Could not download {file}: {file_error}")

                print(f"✅ Tagger model {tagger_model.split('/')[-1]} ready!")

            except Exception as download_error:
                print(f"⚠️ Model download failed: {download_error}")

    def scrape_from_gelbooru(self, tags, dataset_dir, limit=100, confirm_callback=None):
        """
        Scrape and download images from Gelbooru based on tags
        
        Args:
            tags (str): Gelbooru tags (e.g., "1girl, blue_hair, -solo")
            dataset_dir (str): Directory to save images
            limit (int): Maximum number of images to download
            confirm_callback (callable): Function to call for confirmation before download
        """
        import os
        from urllib.parse import urlparse

        import requests

        print(f"🔍 Searching Gelbooru for: {tags}")
        print(f"📁 Will save to: {dataset_dir}")
        print(f"📊 Max images: {limit}")

        # Ask for confirmation if callback provided
        if confirm_callback and not confirm_callback():
            print("❌ Download cancelled by user")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(dataset_dir, exist_ok=True)

            # Build Gelbooru API URL
            api_url = "https://gelbooru.com/index.php"
            params = {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "json": "1",
                "tags": tags,
                "limit": min(limit, 1000)  # Gelbooru max is 1000
            }

            print("📡 Fetching image URLs from Gelbooru...")

            # Add headers to avoid 401 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = requests.get(api_url, params=params, headers=headers, timeout=30)

            if response.status_code == 401:
                print("❌ Gelbooru API returned 401 Unauthorized")
                print("💡 This might be due to:")
                print("   - Rate limiting (wait a few minutes)")
                print("   - API changes requiring authentication")
                print("   - Blocked IP or user agent")
                print("   - Try using different tags or smaller limits")
                return False

            response.raise_for_status()

            data = response.json()
            if not data or not isinstance(data, list):
                print("❌ No images found or invalid response from Gelbooru")
                return False

            print(f"✅ Found {len(data)} images matching your tags")

            # Extract image URLs
            image_urls = []
            for post in data:
                if 'file_url' in post and post['file_url']:
                    # Filter for common image formats
                    file_url = post['file_url']
                    if any(file_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        image_urls.append(file_url)

            if not image_urls:
                print("❌ No valid image URLs found")
                return False

            print(f"📥 Starting download of {len(image_urls)} images...")

            # Download images using aria2c (if available) or fallback to requests
            downloaded_count = 0

            # Method 1: Try aria2c first (fastest for batch downloads)
            if shutil.which("aria2c"):
                print("🚀 Attempting batch download with aria2c...")
                try:
                    # Create temporary URL list file for aria2c
                    url_file = os.path.join(dataset_dir, "temp_urls.txt")
                    with open(url_file, 'w') as f:
                        for url in image_urls:
                            f.write(f"{url}\n")

                    # Use aria2c for fast parallel downloads
                    import subprocess
                    cmd = [
                        'aria2c',
                        '-i', url_file,
                        '-d', dataset_dir,
                        '-j', '4',  # 4 parallel downloads
                        '-x', '2',  # 2 connections per download
                        '--auto-file-renaming=false',
                        '--allow-overwrite=true'
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)

                    # Clean up temp file
                    os.remove(url_file)

                    if result.returncode == 0:
                        downloaded_count = len([f for f in os.listdir(dataset_dir)
                                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                        print(f"✅ Downloaded {downloaded_count} images using aria2c")
                        return downloaded_count > 0
                    else:
                        print("❌ aria2c failed. Trying wget...")

                except Exception as e:
                    print(f"❌ Error with aria2c: {e}. Trying wget...")
            else:
                print("⚠️ aria2c not available. Trying wget...")

            # Method 2: Try wget for batch downloads  
            if shutil.which("wget"):
                print("🚀 Attempting batch download with wget...")
                try:
                    downloaded_count = 0
                    for i, url in enumerate(image_urls):
                        # Get filename from URL
                        parsed_url = urlparse(url)
                        filename = os.path.basename(parsed_url.path)
                        if not filename or '.' not in filename:
                            filename = f"gelbooru_image_{i+1}.jpg"

                        filepath = os.path.join(dataset_dir, filename)

                        # Skip if file already exists
                        if os.path.exists(filepath):
                            downloaded_count += 1
                            continue

                        # Download with wget
                        cmd = ["wget", "-O", filepath, url]
                        result = subprocess.run(cmd, capture_output=True)
                        
                        if result.returncode == 0:
                            downloaded_count += 1
                            if downloaded_count % 10 == 0:
                                print(f"📥 Downloaded {downloaded_count}/{len(image_urls)} images with wget...")
                        else:
                            print(f"⚠️ wget failed for {url}")

                    if downloaded_count > 0:
                        print(f"✅ Downloaded {downloaded_count} images using wget")
                        return downloaded_count > 0
                    else:
                        print("❌ wget batch download failed. Falling back to Python requests...")
                        
                except Exception as e:
                    print(f"❌ Error with wget: {e}. Falling back to Python requests...")
            else:
                print("⚠️ wget not available. Falling back to Python requests...")

            # Method 3: Fallback to Python requests
            print("🚀 Attempting download with Python requests...")
            downloaded_count = 0
            for i, url in enumerate(image_urls):
                try:
                    # Get filename from URL
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                    if not filename or '.' not in filename:
                        filename = f"gelbooru_image_{i+1}.jpg"

                    filepath = os.path.join(dataset_dir, filename)

                    # Skip if file already exists
                    if os.path.exists(filepath):
                        downloaded_count += 1
                        continue

                    # Download the image
                    img_response = requests.get(url, timeout=30, stream=True)
                    img_response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded_count += 1
                    if downloaded_count % 10 == 0:
                        print(f"📥 Downloaded {downloaded_count}/{len(image_urls)} images...")

                except Exception as img_error:
                    print(f"⚠️ Failed to download {url}: {img_error}")
                    continue

            print(f"🎉 Successfully downloaded {downloaded_count} images to {dataset_dir}")
            return downloaded_count > 0

        except Exception as e:
            print(f"❌ Gelbooru scraping failed: {e}")
            return False

    def search_and_replace_tags(self, dataset_dir, search_tags, replace_with="", search_mode="AND"):
        """
        Search and replace tags across all caption files in dataset
        
        Args:
            dataset_dir (str): Dataset directory path
            search_tags (str): Tags to search for (comma or space separated)
            replace_with (str): What to replace with (empty string to remove)
            search_mode (str): "AND" or "OR" - how to match multiple search tags
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if not search_tags.strip():
            print("❌ Error: Please specify tags to search for.")
            return False

        # Parse search tags
        search_list = [tag.strip() for tag in search_tags.replace(',', ' ').split() if tag.strip()]
        replace_text = replace_with.strip() if replace_with else ""

        print(f"🔍 Searching for tags: {', '.join(search_list)} (mode: {search_mode})")
        if replace_text:
            print(f"🔄 Will replace with: '{replace_text}'")
        else:
            print("🗑️ Will remove matching tags")

        files_processed = 0
        total_replacements = 0

        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    original_content = content

                    # Split content into tags
                    tags = [tag.strip() for tag in content.split(',') if tag.strip()]

                    # Apply search and replace logic
                    new_tags = []
                    file_replacements = 0

                    for tag in tags:
                        tag_matches = []
                        for search_tag in search_list:
                            if search_tag.lower() in tag.lower():
                                tag_matches.append(True)
                            else:
                                tag_matches.append(False)

                        # Check if tag should be replaced based on search mode
                        should_replace = False
                        if search_mode.upper() == "AND":
                            should_replace = all(tag_matches)
                        else:  # OR mode
                            should_replace = any(tag_matches)

                        if should_replace:
                            if replace_text:
                                new_tags.append(replace_text)
                            # If replace_text is empty, tag is removed (not added to new_tags)
                            file_replacements += 1
                        else:
                            new_tags.append(tag)

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_tags = []
                    for tag in new_tags:
                        if tag not in seen:
                            unique_tags.append(tag)
                            seen.add(tag)

                    new_content = ', '.join(unique_tags)

                    if new_content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content + "\n")
                        files_processed += 1
                        total_replacements += file_replacements
                        print(f"  ✓ Updated {item}: {file_replacements} replacements")

                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")

        print(f"✅ Search and replace complete. {files_processed} files updated, {total_replacements} total replacements.")
        return True

    def sort_and_deduplicate_tags(self, dataset_dir, sort_alphabetically=True, remove_duplicates=True):
        """
        Sort tags alphabetically and/or remove duplicate tags within each caption file
        
        Args:
            dataset_dir (str): Dataset directory path
            sort_alphabetically (bool): Whether to sort tags alphabetically
            remove_duplicates (bool): Whether to remove duplicate tags
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        print("🔧 Processing caption files...")
        if sort_alphabetically:
            print("   📝 Sorting tags alphabetically")
        if remove_duplicates:
            print("   🗑️ Removing duplicate tags")

        files_processed = 0
        total_duplicates_removed = 0

        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if not content:
                        continue

                    # Split content into tags
                    tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                    original_count = len(tags)

                    # Remove duplicates if requested
                    if remove_duplicates:
                        # Use dict.fromkeys() to preserve order while removing duplicates
                        # Convert to lowercase for comparison but keep original case
                        seen = {}
                        unique_tags = []
                        for tag in tags:
                            tag_lower = tag.lower()
                            if tag_lower not in seen:
                                seen[tag_lower] = True
                                unique_tags.append(tag)
                        tags = unique_tags

                    # Sort alphabetically if requested
                    if sort_alphabetically:
                        tags.sort(key=str.lower)

                    new_content = ', '.join(tags)

                    # Write back if content changed
                    if new_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content + "\n")

                        duplicates_removed = original_count - len(tags)
                        total_duplicates_removed += duplicates_removed
                        files_processed += 1

                        if duplicates_removed > 0:
                            print(f"  ✓ {item}: removed {duplicates_removed} duplicate(s)")
                        else:
                            print(f"  ✓ {item}: sorted")

                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")

        if total_duplicates_removed > 0:
            print(f"✅ Processing complete. {files_processed} files updated, {total_duplicates_removed} duplicates removed.")
        else:
            print(f"✅ Processing complete. {files_processed} files updated.")
        return True

    def preview_rename_files(self, dataset_dir, project_name, pattern="numbered", start_number=1):
        """Preview what files will be renamed to (no actual renaming)"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found at {dataset_path}")
            return []

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = []

        for file in os.listdir(dataset_path):
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_files.append(file)

        if not image_files:
            print("❌ No image files found in dataset directory")
            return []

        # Sort for consistent numbering
        image_files.sort()

        # Generate preview of renames
        preview_data = []
        for i, filename in enumerate(image_files):
            old_path = os.path.join(dataset_path, filename)
            old_name, ext = os.path.splitext(filename)

            if pattern == "numbered":
                new_name = f"{project_name}_{start_number + i:03d}{ext}"
            elif pattern == "sanitized":
                # Keep original name but sanitize it
                clean_name = self._sanitize_filename(old_name)
                new_name = f"{project_name}_{clean_name}{ext}"
            elif pattern == "timestamp":
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d")
                new_name = f"{project_name}_{timestamp}_{start_number + i:03d}{ext}"
            else:
                new_name = f"{project_name}_{start_number + i:03d}{ext}"

            # Check for caption file
            old_caption = os.path.join(dataset_path, f"{old_name}.txt")
            has_caption = os.path.exists(old_caption)

            preview_data.append({
                'old_name': filename,
                'new_name': new_name,
                'has_caption': has_caption,
                'old_caption': f"{old_name}.txt" if has_caption else None,
                'new_caption': f"{os.path.splitext(new_name)[0]}.txt" if has_caption else None
            })

        return preview_data

    def rename_dataset_files(self, dataset_dir, project_name, pattern="numbered", start_number=1, preview_data=None):
        """Rename files in dataset directory with consistent naming"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found at {dataset_path}")
            return False

        # Use preview data if provided, otherwise generate it
        if preview_data is None:
            preview_data = self.preview_rename_files(dataset_dir, project_name, pattern, start_number)

        if not preview_data:
            return False

        print(f"📝 Renaming {len(preview_data)} files with pattern: {pattern}")
        print(f"🎯 Project name: {project_name}")

        renamed_count = 0
        caption_count = 0

        for item in preview_data:
            old_path = os.path.join(dataset_path, item['old_name'])
            new_path = os.path.join(dataset_path, item['new_name'])

            try:
                # Rename image file
                if os.path.exists(old_path) and old_path != new_path:
                    os.rename(old_path, new_path)
                    renamed_count += 1

                    # Rename caption file if it exists
                    if item['has_caption']:
                        old_caption_path = os.path.join(dataset_path, item['old_caption'])
                        new_caption_path = os.path.join(dataset_path, item['new_caption'])

                        if os.path.exists(old_caption_path):
                            os.rename(old_caption_path, new_caption_path)
                            caption_count += 1

            except Exception as e:
                print(f"⚠️ Error renaming {item['old_name']}: {e}")
                continue

        print(f"✅ Renamed {renamed_count} image files and {caption_count} caption files")
        return True

    def _sanitize_filename(self, filename):
        """Remove problematic characters from filename"""
        import re

        # Replace problematic characters with underscores
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "file"  # Fallback if name becomes empty

    def add_emphasis_to_tags(self, dataset_dir, tags_to_emphasize, emphasis_level=1, preview_only=False):
        """
        Add emphasis weights to specific tags using (tag) syntax
        
        Args:
            dataset_dir: Dataset directory
            tags_to_emphasize: Comma-separated tags to emphasize  
            emphasis_level: Number of parentheses (1 = (tag), 2 = ((tag)))
            preview_only: Show preview without making changes
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found at {dataset_path}")
            return False

        if not tags_to_emphasize.strip():
            print("❌ Please specify tags to emphasize")
            return False

        # Parse tags
        tags_list = [tag.strip().lower() for tag in tags_to_emphasize.split(',') if tag.strip()]
        emphasis_chars = '(' * emphasis_level
        closing_chars = ')' * emphasis_level

        print(f"🎯 {'Preview:' if preview_only else 'Applying'} emphasis level {emphasis_level} to tags: {', '.join(tags_list)}")
        print(f"📝 Format: tag → {emphasis_chars}tag{closing_chars}")
        print("=" * 60)

        files_processed = 0
        changes_made = 0
        preview_samples = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.txt', '.caption')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()

                        if not content:
                            continue

                        original_content = content
                        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                        modified_tags = []
                        file_changes = 0

                        for tag in tags:
                            # Check if tag should be emphasized (case insensitive)
                            should_emphasize = any(target_tag in tag.lower() for target_tag in tags_list)

                            if should_emphasize and not tag.startswith('('):
                                # Add emphasis
                                modified_tags.append(f"{emphasis_chars}{tag}{closing_chars}")
                                file_changes += 1
                            else:
                                modified_tags.append(tag)

                        new_content = ', '.join(modified_tags)

                        if new_content != original_content:
                            files_processed += 1
                            changes_made += file_changes

                            if preview_only and len(preview_samples) < 5:
                                preview_samples.append({
                                    'file': file,
                                    'before': original_content[:100] + ('...' if len(original_content) > 100 else ''),
                                    'after': new_content[:100] + ('...' if len(new_content) > 100 else ''),
                                    'changes': file_changes
                                })
                            elif not preview_only:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(new_content + '\n')
                                print(f"  ✅ {file}: {file_changes} tags emphasized")

                    except Exception as e:
                        print(f"⚠️ Error processing {file_path}: {e}")

        if preview_only:
            print("📋 Preview of changes (first 5 files):")
            for sample in preview_samples:
                print(f"\n📁 {sample['file']} ({sample['changes']} changes)")
                print(f"  Before: {sample['before']}")
                print(f"  After:  {sample['after']}")
            print(f"\n🔍 Would process {files_processed} files, emphasizing {changes_made} tags")
            print("💡 Run with preview_only=False to apply changes")
        else:
            print(f"\n✅ Emphasis complete: {files_processed} files updated, {changes_made} tags emphasized")

        return True

    def reduce_common_tag_weights(self, dataset_dir, common_tags="1girl,solo,looking at viewer,simple background,white background", reduction_level=1, preview_only=False):
        """
        Reduce weight of common/overused tags using [tag] syntax
        
        Args:
            dataset_dir: Dataset directory
            common_tags: Comma-separated tags to de-emphasize
            reduction_level: Number of brackets (1 = [tag], 2 = [[tag]])
            preview_only: Show preview without making changes
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found at {dataset_path}")
            return False

        # Parse tags
        tags_list = [tag.strip().lower() for tag in common_tags.split(',') if tag.strip()]
        reduction_chars = '[' * reduction_level
        closing_chars = ']' * reduction_level

        print(f"🔻 {'Preview:' if preview_only else 'Applying'} reduction level {reduction_level} to common tags: {', '.join(tags_list)}")
        print(f"📝 Format: tag → {reduction_chars}tag{closing_chars}")
        print("=" * 60)

        files_processed = 0
        changes_made = 0
        preview_samples = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.txt', '.caption')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()

                        if not content:
                            continue

                        original_content = content
                        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                        modified_tags = []
                        file_changes = 0

                        for tag in tags:
                            # Check if tag should be de-emphasized (case insensitive)
                            should_reduce = any(common_tag in tag.lower() for common_tag in tags_list)

                            if should_reduce and not tag.startswith('['):
                                # Add reduction
                                modified_tags.append(f"{reduction_chars}{tag}{closing_chars}")
                                file_changes += 1
                            else:
                                modified_tags.append(tag)

                        new_content = ', '.join(modified_tags)

                        if new_content != original_content:
                            files_processed += 1
                            changes_made += file_changes

                            if preview_only and len(preview_samples) < 5:
                                preview_samples.append({
                                    'file': file,
                                    'before': original_content[:100] + ('...' if len(original_content) > 100 else ''),
                                    'after': new_content[:100] + ('...' if len(new_content) > 100 else ''),
                                    'changes': file_changes
                                })
                            elif not preview_only:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(new_content + '\n')
                                print(f"  ✅ {file}: {file_changes} tags de-emphasized")

                    except Exception as e:
                        print(f"⚠️ Error processing {file_path}: {e}")

        if preview_only:
            print("📋 Preview of changes (first 5 files):")
            for sample in preview_samples:
                print(f"\n📁 {sample['file']} ({sample['changes']} changes)")
                print(f"  Before: {sample['before']}")
                print(f"  After:  {sample['after']}")
            print(f"\n🔍 Would process {files_processed} files, de-emphasizing {changes_made} tags")
            print("💡 Run with preview_only=False to apply changes")
        else:
            print(f"\n✅ De-emphasis complete: {files_processed} files updated, {changes_made} tags de-emphasized")

        return True

    def display_dataset_tags(self, dataset_dir, max_files=100):
        """Display the generated tags/captions for review (simple text display)"""
        if not os.path.exists(dataset_dir):
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return False

        # Find all caption files
        import glob
        caption_files = []
        for ext in ['.txt', '.caption']:
            caption_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext}")))

        if not caption_files:
            print("❌ No caption/tag files found in dataset directory")
            print("💡 Run tagging first to generate caption files")
            return False

        print(f"📋 Displaying tags for {min(len(caption_files), max_files)} files:")
        print("=" * 80)

        # Sort files for consistent display
        caption_files.sort()

        for i, caption_file in enumerate(caption_files[:max_files]):
            # Get corresponding image file
            base_name = os.path.splitext(os.path.basename(caption_file))[0]
            image_file = None
            for img_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                potential_image = os.path.join(dataset_dir, f"{base_name}{img_ext}")
                if os.path.exists(potential_image):
                    image_file = os.path.basename(potential_image)
                    break

            # Read caption content
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()

                print(f"\n📸 {i+1}. {image_file or base_name}")
                print(f"🏷️  {tags}")

            except Exception as e:
                print(f"\n❌ Error reading {os.path.basename(caption_file)}: {e}")

        if len(caption_files) > max_files:
            print(f"\n... and {len(caption_files) - max_files} more files")
            print(f"💡 Showing first {max_files} files only")

        print("\n" + "=" * 80)
        print(f"✅ Total caption files found: {len(caption_files)}")
        return True

    def scrape_with_gallery_dl(self, site="gelbooru", tags="", dataset_dir="", limit_range="1-200",
                               write_tags=True, use_aria2c=True, custom_url="", sub_folder="",
                               additional_args="--filename /O --no-part"):
        """
        Advanced image scraping using gallery-dl (supports 300+ sites)
        
        Args:
            site: Site to scrape from (gelbooru, danbooru, safebooru, pixiv, twitter, etc.)
            tags: Tags to search for (comma or space separated)
            dataset_dir: Directory to save images
            limit_range: Range of images to download (e.g., "1-200", "1-50")
            write_tags: Whether to download and process tag files
            use_aria2c: Use aria2c for faster parallel downloads
            custom_url: Custom URL instead of using predefined site
            sub_folder: Organize images into subfolder
            additional_args: Additional gallery-dl arguments
        """
        import subprocess
        from urllib.parse import quote_plus

        # Check if gallery-dl is installed
        try:
            result = subprocess.run(['gallery-dl', '--version'], capture_output=True, text=True, timeout=5)
            print(f"🎨 Using gallery-dl version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            print("❌ gallery-dl not found. Install with: pip install gallery-dl")
            return False

        if not tags and not custom_url:
            print("❌ Please specify either tags or a custom URL")
            return False

        # Setup directories
        if dataset_dir:
            base_dir = os.path.join(self.project_root, dataset_dir)
        else:
            base_dir = os.path.join(self.project_root, "dataset")

        if sub_folder:
            if sub_folder.startswith(os.path.sep):
                image_dir = sub_folder  # Absolute path
            else:
                image_dir = os.path.join(base_dir, sub_folder)
        else:
            image_dir = base_dir

        os.makedirs(image_dir, exist_ok=True)

        print("🎨 Gallery-dl scraping setup:")
        print(f"📁 Target directory: {image_dir}")
        print(f"🏷️ Tags: {tags}")
        print(f"📊 Range: {limit_range}")
        print(f"🏃‍♂️ Aria2c: {'Enabled' if use_aria2c else 'Disabled'}")
        print("=" * 60)

        # Prepare tags for URL
        if tags:
            tag_list = [tag.strip() for tag in tags.replace(',', ' ').split() if tag.strip()]
            # URL encode special characters
            encoded_tags = '+'.join(quote_plus(tag.replace(' ', '_'), safe='') for tag in tag_list)
            print(f"🔍 Processed tags: {' + '.join(tag_list)}")

        # Build URL based on site
        if custom_url:
            url = custom_url
            print(f"🌐 Using custom URL: {url}")
        else:
            site_urls = {
                "gelbooru": f"https://gelbooru.com/index.php?page=post&s=list&tags={encoded_tags}",
                "danbooru": f"https://danbooru.donmai.us/posts?tags={encoded_tags}",
                "safebooru": f"https://safebooru.org/index.php?page=post&s=list&tags={encoded_tags}",
                "konachan": f"https://konachan.com/post?tags={encoded_tags}",
                "yande.re": f"https://yande.re/post?tags={encoded_tags}",
                "pixiv": f"https://www.pixiv.net/en/tags/{encoded_tags}/artworks",
            }

            if site.lower() not in site_urls:
                print(f"❌ Unsupported site: {site}")
                print(f"📝 Supported sites: {', '.join(site_urls.keys())}")
                print("💡 Or use custom_url parameter for other sites")
                return False

            url = site_urls[site.lower()]
            print(f"🌐 Built URL: {url}")

        # Common gallery-dl config
        base_config = {
            "user-agent": "gallery-dl/1.26.0",
            "sleep": "1",  # Be respectful with rate limiting
        }

        if limit_range:
            base_config["range"] = limit_range

        # Build gallery-dl arguments
        def build_args(config_dict):
            args = []
            for key, value in config_dict.items():
                if value is True:
                    args.append(f"--{key}")
                elif value is False:
                    continue
                else:
                    args.append(f"--{key}={value}")
            return args

        success = False

        if use_aria2c:
            print("🚀 Phase 1: Getting download URLs with gallery-dl...")

            # Phase 1: Get URLs
            url_config = {**base_config, "get-urls": True}
            url_args = build_args(url_config)

            # Add additional arguments
            if additional_args:
                url_args.extend(additional_args.split())

            try:
                # Run gallery-dl to get URLs
                cmd = ['gallery-dl'] + url_args + [url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    print(f"❌ Error getting URLs: {result.stderr}")
                    return False

                # Save URLs to file
                urls_file = os.path.join(image_dir, "download_urls.txt")
                with open(urls_file, 'w') as f:
                    f.write(result.stdout)

                urls_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                print(f"✅ Found {urls_count} image URLs")

                if urls_count == 0:
                    print("❌ No images found matching criteria")
                    return False

                print("🚀 Phase 2: Downloading images with aria2c...")

                # Phase 2: Download with aria2c
                aria_cmd = [
                    'aria2c',
                    '--console-log-level=error',
                    '--summary-interval=10',
                    '--continue=true',
                    '--max-connection-per-server=16',
                    '--min-split-size=1M',
                    '--split=16',
                    '--input-file=' + urls_file,
                    '--dir=' + image_dir
                ]

                result = subprocess.run(aria_cmd, capture_output=True, text=True, timeout=1800)

                # Clean up URLs file
                os.remove(urls_file)

                if result.returncode == 0:
                    print("✅ Download completed successfully with aria2c")
                    success = True
                else:
                    print(f"⚠️ Aria2c had issues: {result.stderr}")
                    print("🔄 Falling back to direct gallery-dl download...")

            except subprocess.TimeoutExpired:
                print("⚠️ URL fetching timed out, trying direct download...")
            except Exception as e:
                print(f"⚠️ Error with aria2c method: {e}")
                print("🔄 Falling back to direct gallery-dl download...")

        # Fallback or primary method: Direct gallery-dl download
        if not success:
            print("📥 Downloading images directly with gallery-dl...")

            download_config = {
                **base_config,
                "directory": image_dir,
                "write-tags": write_tags
            }

            download_args = build_args(download_config)

            # Add additional arguments
            if additional_args:
                download_args.extend(additional_args.split())

            try:
                cmd = ['gallery-dl'] + download_args + [url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    print("✅ Download completed successfully")
                    success = True
                else:
                    print(f"❌ Gallery-dl error: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                print("❌ Download timed out after 30 minutes")
                return False
            except Exception as e:
                print(f"❌ Error during download: {e}")
                return False

        # Post-process tags if enabled
        if write_tags and success:
            print("🏷️ Post-processing tag files...")
            self._process_gallery_dl_tags(image_dir)

        # Count results
        image_count = len([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))])
        tag_count = len([f for f in os.listdir(image_dir) if f.endswith('.txt')])

        print("\n" + "=" * 60)
        print("🎉 Scraping complete!")
        print(f"📁 Location: {image_dir}")
        print(f"🖼️ Images downloaded: {image_count}")
        print(f"🏷️ Tag files: {tag_count}")

        return success

    def _process_gallery_dl_tags(self, directory):
        """Process and clean up tag files from gallery-dl"""
        import html

        def process_dir(dir_path):
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)

                if os.path.isfile(item_path) and item.endswith(".txt"):
                    try:
                        # Handle double extensions (e.g., image.jpg.txt -> image.txt)
                        name_parts = item.split('.')
                        if len(name_parts) > 2:  # Has double extension
                            base_name = '.'.join(name_parts[:-2])  # Remove last 2 extensions
                            new_name = f"{base_name}.txt"
                            new_path = os.path.join(dir_path, new_name)

                            if item_path != new_path:
                                os.rename(item_path, new_path)
                                item_path = new_path

                        # Clean up tag content
                        with open(item_path, 'r', encoding='utf-8') as f:
                            contents = f.read()

                        # Process tags
                        contents = html.unescape(contents)  # Decode HTML entities
                        contents = contents.replace("_", " ")  # Convert underscores to spaces

                        # Handle different tag formats
                        if '\n' in contents:
                            # Line-separated tags -> comma-separated
                            tags = [tag.strip() for tag in contents.split('\n') if tag.strip()]
                            contents = ", ".join(tags)
                        elif ' ' in contents and ',' not in contents:
                            # Space-separated -> comma-separated
                            tags = [tag.strip() for tag in contents.split() if tag.strip()]
                            contents = ", ".join(tags)

                        # Write back cleaned content
                        with open(item_path, 'w', encoding='utf-8') as f:
                            f.write(contents)

                    except Exception as e:
                        print(f"⚠️ Error processing tag file {item}: {e}")

                elif os.path.isdir(item_path):
                    # Recursively process subdirectories
                    process_dir(item_path)

        process_dir(directory)
        print("✅ Tag files processed and cleaned")

    def upload_dataset_to_huggingface(self, dataset_path, dataset_name, hf_token="",
                                    orgs_name="", make_private=False, description=""):
        """
        Upload dataset to HuggingFace Hub
        
        Args:
            dataset_path: Local path to dataset directory
            dataset_name: Name for the dataset repository
            hf_token: HuggingFace write token (get from https://huggingface.co/settings/tokens)
            orgs_name: Organization name (optional, uses personal account if empty)
            make_private: Whether to create private repository
            description: Description for the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from huggingface_hub import HfApi, create_repo, login
            from huggingface_hub.utils import HfHubHTTPError, validate_repo_id
        except ImportError:
            print("❌ HuggingFace Hub not installed. Install with: pip install huggingface_hub")
            return False

        if not hf_token:
            error("HuggingFace token required!")
            info("Get your WRITE token here: https://huggingface.co/settings/tokens")
            return False

        if not dataset_name.strip():
            error("Dataset name is required!")
            return False

        if not os.path.exists(dataset_path):
            error(f"Dataset path not found: {dataset_path}")
            return False

        print_header("HuggingFace Dataset Upload")
        info(f"Dataset path: {dataset_path}")
        info(f"Dataset name: {dataset_name}")

        try:
            # Authenticate with HuggingFace
            progress("Authenticating with HuggingFace Hub...")
            login(hf_token, add_to_git_credential=True)
            api = HfApi()

            # Get user info
            user_info = api.whoami(hf_token)
            username = user_info["name"]
            success(f"Authenticated as: {username}")

            # Build repository ID
            if orgs_name.strip():
                repo_id = f"{orgs_name.strip()}/{dataset_name.strip()}"
                print(f"🏢 Using organization: {orgs_name}")
            else:
                repo_id = f"{username}/{dataset_name.strip()}"
                print(f"👤 Using personal account: {username}")

            # Validate repository ID
            try:
                validate_repo_id(repo_id)
            except Exception as e:
                print(f"❌ Invalid repository ID '{repo_id}': {e}")
                return False

            print(f"📦 Repository ID: {repo_id}")

            # Create repository (if it doesn't exist)
            try:
                print("🔨 Creating repository...")
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=make_private,
                    token=hf_token,
                    exist_ok=True  # Don't error if repo already exists
                )
                print(f"✅ Repository created/verified: {repo_id}")

            except HfHubHTTPError as e:
                if "already exists" in str(e).lower():
                    print(f"📝 Repository '{repo_id}' already exists, continuing...")
                else:
                    print(f"❌ Error creating repository: {e}")
                    return False

            # Create README if description provided
            if description.strip():
                readme_content = f"""---
license: mit
task_categories:
- image-classification
- text-to-image
tags:
- stable-diffusion
- lora
- dataset
- training
pretty_name: "{dataset_name}"
---

# {dataset_name}

{description.strip()}

## Dataset Structure

This dataset contains images and their corresponding caption/tag files for training LoRA models.

## Usage

This dataset is designed for use with LoRA training scripts like:
- Kohya SS sd-scripts
- LoRA Easy Training scripts

## Attribution

Dataset created using [LoRA Easy Training](https://github.com/Linaqruf/kohya-trainer) tools.
Special thanks to [Linaqruf](https://github.com/Linaqruf) for their contributions to the community.
"""

                try:
                    # Upload README
                    api.upload_file(
                        path_or_fileobj=readme_content.encode('utf-8'),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token
                    )
                    print("✅ README.md created")
                except Exception as e:
                    print(f"⚠️ Warning: Could not create README.md: {e}")

            # Upload dataset files
            print("📤 Uploading dataset files...")

            # Check dataset structure
            image_count = 0
            caption_count = 0

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        image_count += 1
                    elif file.lower().endswith('.txt'):
                        caption_count += 1

            print(f"📊 Dataset contains: {image_count} images, {caption_count} caption files")

            # Upload the entire dataset directory
            try:
                api.upload_folder(
                    folder_path=dataset_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message=f"Upload {dataset_name} dataset with {image_count} images"
                )

                success("Dataset uploaded successfully!")
                print(f"🔗 View your dataset at: https://huggingface.co/datasets/{repo_id}")

                # Provide sharing information
                print_header("Dataset Info")
                info(f"Repository: {repo_id}")
                info(f"Type: {'Private' if make_private else 'Public'}")
                info(f"Images: {image_count}")
                info(f"Captions: {caption_count}")
                print(f"🔗 URL: https://huggingface.co/datasets/{repo_id}")

                return True

            except Exception as e:
                print(f"❌ Error uploading files: {e}")
                return False

        except Exception as e:
            print(f"❌ Authentication or API error: {e}")
            print("💡 Make sure your token has WRITE permissions")
            print("🔑 Get your token here: https://huggingface.co/settings/tokens")
            return False

    def scrape_with_gallery_dl(self, site="gelbooru", tags="", dataset_dir="", limit_range="1-200",
                               write_tags=True, use_aria2c=True, custom_url="", sub_folder="",
                               additional_args="--filename /O --no-part"):
        """
        Advanced image scraping using gallery-dl (supports 300+ sites)
        
        Args:
            site: Site to scrape from (gelbooru, danbooru, safebooru, pixiv, twitter, etc.)
            tags: Tags to search for (comma or space separated)
            dataset_dir: Directory to save images
            limit_range: Range of images to download (e.g., "1-200", "1-50")
            write_tags: Whether to download and process tag files
            use_aria2c: Use aria2c for faster parallel downloads
            custom_url: Custom URL instead of using predefined site
            sub_folder: Organize images into subfolder
            additional_args: Additional gallery-dl arguments
        """
        import subprocess
        from urllib.parse import quote_plus

        # Check if gallery-dl is installed
        try:
            result = subprocess.run(['gallery-dl', '--version'], capture_output=True, text=True, timeout=5)
            print(f"🎨 Using gallery-dl version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            print("❌ gallery-dl not found. Install with: pip install gallery-dl")
            return False

        if not tags and not custom_url:
            print("❌ Please specify either tags or a custom URL")
            return False

        # Setup directories
        if dataset_dir:
            base_dir = os.path.join(self.project_root, dataset_dir)
        else:
            base_dir = os.path.join(self.project_root, "dataset")

        if sub_folder:
            if sub_folder.startswith(os.path.sep):
                image_dir = sub_folder  # Absolute path
            else:
                image_dir = os.path.join(base_dir, sub_folder)
        else:
            image_dir = base_dir

        os.makedirs(image_dir, exist_ok=True)

        print("🎨 Gallery-dl scraping setup:")
        print(f"📁 Target directory: {image_dir}")
        print(f"🏷️ Tags: {tags}")
        print(f"📊 Range: {limit_range}")
        print(f"🏃‍♂️ Aria2c: {'Enabled' if use_aria2c else 'Disabled'}")
        print("=" * 60)

        # Prepare tags for URL
        if tags:
            tag_list = [tag.strip() for tag in tags.replace(',', ' ').split() if tag.strip()]
            # URL encode special characters
            encoded_tags = '+'.join(quote_plus(tag.replace(' ', '_'), safe='') for tag in tag_list)
            print(f"🔍 Processed tags: {' + '.join(tag_list)}")

        # Build URL based on site
        if custom_url:
            url = custom_url
            print(f"🌐 Using custom URL: {url}")
        else:
            site_urls = {
                "gelbooru": f"https://gelbooru.com/index.php?page=post&s=list&tags={encoded_tags}",
                "danbooru": f"https://danbooru.donmai.us/posts?tags={encoded_tags}",
                "safebooru": f"https://safebooru.org/index.php?page=post&s=list&tags={encoded_tags}",
                "konachan": f"https://konachan.com/post?tags={encoded_tags}",
                "yande.re": f"https://yande.re/post?tags={encoded_tags}",
                "pixiv": f"https://www.pixiv.net/en/tags/{encoded_tags}/artworks",
            }

            if site.lower() not in site_urls:
                print(f"❌ Unsupported site: {site}")
                print(f"📝 Supported sites: {', '.join(site_urls.keys())}")
                print("💡 Or use custom_url parameter for other sites")
                return False

            url = site_urls[site.lower()]
            print(f"🌐 Built URL: {url}")

        # Common gallery-dl config
        base_config = {
            "user-agent": "gallery-dl/1.26.0",
            "sleep": "1",  # Be respectful with rate limiting
        }

        if limit_range:
            base_config["range"] = limit_range

        # Build gallery-dl arguments
        def build_args(config_dict):
            args = []
            for key, value in config_dict.items():
                if value is True:
                    args.append(f"--{key}")
                elif value is False:
                    continue
                else:
                    args.append(f"--{key}={value}")
            return args

        success = False

        if use_aria2c:
            print("🚀 Phase 1: Getting download URLs with gallery-dl...")

            # Phase 1: Get URLs
            url_config = {**base_config, "get-urls": True}
            url_args = build_args(url_config)

            # Add additional arguments
            if additional_args:
                url_args.extend(additional_args.split())

            try:
                # Run gallery-dl to get URLs
                cmd = ['gallery-dl'] + url_args + [url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    print(f"❌ Error getting URLs: {result.stderr}")
                    return False

                # Save URLs to file
                urls_file = os.path.join(image_dir, "download_urls.txt")
                with open(urls_file, 'w') as f:
                    f.write(result.stdout)

                urls_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                print(f"✅ Found {urls_count} image URLs")

                if urls_count == 0:
                    print("❌ No images found matching criteria")
                    return False

                print("🚀 Phase 2: Downloading images with aria2c...")

                # Phase 2: Download with aria2c
                aria_cmd = [
                    'aria2c',
                    '--console-log-level=error',
                    '--summary-interval=10',
                    '--continue=true',
                    '--max-connection-per-server=16',
                    '--min-split-size=1M',
                    '--split=16',
                    '--input-file=' + urls_file,
                    '--dir=' + image_dir
                ]

                result = subprocess.run(aria_cmd, capture_output=True, text=True, timeout=1800)

                # Clean up URLs file
                os.remove(urls_file)

                if result.returncode == 0:
                    print("✅ Download completed successfully with aria2c")
                    success = True
                else:
                    print(f"⚠️ Aria2c had issues: {result.stderr}")
                    print("🔄 Falling back to direct gallery-dl download...")

            except subprocess.TimeoutExpired:
                print("⚠️ URL fetching timed out, trying direct download...")
            except Exception as e:
                print(f"⚠️ Error with aria2c method: {e}")
                print("🔄 Falling back to direct gallery-dl download...")

        # Fallback or primary method: Direct gallery-dl download
        if not success:
            print("📥 Downloading images directly with gallery-dl...")

            download_config = {
                **base_config,
                "directory": image_dir,
                "write-tags": write_tags
            }

            download_args = build_args(download_config)

            # Add additional arguments
            if additional_args:
                download_args.extend(additional_args.split())

            try:
                cmd = ['gallery-dl'] + download_args + [url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    print("✅ Download completed successfully")
                    success = True
                else:
                    print(f"❌ Gallery-dl error: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                print("❌ Download timed out after 30 minutes")
                return False
            except Exception as e:
                print(f"❌ Error during download: {e}")
                return False

        # Post-process tags if enabled
        if write_tags and success:
            print("🏷️ Post-processing tag files...")
            self._process_gallery_dl_tags(image_dir)

        # Count results
        image_count = len([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))])
        tag_count = len([f for f in os.listdir(image_dir) if f.endswith('.txt')])

        print("\n" + "=" * 60)
        print("🎉 Scraping complete!")
        print(f"📁 Location: {image_dir}")
        print(f"🖼️ Images downloaded: {image_count}")
        print(f"🏷️ Tag files: {tag_count}")

        return success

    def convert_image_formats(self, dataset_dir, target_format, quality=95):
        """
        Convert image formats in a dataset directory without resizing.
        
        Args:
            dataset_dir (str): The path to the dataset directory.
            target_format (str): The target format ('jpg', 'png', 'webp').
            quality (int): The quality for lossy formats (85-100).
        """
        import os
        from PIL import Image

        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found at {dataset_path}")
            return False

        print(f"🔄 Converting images in {dataset_path} to {target_format.upper()}...")

        converted_count = 0
        skipped_count = 0

        source_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        target_ext = f'.{target_format}'

        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_ext = os.path.splitext(file.lower())[1]
                if file_ext in source_extensions and file_ext != target_ext:
                    image_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    new_path = os.path.join(root, f"{base_name}{target_ext}")
                    
                    try:
                        with Image.open(image_path) as img:
                            # Convert mode if needed
                            if target_format == 'jpg' and img.mode in ('RGBA', 'LA', 'P'):
                                # Convert to RGB for JPEG (no transparency)
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                                img = background
                            elif target_format == 'png' and img.mode not in ('RGBA', 'LA', 'P'):
                                # Keep original mode for PNG
                                pass
                            
                            # Save in new format
                            save_kwargs = {}
                            if target_format in ['jpg', 'webp']:
                                save_kwargs['quality'] = quality
                            elif target_format == 'png':
                                save_kwargs['optimize'] = True
                            
                            img.save(new_path, target_format.upper(), **save_kwargs)
                            
                            # Remove original file after successful conversion
                            os.remove(image_path)
                            converted_count += 1
                            
                    except Exception as e:
                        print(f"⚠️ Could not convert {file}: {e}")
                        skipped_count += 1

        print(f"✅ Format conversion complete. Converted {converted_count} images, skipped {skipped_count}.")
        return True

    def _process_gallery_dl_tags(self, directory):
        """Process and clean up tag files from gallery-dl"""
        import html

        def process_dir(dir_path):
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)

                if os.path.isfile(item_path) and item.endswith(".txt"):
                    try:
                        # Handle double extensions (e.g., image.jpg.txt -> image.txt)
                        name_parts = item.split('.')
                        if len(name_parts) > 2:  # Has double extension
                            base_name = '.'.join(name_parts[:-2])  # Remove last 2 extensions
                            new_name = f"{base_name}.txt"
                            new_path = os.path.join(dir_path, new_name)

                            if item_path != new_path:
                                os.rename(item_path, new_path)
                                item_path = new_path

                        # Clean up tag content
                        with open(item_path, 'r', encoding='utf-8') as f:
                            contents = f.read()

                        # Process tags
                        contents = html.unescape(contents)  # Decode HTML entities
                        contents = contents.replace("_", " ")  # Convert underscores to spaces

                        # Handle different tag formats
                        if '\n' in contents:
                            # Line-separated tags -> comma-separated
                            tags = [tag.strip() for tag in contents.split('\n') if tag.strip()]
                            contents = ", ".join(tags)
                        elif ' ' in contents and ',' not in contents:
                            # Space-separated -> comma-separated
                            tags = [tag.strip() for tag in contents.split() if tag.strip()]
                            contents = ", ".join(tags)

                        # Write back cleaned content
                        with open(item_path, 'w', encoding='utf-8') as f:
                            f.write(contents)

                    except Exception as e:
                        print(f"⚠️ Error processing tag file {item}: {e}")

                elif os.path.isdir(item_path):
                    # Recursively process subdirectories
                    process_dir(item_path)

        process_dir(directory)
        print("✅ Tag files processed and cleaned")

    def upload_dataset_to_huggingface(self, dataset_path, dataset_name, hf_token="",
                                    orgs_name="", make_private=False, description=""):
        pass  # TODO: Implement HuggingFace dataset upload
