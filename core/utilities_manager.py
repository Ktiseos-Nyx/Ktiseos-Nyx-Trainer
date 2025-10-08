# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/utilities_manager.py
import gc
import glob
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, login

from .managers import get_subprocess_environment


class UtilitiesManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        
        # File types for enhanced uploader
        self.file_types = [
            # AI Model Files 🤖
            ('SafeTensors', 'safetensors'), ('PyTorch Models', 'pt'), ('PyTorch Legacy', 'pth'),
            ('ONNX Models', 'onnx'), ('TensorFlow Models', 'pb'), ('Keras Models', 'h5'),
            # Checkpoint Files 🎯
            ('Checkpoints', 'ckpt'), ('Binary Files', 'bin'),
            # Config & Data Files 📝
            ('JSON Files', 'json'), ('YAML Files', 'yaml'), ('YAML Alt', 'yml'),
            ('Text Files', 'txt'), ('CSV Files', 'csv'), ('Pickle Files', 'pkl'),
            # Image Files 🎨
            ('PNG Images', 'png'), ('JPEG Images', 'jpg'), ('JPEG Alt', 'jpeg'),
            ('WebP Images', 'webp'), ('GIF Images', 'gif'),
            # Archive Files 📦
            ('ZIP Archives', 'zip'), ('TAR Files', 'tar'), ('GZ Archives', 'gz')
        ]
        
        # Enable hf_transfer for faster uploads if available
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    def get_files_in_directory(self, directory_path, file_extension, sort_by='name'):
        """Get list of files in directory with specified extension"""
        if not os.path.isdir(directory_path):
            return []
        
        try:
            glob_pattern = f"*.{file_extension}"
            found_paths = list(Path(directory_path).glob(glob_pattern))
            
            valid_files_info = []
            for p in found_paths:
                if p.is_symlink() or not p.is_file():
                    continue
                
                if sort_by == 'date':
                    sort_key = p.stat().st_mtime
                else:  # 'name'
                    sort_key = p.name.lower()
                valid_files_info.append((str(p), sort_key))

            # Sort based on the pre-calculated sort_key
            if sort_by == 'date':
                valid_files_info.sort(key=lambda item: item[1], reverse=True)  # newest first
            else:  # 'name'
                valid_files_info.sort(key=lambda item: item[1])
            
            return [item[0] for item in valid_files_info]
        
        except Exception as e:
            print(f"❌ Error listing files: {type(e).__name__} - {str(e)}")
            return []

    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 0:
            return "Invalid size"
        if size_bytes == 0:
            return "0 B"
        
        units = ("B", "KB", "MB", "GB", "TB")
        i = math.floor(math.log(size_bytes, 1024)) if size_bytes > 0 else 0
        if i >= len(units):
            i = len(units) - 1
        
        s = round(size_bytes / (1024 ** i), 2)
        return f"{s} {units[i]}"

    def check_hf_transfer_availability(self):
        """Check if hf_transfer is available for faster uploads"""
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            try:
                import hf_transfer
                return True
            except ImportError:
                return False
        return False

    def upload_multiple_files_to_huggingface(self, hf_token, owner, repo_name, repo_type, 
                                           selected_files, remote_folder="", commit_message="", 
                                           create_pr=False, progress_callback=None):
        """Enhanced multi-file upload to HuggingFace with progress tracking"""
        
        if not hf_token or not hf_token.strip():
            return {"success": False, "error": "HuggingFace token is required"}
        
        if not owner or not repo_name:
            return {"success": False, "error": "Owner and repository name are required"}
        
        if not selected_files:
            return {"success": False, "error": "No files selected for upload"}

        try:
            login(token=hf_token)
            api = HfApi()
            
            repo_id = f"{owner}/{repo_name}"
            total_files = len(selected_files)
            success_count = 0
            
            hf_transfer_active = self.check_hf_transfer_availability()
            
            results = {
                "success": True,
                "repo_id": repo_id,
                "total_files": total_files,
                "uploaded_files": [],
                "failed_files": [],
                "hf_transfer_active": hf_transfer_active
            }
            
            for idx, file_path_str in enumerate(selected_files, 1):
                current_file_path = Path(file_path_str)
                
                if progress_callback:
                    progress_callback(idx, total_files, current_file_path.name)
                
                if not current_file_path.exists():
                    error_msg = f"File not found: {current_file_path.name}"
                    results["failed_files"].append({"file": current_file_path.name, "error": error_msg})
                    continue

                try:
                    # Calculate path in repo
                    path_in_repo = current_file_path.name
                    if remote_folder:
                        clean_folder = remote_folder.strip('/')
                        path_in_repo = f"{clean_folder}/{current_file_path.name}"
                    
                    file_commit_msg = commit_message or f"Upload {current_file_path.name}"
                    if total_files > 1:
                        file_commit_msg += f" (file {idx}/{total_files})"
                    
                    start_time = time.time()
                    
                    response_url = api.upload_file(
                        path_or_fileobj=str(current_file_path),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        create_pr=create_pr,
                        commit_message=file_commit_msg,
                    )
                    
                    duration = time.time() - start_time
                    file_size = current_file_path.stat().st_size
                    
                    results["uploaded_files"].append({
                        "file": current_file_path.name,
                        "path_in_repo": path_in_repo,
                        "size": self.format_file_size(file_size),
                        "duration": f"{duration:.1f}s",
                        "url": response_url
                    })
                    success_count += 1
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    results["failed_files"].append({"file": current_file_path.name, "error": error_msg})
            
            results["success_count"] = success_count
            return results

        except Exception as e:
            return {"success": False, "error": f"Upload failed: {type(e).__name__}: {str(e)}"}
        finally:
            # Force garbage collection after upload operations
            gc.collect()

    def upload_to_huggingface(self, hf_token, model_path, repo_name):
        if not hf_token or not hf_token.strip():
            print("❌ Error: Hugging Face token is required.")
            print("💡 Get your token from: https://huggingface.co/settings/tokens")
            return False
        if not model_path or not model_path.strip():
            print("❌ Error: Model file path is required.")
            return False
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at {model_path}")
            return False
        if not repo_name or not repo_name.strip():
            print("❌ Error: Repository name is required.")
            print("💡 Example: 'my-awesome-loras' (no spaces, lowercase preferred)")
            return False

        # Validate file extension
        if not model_path.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
            print("⚠️ Warning: File doesn't appear to be a model file (.safetensors, .ckpt, .pt, .pth)")
            print("🤔 Continuing anyway...")

        try:
            login(token=hf_token)
            api = HfApi()

            filename = os.path.basename(model_path)
            repo_id = f"{api.whoami()['name']}/{repo_name}"

            print(f"🚀 Uploading {filename} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=repo_id,
                commit_message=f"Upload {filename}",
                repo_type="model"
            )
            print("✅ Upload complete!")
            print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
            print(f"📁 Direct file link: https://huggingface.co/{repo_id}/blob/main/{filename}")
            return True

        except Exception as e:
            print(f"❌ Error during Hugging Face upload: {e}")
            if "Invalid token" in str(e):
                print("💡 Check your HuggingFace token: https://huggingface.co/settings/tokens")
            elif "Repository not found" in str(e):
                print("💡 Repository will be created automatically on first upload")
            return False
        finally:
            # Force garbage collection after HuggingFace operations
            gc.collect()

    def resize_lora(self, input_path, output_path, new_dim, new_alpha):
        if not os.path.exists(input_path):
            print(f"Error: Input LoRA file not found at {input_path}")
            return False
        if not output_path:
            print("Error: Please specify an output path for the resized LoRA.")
            return False
        if not new_dim or not new_alpha:
            print("Error: Please specify new dim and alpha values.")
            return False

        # Always use current Python executable for environment-agnostic execution
        # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
        venv_python = sys.executable

        # Try Derrian's enhanced resize script first, fallback to Kohya's
        derrian_resize_script = os.path.join(self.trainer_dir, "derrian_backend", "utils", "resize_lora.py")
        kohya_resize_script = os.path.join(self.sd_scripts_dir, "networks/resize_lora.py")

        if os.path.exists(derrian_resize_script):
            resize_script = derrian_resize_script
            print("🔧 Using Derrian's enhanced resize script with dynamic ranking")
        elif os.path.exists(kohya_resize_script):
            resize_script = kohya_resize_script
            print("🔧 Using Kohya's standard resize script")
        else:
            print("❌ Error: No LoRA resize script found")
            print(f"   Looked for: {derrian_resize_script}")
            print(f"   Looked for: {kohya_resize_script}")
            print("💡 Please ensure the trainer environment setup completed successfully.")
            return False

        command = [
            venv_python, resize_script,
            "--save_precision", "fp16",
            "--save_to", output_path,
            "--model", input_path,
            "--new_rank", str(new_dim),
            "--new_alpha", str(new_alpha)
        ]

        print(f"Resizing LoRA from {input_path} to {output_path} with dim={new_dim}, alpha={new_alpha}...")

        # Get standardized subprocess environment (fixes CAME import issues!)
        env = get_subprocess_environment(self.project_root)

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.trainer_dir,  # Run from trainer directory like training scripts
                env=env  # Pass the proper environment with PYTHONPATH
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')

            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            print("\nLoRA resizing complete.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while resizing LoRA: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False
        finally:
            # Force garbage collection after LoRA processing
            gc.collect()

    def count_dataset_files(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found at {dataset_path}")
            return False

        image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        caption_extensions = (".txt", ".caption")

        image_count = 0
        caption_count = 0
        other_files = 0

        print(f"Counting files in {dataset_path}...")
        try:
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
                    elif file.lower().endswith(caption_extensions):
                        caption_count += 1
                    else:
                        other_files += 1

            print(f"\nResults for {dataset_path}:")
            print(f"  Images: {image_count}")
            print(f"  Captions: {caption_count}")
            print(f"  Other files: {other_files}")
            return True

        except Exception as e:
            print(f"Error counting files: {e}")
            return False

    def optimize_dataset_images(self, dataset_path, target_format="webp", max_file_size_mb=2, quality=95, max_dimension=None):
        """
        Optimize all images in a dataset directory
        
        Args:
            dataset_path (str): Path to dataset directory
            target_format (str): Target format - "webp" or "jpeg"
            max_file_size_mb (float): Maximum file size in MB before resizing
            quality (int): Quality setting (1-100)
            max_dimension (int): Maximum width/height in pixels (None = no limit)
        """
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset path not found at {dataset_path}")
            return False

        # Import PIL here since it's only needed for this function
        try:
            from PIL import Image, ImageFile

            # Enable loading of truncated images
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        except ImportError:
            print("❌ Error: PIL (Pillow) not available. Install with: pip install Pillow")
            return False

        # Supported input formats
        input_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        target_format = target_format.lower()

        if target_format not in ['webp', 'jpeg']:
            print("❌ Error: target_format must be 'webp' or 'jpeg'")
            return False

        target_ext = '.webp' if target_format == 'webp' else '.jpg'
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        print(f"🖼️ Optimizing images in: {dataset_path}")
        print(f"📊 Target format: {target_format.upper()}")
        print(f"📏 Max file size: {max_file_size_mb} MB")
        print(f"⚙️ Quality: {quality}")
        if max_dimension:
            print(f"📐 Max dimension: {max_dimension}px")
        print("="*60)

        processed_count = 0
        converted_count = 0
        resized_count = 0
        error_count = 0
        total_size_before = 0
        total_size_after = 0

        try:
            # Get all image files
            image_files = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in input_extensions:
                        image_files.append(os.path.join(root, file))

            if not image_files:
                print("⚠️ No image files found to optimize")
                return True

            print(f"📁 Found {len(image_files)} images to process")
            print()

            for i, image_path in enumerate(image_files, 1):
                try:
                    file_name = os.path.basename(image_path)
                    file_size_before = os.path.getsize(image_path)
                    total_size_before += file_size_before

                    # Load image
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary (for JPEG/WebP compatibility)
                        if img.mode in ('RGBA', 'LA', 'P') and target_format == 'jpeg':
                            # Create white background for JPEG
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = rgb_img
                        elif img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGB')

                        original_size = img.size
                        needs_resize = False

                        # Check if we need to resize based on file size or dimensions
                        if file_size_before > max_file_size_bytes:
                            needs_resize = True

                        if max_dimension and (img.width > max_dimension or img.height > max_dimension):
                            needs_resize = True

                        # Resize if needed
                        if needs_resize:
                            if max_dimension:
                                # Resize to fit within max_dimension while maintaining aspect ratio
                                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                            else:
                                # Resize based on file size (reduce by percentage)
                                scale_factor = 0.8  # Reduce by 20%
                                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            resized_count += 1

                        # Determine output path
                        file_dir = os.path.dirname(image_path)
                        file_basename = os.path.splitext(file_name)[0]
                        output_path = os.path.join(file_dir, f"{file_basename}{target_ext}")

                        # Save in target format
                        save_kwargs = {'quality': quality, 'optimize': True}
                        if target_format == 'webp':
                            save_kwargs['method'] = 6  # Best compression

                        img.save(output_path, format=target_format.upper(), **save_kwargs)

                        # Get new file size
                        file_size_after = os.path.getsize(output_path)
                        total_size_after += file_size_after

                        # Remove original if format changed
                        format_changed = not image_path.lower().endswith(target_ext)
                        if format_changed:
                            os.remove(image_path)
                            converted_count += 1

                        processed_count += 1

                        # Progress update
                        size_reduction = ((file_size_before - file_size_after) / file_size_before * 100) if file_size_before > 0 else 0

                        status_parts = []
                        if format_changed:
                            status_parts.append("converted")
                        if needs_resize:
                            status_parts.append(f"resized {original_size[0]}x{original_size[1]}→{img.size[0]}x{img.size[1]}")
                        if size_reduction > 0:
                            status_parts.append(f"{size_reduction:.1f}% smaller")

                        status = f"({', '.join(status_parts)})" if status_parts else ""

                        print(f"  ✅ [{i:3d}/{len(image_files)}] {file_name} {status}")

                        # Show progress every 25 files
                        if i % 25 == 0:
                            print(f"     📊 Progress: {i}/{len(image_files)} processed...")

                except Exception as e:
                    print(f"  ❌ Error processing {file_name}: {e}")
                    error_count += 1
                    continue

            # Final summary
            total_size_reduction = total_size_before - total_size_after
            size_reduction_percent = (total_size_reduction / total_size_before * 100) if total_size_before > 0 else 0

            print("\n" + "="*60)
            print("🎉 Dataset optimization complete!")
            print(f"📊 Processed: {processed_count}/{len(image_files)} images")
            if converted_count > 0:
                print(f"🔄 Converted: {converted_count} images to {target_format.upper()}")
            if resized_count > 0:
                print(f"📏 Resized: {resized_count} images")
            if error_count > 0:
                print(f"❌ Errors: {error_count} images failed")

            print(f"💾 Size reduction: {total_size_reduction / (1024*1024):.1f} MB ({size_reduction_percent:.1f}%)")
            print(f"📁 Before: {total_size_before / (1024*1024):.1f} MB")
            print(f"📁 After: {total_size_after / (1024*1024):.1f} MB")

            return error_count == 0

        except Exception as e:
            print(f"❌ Unexpected error during optimization: {e}")
            return False
