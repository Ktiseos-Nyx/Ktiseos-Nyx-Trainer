import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile
from tqdm import tqdm

# Fix for truncated JPEG images - allows PIL to handle corrupted/truncated files gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
import glob
import warnings

# Suppress diffusers FutureWarning about deprecated PyTorch function
# This is a known issue with diffusers 0.25.0 + newer PyTorch versions
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

# Simple logging setup (replaces library.utils.setup_logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_safe_gpu_memory_limit():
    """
    Dynamically calculate safe GPU memory limit based on available VRAM.
    Returns appropriate memory limit for cross-platform compatibility.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get total VRAM and leave 20% buffer for other processes
            total_vram = torch.cuda.get_device_properties(0).total_memory
            safe_limit = int(total_vram * 0.8)  # Use 80% of available VRAM
            logger.info(f"🔍 Detected {total_vram/1024**3:.1f}GB VRAM, setting ONNX limit to {safe_limit/1024**3:.1f}GB")
            return safe_limit
        else:
            logger.info("🔍 No CUDA available, skipping GPU memory limit")
            return None  # No GPU, no limit needed
    except Exception as e:
        logger.warning(f"GPU memory detection failed: {e}")
        return 2 * 1024 * 1024 * 1024  # Fallback to 2GB for safety

def robust_download_fallback(repo_id, filename, local_path):
    """
    Robust download fallback system: hf_hub_download → aria2c → wget → Python requests
    """
    logger.info(f"Attempting robust download fallback for {filename}")
    
    # Construct HuggingFace URL
    hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    # Method 1: Try aria2c
    if shutil.which("aria2c"):
        logger.info("🚀 Attempting download with aria2c...")
        try:
            command = [
                "aria2c", hf_url,
                "--console-log-level=warn",
                "-c", "-s", "16", "-x", "16", "-k", "10M",
                "-d", os.path.dirname(local_path),
                "-o", os.path.basename(local_path)
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(local_path):
                logger.info(f"✅ Download complete with aria2c: {local_path}")
                return local_path
            else:
                logger.warning("❌ aria2c failed. Trying wget...")
        except Exception as e:
            logger.warning(f"❌ Error with aria2c: {e}. Trying wget...")
    else:
        logger.info("⚠️ aria2c not available. Trying wget...")

    # Method 2: Try wget
    if shutil.which("wget"):
        logger.info("🚀 Attempting download with wget...")
        try:
            command = ["wget", "-O", local_path, hf_url]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(local_path):
                logger.info(f"✅ Download complete with wget: {local_path}")
                return local_path
            else:
                logger.warning("❌ wget failed. Trying Python requests...")
        except Exception as e:
            logger.warning(f"❌ Error with wget: {e}. Trying Python requests...")
    else:
        logger.info("⚠️ wget not available. Trying Python requests...")

    # Method 3: Python requests fallback
    logger.info("🚀 Attempting download with Python requests...")
    try:
        import requests
        
        response = requests.get(hf_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
        
        if os.path.exists(local_path):
            logger.info(f"✅ Download complete with Python requests: {local_path}")
            return local_path
        else:
            logger.error("❌ Python requests download failed")
            
    except Exception as e:
        logger.error(f"❌ Error with Python requests: {e}")

    logger.error("💥 All download methods failed!")
    return None

# from wd14 tagger
IMAGE_SIZE = 448

# Enhanced WD14 Tagger with Kohya robustness improvements + v3 model support
# Supported models:
# v1.4: wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2 / wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
# v3: wd-eva02-large-tagger-v3 / wd-vit-large-tagger-v3 / wd-swinv2-tagger-v3
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    # Simple resize for inference (448x448 for WD14 models)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            # Remove unnecessary tensor conversion for better performance
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    # model location is model_dir + repo_id
    # repo id may be like "user/repo" or "user/repo/branch", so we need to remove slash
    model_location = os.path.join(args.model_dir, args.repo_id.replace("/", "_"))

    # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
    # depreacatedの警告が出るけどなくなったらその時
    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22
    if not os.path.exists(model_location) or args.force_download:
        # Create both base model directory and full model location directory
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(model_location, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        files = FILES
        if args.onnx:
            files = ["selected_tags.csv"]
            files += FILES_ONNX
        else:
            # Download SUB_DIR_FILES with fallback support
            for file in SUB_DIR_FILES:
                subdir_path = os.path.join(model_location, SUB_DIR)
                os.makedirs(subdir_path, exist_ok=True)
                local_file_path = os.path.join(subdir_path, file)
                
                try:
                    logger.info(f"Attempting HuggingFace Hub download for subfolder file {file}")
                    hf_hub_download(
                        args.repo_id,
                        file,
                        subfolder=SUB_DIR,
                        cache_dir=subdir_path,
                        force_download=True,
                        force_filename=file,
                    )
                    logger.info(f"✅ HF Hub subfolder download successful: {file}")
                except Exception as e:
                    logger.warning(f"❌ HF Hub subfolder download failed for {file}: {e}")
                    logger.info("Attempting robust download fallback for subfolder file...")
                    
                    # Use robust fallback download for subfolder files
                    subfolder_file = f"{SUB_DIR}/{file}" if SUB_DIR else file
                    result_path = robust_download_fallback(args.repo_id, subfolder_file, local_file_path)
                    if not (result_path and os.path.exists(result_path)):
                        logger.error(f"💥 All download methods failed for subfolder file {file}")
                        raise Exception(f"Failed to download subfolder file {file}")
        for file in files:
            local_file_path = os.path.join(model_location, file)
            # Create subdirectories if the file is in a subdirectory
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Try hf_hub_download first, fallback to robust download if it fails
            download_success = False
            try:
                logger.info(f"Attempting HuggingFace Hub download for {file}")
                downloaded_path = hf_hub_download(args.repo_id, file, force_download=args.force_download)
                # Copy to local model directory
                shutil.copy2(downloaded_path, local_file_path)
                logger.info(f"✅ HF Hub download successful: {file}")
                download_success = True
            except Exception as e:
                logger.warning(f"❌ HF Hub download failed for {file}: {e}")
                logger.info("Attempting robust download fallback...")
                
                # Use robust fallback download
                result_path = robust_download_fallback(args.repo_id, file, local_file_path)
                if result_path and os.path.exists(result_path):
                    logger.info(f"✅ Fallback download successful: {file}")
                    download_success = True
                else:
                    logger.error(f"💥 All download methods failed for {file}")
            
            # Verify file was downloaded successfully
            if download_success and os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
                logger.info(f"✅ File verified: {local_file_path} ({os.path.getsize(local_file_path)} bytes)")
            else:
                logger.error(f"❌ Download verification failed for {file}")
                raise Exception(f"Failed to download {file} - all methods exhausted")
    else:
        logger.info("using existing wd14 tagger model")

    # モデルを読み込む
    if args.onnx:
        import onnx
        import onnxruntime as ort

        onnx_path = f"{model_location}/model.onnx"
        logger.info("Running wd14 tagger with onnx")
        logger.info(f"loading onnx model: {onnx_path}")

        if not os.path.exists(onnx_path):
            raise Exception(
                f"onnx model not found: {onnx_path}, please redownload the model with --force_download"
                + " / onnxモデルが見つかりませんでした。--force_downloadで再ダウンロードしてください"
            )

        model = onnx.load(onnx_path)
        input_name = model.graph.input[0].name
        try:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        except Exception:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

        if args.batch_size != batch_size and not isinstance(batch_size, str) and batch_size > 0:
            # some rebatch model may use 'N' as dynamic axes
            logger.warning(
                f"Batch size {args.batch_size} doesn't match onnx model batch size {batch_size}, use model batch size {batch_size}"
            )
            args.batch_size = batch_size

        del model

        if "OpenVINOExecutionProvider" in ort.get_available_providers():
            # requires provider options for gpu support
            # fp16 causes nonsense outputs
            ort_sess = ort.InferenceSession(
                onnx_path,
                providers=(["OpenVINOExecutionProvider"]),
                provider_options=[{'device_type' : "GPU_FP32"}],
            )
        else:
            # Try to create session with GPU first, but gracefully fall back to optimized CPU
            try:
                available_providers = ort.get_available_providers()
                logger.info(f"Available ONNX providers: {available_providers}")
                
                if "CUDAExecutionProvider" in available_providers:
                    logger.info("Attempting CUDA execution provider with enhanced configuration...")
                    
                    # Dynamic CUDA provider options for cross-platform compatibility
                    cuda_provider_options = {'device_id': 0}
                    
                    # Add dynamic GPU memory limit based on available VRAM
                    safe_memory_limit = get_safe_gpu_memory_limit()
                    if safe_memory_limit:
                        cuda_provider_options['gpu_mem_limit'] = safe_memory_limit
                    # Removed aggressive cuDNN options that cause CUDA library conflicts
                    
                    # Basic session options for maximum compatibility
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    # Removed aggressive options that may conflict with different ONNX/CUDA versions
                    
                    # Try creating session with comprehensive error handling
                    try:
                        ort_sess = ort.InferenceSession(
                            onnx_path, 
                            providers=[("CUDAExecutionProvider", cuda_provider_options)],
                            sess_options=sess_options
                        )
                        logger.info("✅ CUDA execution provider initialized successfully")
                    except Exception as cuda_error:
                        logger.warning(f"CUDA provider with options failed: {str(cuda_error)[:100]}...")
                        logger.info("Trying CUDA provider with minimal options...")
                        
                        # Fallback: try minimal CUDA configuration
                        try:
                            minimal_cuda_options = {'device_id': 0}
                            ort_sess = ort.InferenceSession(
                                onnx_path, 
                                providers=[("CUDAExecutionProvider", minimal_cuda_options)],
                                sess_options=ort.SessionOptions()
                            )
                            logger.info("✅ CUDA execution provider with minimal config successful")
                        except Exception as minimal_cuda_error:
                            logger.warning(f"Minimal CUDA config also failed: {str(minimal_cuda_error)[:100]}...")
                            raise Exception("All CUDA configurations failed, falling back to CPU")
                        
                elif "ROCMExecutionProvider" in available_providers:
                    logger.info("Attempting ROCm execution provider") 
                    ort_sess = ort.InferenceSession(onnx_path, providers=["ROCMExecutionProvider"])
                    logger.info("✅ ROCm execution provider initialized successfully")
                else:
                    raise Exception("No GPU providers available, using CPU")
                    
            except Exception as e:
                logger.warning(f"All GPU execution providers failed: {str(e)[:100]}...")
                logger.info("Falling back to optimized CPU execution")
                
                # Enhanced CPU execution with all available optimizations
                cpu_sess_options = ort.SessionOptions()
                cpu_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                cpu_sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                cpu_sess_options.intra_op_num_threads = 0  # Use all available cores
                cpu_sess_options.inter_op_num_threads = 0  # Use all available cores
                
                ort_sess = ort.InferenceSession(
                    onnx_path, 
                    providers=["CPUExecutionProvider"],
                    sess_options=cpu_sess_options
                )
                logger.info("✅ Optimized CPU execution provider initialized successfully")
    else:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model_path = f"{model_location}"
        logger.info("Running wd14 tagger with TensorFlow/Keras")
        logger.info(f"loading keras model: {model_path}")
        
        # Try GPU first, gracefully fall back to CPU if GPU fails
        try:
            # Check if GPU is available and try GPU inference
            if tf.config.list_physical_devices('GPU'):
                logger.info("GPU detected, attempting GPU inference...")
                with tf.device('/GPU:0'):
                    model = load_model(model_path)
                logger.info("✅ TensorFlow GPU model loaded successfully")
            else:
                logger.info("No GPU detected, using CPU inference...")
                with tf.device('/CPU:0'):
                    model = load_model(model_path)
                logger.info("✅ TensorFlow CPU model loaded successfully")
        except Exception as e:
            logger.warning(f"❌ GPU inference failed ({str(e)[:50]}...), falling back to CPU")
            try:
                with tf.device('/CPU:0'):
                    model = load_model(model_path)
                logger.info("✅ TensorFlow CPU fallback successful")
            except Exception as cpu_error:
                logger.error(f"💥 Both GPU and CPU model loading failed: {cpu_error}")
                raise Exception(f"Failed to load TensorFlow model: {cpu_error}")

    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    # 依存ライブラリを増やしたくないので自力で読むよ

    with open(os.path.join(model_location, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]  # tag_id,name,category,count
        rows = line[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
    general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

    # preprocess tags in advance
    if args.character_tag_expand:
        for i, tag in enumerate(character_tags):
            if tag.endswith(")"):
                # chara_name_(series) -> chara_name, series
                # chara_name_(costume)_(series) -> chara_name_(costume), series
                tags = tag.split("(")
                character_tag = "(".join(tags[:-1])
                if character_tag.endswith("_"):
                    character_tag = character_tag[:-1]
                series_tag = tags[-1].replace(")", "")
                character_tags[i] = character_tag + args.caption_separator + series_tag

    if args.remove_underscore:
        rating_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in rating_tags]
        general_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in general_tags]
        character_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in character_tags]

    if args.tag_replacement is not None:
        # escape , and ; in tag_replacement: wd14 tag names may contain , and ;
        escaped_tag_replacements = args.tag_replacement.replace("\\,", "@@@@").replace("\\;", "####")
        tag_replacements = escaped_tag_replacements.split(";")
        for tag_replacement in tag_replacements:
            tags = tag_replacement.split(",")  # source, target
            assert len(tags) == 2, f"tag replacement must be in the format of `source,target` / タグの置換は `置換元,置換先` の形式で指定してください: {args.tag_replacement}"

            source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
            logger.info(f"replacing tag: {source} -> {target}")

            if source in general_tags:
                general_tags[general_tags.index(source)] = target
            elif source in character_tags:
                character_tags[character_tags.index(source)] = target
            elif source in rating_tags:
                rating_tags[rating_tags.index(source)] = target

    # 画像を読み込む

    # Find image files (replaces train_util.glob_images_pathlib)
    train_data_dir_path = Path(args.train_data_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    image_paths = []
    
    if args.recursive:
        for ext in image_extensions:
            image_paths.extend(train_data_dir_path.rglob(f"*{ext}"))
            image_paths.extend(train_data_dir_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in image_extensions:
            image_paths.extend(train_data_dir_path.glob(f"*{ext}"))
            image_paths.extend(train_data_dir_path.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
    logger.info(f"found {len(image_paths)} images.")

    tag_freq = {}

    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = args.undesired_tags.split(stripped_caption_separator)
    undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

    always_first_tags = None
    if args.always_first_tags is not None:
        always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator) if tag.strip() != ""]

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        if args.onnx:
            # Remove unnecessary padding for better batch handling
            # if len(imgs) < args.batch_size:
            #     imgs = np.concatenate([imgs, np.zeros((args.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            probs = ort_sess.run(None, {input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(path_imgs)]
        else:
            # TensorFlow inference with GPU → CPU fallback
            try:
                probs = model(imgs, training=False)
                probs = probs.numpy()
            except Exception as e:
                logger.warning(f"TensorFlow GPU inference failed ({str(e)[:50]}...), retrying with CPU")
                try:
                    import tensorflow as tf
                    with tf.device('/CPU:0'):
                        probs = model(imgs, training=False)
                        probs = probs.numpy()
                    logger.info("✅ TensorFlow CPU inference successful")
                except Exception as cpu_error:
                    logger.error(f"💥 Both GPU and CPU inference failed: {cpu_error}")
                    raise Exception(f"TensorFlow inference failed: {cpu_error}")

        for (image_path, _), prob in zip(path_imgs, probs):
            combined_tags = []
            rating_tag_text = ""
            character_tag_text = ""
            general_tag_text = ""

            # 最初の4つ以降はタグなのでconfidenceがthreshold以上のものを追加する
            # First 4 labels are ratings, the rest are tags: pick any where prediction confidence >= threshold
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tag_name = general_tags[i]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += caption_separator + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    tag_name = character_tags[i - len(general_tags)]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += caption_separator + tag_name
                        if args.character_tags_first: # insert to the beginning
                            combined_tags.insert(0, tag_name)
                        else:
                            combined_tags.append(tag_name)

            # 最初の4つはratingなのでargmaxで選ぶ
            # First 4 labels are actually ratings: pick one with argmax
            if args.use_rating_tags or args.use_rating_tags_as_last_tag:
                ratings_probs = prob[:4]
                rating_index = ratings_probs.argmax()
                found_rating = rating_tags[rating_index]

                if found_rating not in undesired_tags:
                    tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                    rating_tag_text = found_rating
                    if args.use_rating_tags:
                        combined_tags.insert(0, found_rating) # insert to the beginning
                    else:
                        combined_tags.append(found_rating)

            # 一番最初に置くタグを指定する
            # Always put some tags at the beginning
            if always_first_tags is not None:
                for tag in always_first_tags:
                    if tag in combined_tags:
                        combined_tags.remove(tag)
                        combined_tags.insert(0, tag)

            # 先頭のカンマを取る
            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[len(caption_separator) :]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[len(caption_separator) :]

            caption_file = os.path.splitext(image_path)[0] + args.caption_extension

            tag_text = caption_separator.join(combined_tags)

            if args.append_tags:
                # Check if file exists
                if os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        # Read file and remove new lines
                        existing_content = f.read().strip("\n")  # Remove newlines

                    # Split the content into tags and store them in a list
                    existing_tags = [tag.strip() for tag in existing_content.split(stripped_caption_separator) if tag.strip()]

                    # Check and remove repeating tags in tag_text
                    new_tags = [tag for tag in combined_tags if tag not in existing_tags]

                    # Create new tag_text
                    tag_text = caption_separator.join(existing_tags + new_tags)

            with open(caption_file, "wt", encoding="utf-8") as f:
                f.write(tag_text + "\n")
                if args.debug:
                    logger.info("")
                    logger.info(f"{image_path}:")
                    logger.info(f"\tRating tags: {rating_tag_text}")
                    logger.info(f"\tCharacter tags: {character_tag_text}")
                    logger.info(f"\tGeneral tags: {general_tag_text}")

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                # Handle both PyTorch tensors and numpy arrays
                if hasattr(image, 'detach'):
                    image = image.detach().numpy()
                # If it's already numpy, leave it as is
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    logger.error(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= args.batch_size:
                b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
        run_batch(b_imgs)

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("\nTag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_WD14_TAGGER_REPO,
        help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="wd14_tagger_model",
        help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ",
    )
    parser.add_argument(
        "--force_download", action="store_true", help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）",
    )
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for character category, same as --thres if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト",
    )
    parser.add_argument("--frequency_tags", action="store_true", help="Show frequency of tags for images / 画像ごとのタグの出現頻度を表示する")
    parser.add_argument("--onnx", action="store_true", help="use onnx model for inference / onnxモデルを推論に使用する")
    parser.add_argument("--append_tags", action="store_true", help="Append captions instead of overwriting / 上書きではなくキャプションを追記する")
    parser.add_argument(
        "--use_rating_tags", action="store_true", help="Adds rating tags as the first tag / レーティングタグを最初のタグとして追加する",
    )
    parser.add_argument(
        "--use_rating_tags_as_last_tag", action="store_true", help="Adds rating tags as the last tag / レーティングタグを最後のタグとして追加する",
    )
    parser.add_argument(
        "--character_tags_first", action="store_true", help="Always inserts character tags before the general tags / characterタグを常にgeneralタグの前に出力する",
    )
    parser.add_argument(
        "--always_first_tags",
        type=str,
        default=None,
        help="comma-separated list of tags to always put at the beginning, e.g. `1girl,1boy`"
        + " / 必ず先頭に置くタグのカンマ区切りリスト、例 : `1girl,1boy`",
    )
    parser.add_argument(
        "--caption_separator",
        type=str,
        default=", ",
        help="Separator for captions, include space if needed / キャプションの区切り文字、必要ならスペースを含めてください",
    )
    parser.add_argument(
        "--tag_replacement",
        type=str,
        default=None,
        help="tag replacement in the format of `source1,target1;source2,target2; ...`. Escape `,` and `;` with `\`. e.g. `tag1,tag2;tag3,tag4`"
        + " / タグの置換を `置換元1,置換先1;置換元2,置換先2; ...`で指定する。`\` で `,` と `;` をエスケープできる。例: `tag1,tag2;tag3,tag4`",
    )
    parser.add_argument(
        "--character_tag_expand",
        action="store_true",
        help="expand tag tail parenthesis to another tag for character tags. `chara_name_(series)` becomes `chara_name, series`"
        + " / キャラクタタグの末尾の括弧を別のタグに展開する。`chara_name_(series)` は `chara_name, series` になる",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
