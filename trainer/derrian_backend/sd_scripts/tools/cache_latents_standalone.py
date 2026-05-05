"""
Standalone SDXL latent caching script that mirrors the training-time pipeline.

- Uses the same dataset pipeline as `sdxl_train.py` / `train_util.py`
- Optional VAE reflection padding
- Multi-GPU aware via Accelerate + DistributedSampler (each rank caches a unique shard)
- Wraps VAE encode calls in torch.inference_mode() and uses non-blocking transfers to improve throughput
"""

import argparse
import os
from typing import List, Tuple
import math
from contextlib import nullcontext
import signal
from threading import Event

import numpy as np
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import hashlib

from library import config_util, model_util, sdxl_train_util, train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from copy import deepcopy

from library.train_util import IMAGE_TRANSFORMS, load_image, trim_and_resize_if_required
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

stop_event = Event()


def build_dataset_group(args, tokenizers):
    masked_loss = getattr(args, "masked_loss", False)
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            logger.info("dataset_config is required for caching latents")
            raise ValueError("--dataset_config is required for caching latents")

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizers)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        # fallback for arbitrary dataset classes
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizers)

    train_dataset_group.verify_bucket_reso_steps(32)
    return train_dataset_group

def save_latents_to_disk(
    npz_path,
    latents_tensor,
    original_size,
    crop_ltrb,
    flipped_latents_tensor=None,
    alpha_mask=None,
    key_reso_suffix="",
):
    """
    Args:
        npz_path (str): Path to the npz file.
        latents_tensor (torch.Tensor): Latent tensor
        original_size (List[int]): Original size of the image
        crop_ltrb (List[int]): Crop left top right bottom
        flipped_latents_tensor (Optional[torch.Tensor]): Flipped latent tensor
        alpha_mask (Optional[torch.Tensor]): Alpha mask
        key_reso_suffix (str): Key resolution suffix

    Returns:
        None
    """
    kwargs = {}

    if os.path.exists(npz_path):
        # load existing npz and update it
        npz = np.load(npz_path)
        for key in npz.files:
            kwargs[key] = npz[key]

    # TODO float() is needed if vae is in bfloat16. Remove it if vae is float16.
    kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
    kwargs["original_size" + key_reso_suffix] = np.array(original_size)
    kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped" + key_reso_suffix] = flipped_latents_tensor.float().cpu().numpy()
    if alpha_mask is not None:
        kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()
    np.savez(npz_path, **kwargs)


class ShardSampler(Sampler[int]):
    """
    Simple deterministic shard sampler: each rank gets indices[rank :: world_size] with no padding.
    This avoids duplicate samples (and therefore avoids races when writing the same npz from two ranks).
    """

    def __init__(self, dataset_len: int, num_replicas: int, rank: int):
        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        # uneven splits are fine; last ranks may have fewer samples
        return (self.dataset_len - self.rank + self.num_replicas - 1) // self.num_replicas


def build_dataloader(dataset_group, args, accelerator):
    # shard the dataset so each process caches distinct files (no padding/duplicates)
    sampler = ShardSampler(len(dataset_group), accelerator.num_processes, accelerator.process_index)

    def passthrough_collate_fn(examples):
        return examples[0]

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count()) if args.max_data_loader_n_workers is not None else 0
    if args.vae_batch_size is not None and args.vae_batch_size >= 512:
        logger.warning("VAE batch sizes >=512 can overflow pytorch's 32-bit indexing; lowering to 512.")
        args.vae_batch_size = 512

    dataloader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        collate_fn=passthrough_collate_fn,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers and n_workers > 0,
        pin_memory=True,
    )
    if n_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 8

    train_dataloader = DataLoader(dataset_group, **dataloader_kwargs)
    return accelerator.prepare(train_dataloader)


def cache_batch_latents_fast(
    vae,
    batch_bucket_reso: Tuple[int, int],
    images: List[np.ndarray],
    resized_sizes: List[Tuple[int, int]],
    flip_aug: bool,
    use_alpha_mask: bool,
    random_crop: bool,
    device: torch.device,
    non_blocking: bool = True,
    autocast_dtype=None,
):
    processed_images, original_sizes, crop_ltrbs, alpha_masks = [], [], [], []

    for img_np, resized_size in zip(images, resized_sizes):
        image_np, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, img_np, batch_bucket_reso, resized_size)
        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image_np.shape[2] == 4:
                alpha_mask = image_np[:, :, 3].astype(np.float32) / 255.0
                alpha_mask = torch.from_numpy(alpha_mask)
            else:
                alpha_mask = torch.ones_like(image_np[:, :, 0], dtype=torch.float32)
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image_np = image_np[:, :, :3]  # drop alpha if present
        processed_images.append(IMAGE_TRANSFORMS(image_np))

    img_tensors = torch.stack(processed_images)
    if img_tensors.device.type == "cpu":
        img_tensors = img_tensors.pin_memory()
    img_tensors = img_tensors.to(device=device, dtype=vae.dtype, non_blocking=non_blocking)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        latents = vae.encode(img_tensors).latent_dist.sample()

        if flip_aug:
            flipped_tensors = torch.flip(img_tensors, dims=[3])
            flipped_latents = vae.encode(flipped_tensors).latent_dist.sample()
        else:
            flipped_latents = None

    # sync copy back to CPU to avoid partially-copied buffers being saved
    latents_cpu = latents.to("cpu").contiguous()
    flipped_latents_cpu = flipped_latents.to("cpu").contiguous() if flipped_latents is not None else None

    return latents_cpu, flipped_latents_cpu, original_sizes, crop_ltrbs, alpha_masks


def cache_latents_fast(dataset, vae, accelerator, args):
    """
    Faster caching path using inference_mode + non_blocking transfers.
    Shards unique images per-rank to avoid duplicate writes.
    """
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    class Condition:
        def __init__(self, reso, flip_aug, alpha_mask, random_crop):
            self.reso = reso
            self.flip_aug = flip_aug
            self.alpha_mask = alpha_mask
            self.random_crop = random_crop

        def __eq__(self, other):
            return (
                self.reso == other.reso
                and self.flip_aug == other.flip_aug
                and self.alpha_mask == other.alpha_mask
                and self.random_crop == other.random_crop
            )

    image_infos = list(dataset.image_data.values())
    image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

    # contiguous split among ranks (single pass)
    total = len(image_infos)
    if total:
        chunk = math.ceil(total / world_size)
        start = rank * chunk
        end = min(total, start + chunk)
        image_infos = image_infos[start:end]
    else:
        image_infos = []

    batches = []
    batch = []
    current_condition = None

    for info in image_infos:
        subset = dataset.image_to_subset[info.image_key]
        info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
        if args.skip_existing and os.path.exists(info.latents_npz):
            continue

        condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
        if len(batch) > 0 and current_condition != condition:
            batches.append((current_condition, batch))
            batch = []

        batch.append(info)
        current_condition = condition

        if len(batch) >= args.vae_batch_size:
            batches.append((current_condition, batch))
            batch = []
            current_condition = None

    if len(batch) > 0:
        batches.append((current_condition, batch))

    iterator = tqdm(
        batches,
        desc=f"Fast cache on process {rank}",
        position=rank,
        mininterval=0.5,
        dynamic_ncols=True,
        leave=True,
    )

    fast_autocast_dtype = None  # None disables autocast; float32 is invalid for CUDA autocast
    if args.fast_autocast_dtype == "fp16":
        fast_autocast_dtype = torch.float16
    elif args.fast_autocast_dtype == "bf16":
        fast_autocast_dtype = torch.bfloat16

    for condition, batch_infos in iterator:
        if stop_event.is_set():
            logger.info("Interrupt received; stopping after current batch.")
            break

        images = []
        resized_sizes = []
        for info in batch_infos:
            img = load_image(info.absolute_path, condition.alpha_mask)
            images.append(img)
            resized_sizes.append(info.resized_size)

        latents_cpu, flipped_latents_cpu, original_sizes, crop_ltrbs, alpha_masks = cache_batch_latents_fast(
            vae=vae,
            batch_bucket_reso=condition.reso,
            images=images,
            resized_sizes=resized_sizes,
            flip_aug=condition.flip_aug,
            use_alpha_mask=condition.alpha_mask,
            random_crop=condition.random_crop,
            device=device,
            non_blocking=True,
            autocast_dtype=fast_autocast_dtype,
        )

        for i, (info, original_size, crop_ltrb, alpha_mask) in enumerate(
            zip(batch_infos, original_sizes, crop_ltrbs, alpha_masks)
        ):
            save_latents_to_disk(
                info.latents_npz,
                latents_cpu[i],
                original_size,
                crop_ltrb,
                flipped_latents_cpu[i] if flipped_latents_cpu is not None else None,
                alpha_mask,
            )

        if stop_event.is_set():
            logger.info("Interrupt received; exiting fast cache loop.")
            break


def cache_latents_for_dataset(dataset, vae, accelerator, args):
    """
    Cache latents per unique image, sharded by rank to avoid duplicate writes.
    Mirrors train_util.cache_latents batching logic, but adds rank-based filtering and inference_mode.
    """
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    class Condition:
        def __init__(self, reso, flip_aug, alpha_mask, random_crop):
            self.reso = reso
            self.flip_aug = flip_aug
            self.alpha_mask = alpha_mask
            self.random_crop = random_crop

        def __eq__(self, other):
            return (
                self.reso == other.reso
                and self.flip_aug == other.flip_aug
                and self.alpha_mask == other.alpha_mask
                and self.random_crop == other.random_crop
            )

    image_infos = list(dataset.image_data.values())
    image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

    batches = []
    batch = []
    current_condition = None

    for info in image_infos:
        subset = dataset.image_to_subset[info.image_key]

        # set latents_npz path and skip existing if requested
        info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
        if args.skip_existing and os.path.exists(info.latents_npz):
            continue

        condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
        if len(batch) > 0 and current_condition != condition:
            batches.append((current_condition, batch))
            batch = []

        batch.append(info)
        current_condition = condition

        if len(batch) >= args.vae_batch_size:
            batches.append((current_condition, batch))
            batch = []
            current_condition = None

    if len(batch) > 0:
        batches.append((current_condition, batch))

    iterator = tqdm(
        batches,
        desc=f"Caching latents on process {rank}",
        position=rank,
        mininterval=0.5,
        dynamic_ncols=True,
        leave=True,
    )

    for condition, batch_infos in iterator:
        if stop_event.is_set():
            logger.info("Interrupt received; stopping after current batch.")
            break

        images = []
        resized_sizes = []
        for info in batch_infos:
            img = load_image(info.absolute_path, condition.alpha_mask)
            images.append(img)
            resized_sizes.append(info.resized_size)

        latents_cpu, flipped_latents_cpu, original_sizes, crop_ltrbs, alpha_masks = cache_batch_latents_fast(
            vae=vae,
            batch_bucket_reso=condition.reso,
            images=images,
            resized_sizes=resized_sizes,
            flip_aug=condition.flip_aug,
            use_alpha_mask=condition.alpha_mask,
            random_crop=condition.random_crop,
            device=device,
            non_blocking=True,
        )

        for i, (info, original_size, crop_ltrb, alpha_mask) in enumerate(
            zip(batch_infos, original_sizes, crop_ltrbs, alpha_masks)
        ):
            save_latents_to_disk(
                info.latents_npz,
                latents_cpu[i],
                original_size,
                crop_ltrb,
                flipped_latents_cpu[i] if flipped_latents_cpu is not None else None,
                alpha_mask,
            )

        if stop_event.is_set():
            logger.info("Interrupt received; exiting fast cache loop.")
            break


def cache_latents(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)

    # Just force to true, caching to disk is the only valid approach
    args.cache_latents_to_disk = True

    assert args.cache_latents_to_disk, "cache_latents_to_disk must be True"

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
    tokenizers = [tokenizer1, tokenizer2]

    train_dataset_group = build_dataset_group(args, tokenizers)
    if len(train_dataset_group) == 0:
        logger.error("No images were loaded from the dataset. Please check your dataset_config.")
        return

    # prepare accelerator and VAE
    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    logger.info(f"accelerator device: {accelerator.device}, world_size: {accelerator.num_processes}")

    weight_dtype, _ = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    logger.info("load model")
    setattr(args, "disable_mmap_load_safetensors", False)
    _, _, _, vae, _, _, _ = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    if getattr(args, "vae_reflection_padding", False):
        vae = model_util.use_reflection_padding(vae)

    if getattr(args, "vae_channels_last", False):
        vae = vae.to(memory_format=torch.channels_last)

    if getattr(args, "allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    def handle_interrupt(signum, frame):
        logger.warning(f"Received signal {signum}; will stop after the current batch.")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # use latents mode to expose images and meta
    train_dataset_group.set_caching_mode("latents")

    # fast path uses our inference-mode/non-blocking implementation; safe path reuses dataset.cache_latents with per-rank filtering
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    for idx, dataset in enumerate(train_dataset_group.datasets):
        logger.info(f"[Dataset {idx}] ({'fast' if args.fast_cache else 'safe'}) caching on rank {rank}/{world_size}")

        if args.fast_cache:
            cache_latents_fast(dataset, vae, accelerator, args)
            accelerator.wait_for_everyone()
            continue

        dataset = deepcopy(dataset)

        # contiguous split among ranks (safe mode)
        image_keys = sorted(dataset.image_data.keys())
        total = len(image_keys)
        if total:
            chunk = math.ceil(total / world_size)
            start = rank * chunk
            end = min(total, start + chunk)
            my_keys = set(image_keys[start:end])
        else:
            my_keys = set()

        dataset.image_data = {k: v for k, v in dataset.image_data.items() if k in my_keys}
        dataset.image_to_subset = {k: v for k, v in dataset.image_to_subset.items() if k in my_keys}
        new_buckets = []
        for bucket in dataset.bucket_manager.buckets:
            filtered = [k for k in bucket if k in my_keys]
            if filtered:
                new_buckets.append(filtered)
        dataset.bucket_manager.buckets = new_buckets

        # allow every rank to write its shard
        dataset.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, True)
        accelerator.wait_for_everyone()

    accelerator.print("Finished caching latents.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)

    parser.add_argument("--sdxl", action="store_true", default=True, help="Use SDXL model (required)")
    parser.add_argument("--no_half_vae", action="store_true", help="do not use fp16/bf16 VAE in mixed precision")
    parser.add_argument("--skip_existing", action="store_true", help="skip images if npz already exists")
    parser.add_argument("--vae_reflection_padding", action="store_true", help="apply reflection padding to VAE")
    parser.add_argument("--vae_channels_last", action="store_true", help="convert VAE to channels_last for possible speedup")
    parser.add_argument("--allow_tf32", action="store_true", help="enable TF32 matmul on Ampere+ GPUs")
    parser.add_argument("--fast_cache", action="store_true", help="use optimized caching path (inference_mode + non_blocking)")
    parser.add_argument(
        "--fast_autocast_dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16"],
        help="optional autocast dtype for fast caching (can speed up when --no_half_vae is used)",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_latents(args)
