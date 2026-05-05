# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml
import itertools
import ast
import random
import numpy as np

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device
from library.ramtorch_util import apply_ramtorch_to_module

init_ipex()


from diffusers import DDPMScheduler
from library import deepspeed_utils, sdxl_model_util, strategy_base, strategy_sd, strategy_sdxl, sai_model_spec, model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library.edm2_loss_utils import prepare_edm2_loss_weighting, plot_edm2_loss_weighting_check, plot_edm2_loss_weighting
import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel


UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def _unet_block_index_from_name(name: str) -> int:
    if name.startswith("time_embed.") or name.startswith("label_emb."):
        return 0  # 0
    if name.startswith("input_blocks."):  # 1-9
        return 1 + int(name.split(".")[1])
    if name.startswith("middle_block."):  # 10-12
        return 10 + int(name.split(".")[1])
    if name.startswith("output_blocks."):  # 13-21
        return 13 + int(name.split(".")[1])
    if name.startswith("out."):  # 22
        return 22
    raise ValueError(f"unexpected parameter name: {name}")


def _unet_block_prefix_from_name(name: str) -> str:
    if name.startswith("time_embed.") or name.startswith("label_emb."):
        return name.split(".")[0]
    if name.startswith("input_blocks.") or name.startswith("output_blocks.") or name.startswith("middle_block."):
        parts = name.split(".")
        if len(parts) < 2:
            raise ValueError(f"unexpected block format: {name}")
        return ".".join(parts[:2])
    if name.startswith("out."):
        return name.split(".")[0]
    raise ValueError(f"unexpected parameter name: {name}")


def get_block_params_to_optimize(
    unet: SdxlUNet2DConditionModel, block_lrs: List[float], frozen_blocks: set[int] | None = None
) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        block_index = _unet_block_index_from_name(name)
        if frozen_blocks and block_index in frozen_blocks:
            continue

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def freeze_unet_blocks(unet: SdxlUNet2DConditionModel, frozen_blocks: set[int]) -> None:
    """Mark selected U-Net blocks as frozen (no gradients)."""

    if not frozen_blocks:
        return

    for name, param in unet.named_parameters():
        block_index = _unet_block_index_from_name(name)
        if block_index in frozen_blocks:
            param.requires_grad_(False)


def describe_unet_blocks(unet: SdxlUNet2DConditionModel):
    """Collect a short description of each U-Net block index."""

    info = {}
    for name, param in unet.named_parameters():
        block_index = _unet_block_index_from_name(name)
        block_prefix = _unet_block_prefix_from_name(name)
        block_entry = info.setdefault(block_index, {"example": name, "params": 0, "layers": set()})
        block_entry["params"] += param.numel()

        layer_path = name.rsplit(".", 1)[0]  # strip parameter name
        suffix = ""
        if layer_path == block_prefix:
            suffix = ""
        elif layer_path.startswith(f"{block_prefix}."):
            suffix = layer_path[len(block_prefix) + 1 :]
        else:
            suffix = layer_path

        if suffix:
            tokens = [token for token in suffix.split(".") if token]
            while tokens and tokens[0].isdigit():
                tokens.pop(0)
            layer_name = ".".join(tokens) if tokens else block_prefix
        else:
            layer_name = block_prefix

        block_entry["layers"].add(layer_name)

    for entry in info.values():
        entry["layers"] = sorted(entry["layers"])
    return info


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def switch_rng_state(val_seed: int, accelerator) -> tuple[torch.ByteTensor, torch.ByteTensor | None, tuple]:
    cpu_rng_state = torch.get_rng_state()
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    if accelerator.device.type == "cuda":
        gpu_rng_state = torch.cuda.get_rng_state()
    elif accelerator.device.type == "xpu":
        gpu_rng_state = torch.xpu.get_rng_state()
    elif accelerator.device.type == "mps":
        gpu_rng_state = torch.mps.get_rng_state()
    else:
        gpu_rng_state = None

    random.seed(val_seed)
    np.random.seed(val_seed)
    torch.manual_seed(val_seed)
    if accelerator.device.type == "cuda":
        torch.cuda.manual_seed_all(val_seed)

    return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)


def restore_rng_state(rng_states: tuple[torch.ByteTensor, torch.ByteTensor | None, tuple], accelerator):
    cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states
    torch.set_rng_state(cpu_rng_state)
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    if gpu_rng_state is not None:
        if accelerator.device.type == "cuda":
            torch.cuda.set_rng_state(gpu_rng_state)
        elif accelerator.device.type == "xpu":
            torch.xpu.set_rng_state(gpu_rng_state)
        elif accelerator.device.type == "mps":
            torch.mps.set_rng_state(gpu_rng_state)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    train_util.set_torch_cuda_reduced_precision(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    assert (
        not args.weighted_captions or not args.cache_text_encoder_outputs
    ), "weighted_captions is not supported when caching text encoder outputs / cache_text_encoder_outputsを使うときはweighted_captionsはサポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    frozen_unet_blocks = set()
    if args.freeze_unet_blocks:
        for token in args.freeze_unet_blocks.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid U-Net block index '{token}' in --freeze_unet_blocks") from exc
            if idx < 0 or idx >= UNET_NUM_BLOCKS_FOR_BLOCK_LR:
                raise ValueError(f"--freeze_unet_blocks indices must be in [0, {UNET_NUM_BLOCKS_FOR_BLOCK_LR - 1}]")
            frozen_unet_blocks.add(idx)

    vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
    vae_shift_factor = 0.0
    if args.vae_custom_scale is not None:
        vae_scale_factor = float(args.vae_custom_scale)
        logger.info(f"Using custom VAE scale factor: {vae_scale_factor}")
    if args.vae_custom_shift is not None:
        vae_shift_factor = float(args.vae_custom_shift)
        logger.info(f"Using custom VAE shift factor: {vae_shift_factor}")
    args.vae_scale_factor = vae_scale_factor
    args.vae_shift_factor = vae_shift_factor

    if args.flow_model:
        logger.info("Using Rectified Flow training objective.")
        if args.v_parameterization:
            raise ValueError("`--flow_model` is incompatible with `--v_parameterization`; Rectified Flow already predicts velocity.")
        if args.min_snr_gamma:
            logger.warning("`--min_snr_gamma` is ignored when Rectified Flow is enabled.")
            args.min_snr_gamma = None
        if args.debiased_estimation_loss:
            logger.warning("`--debiased_estimation_loss` is ignored when Rectified Flow is enabled.")
            args.debiased_estimation_loss = False
        if args.scale_v_pred_loss_like_noise_pred:
            logger.warning("`--scale_v_pred_loss_like_noise_pred` is ignored when Rectified Flow is enabled.")
            args.scale_v_pred_loss_like_noise_pred = False
        if args.v_pred_like_loss:
            logger.warning("`--v_pred_like_loss` is ignored when Rectified Flow is enabled.")
            args.v_pred_like_loss = None
        if args.flow_use_ot:
            logger.info("Using cosine optimal transport pairing for Rectified Flow batches.")
        shift_enabled = args.flow_uniform_shift or args.flow_uniform_static_ratio is not None
        if args.flow_timestep_distribution == "logit_normal":
            flow_logit_std = float(getattr(args, "flow_logit_std", 1.0))
            flow_logit_mean = float(getattr(args, "flow_logit_mean", 0.0))
            if flow_logit_std == 0:
                raise ValueError("`--flow_logit_std` must be non-zero.")
            logger.info(
                "Rectified Flow timesteps sampled from logit-normal distribution with "
                f"mean={flow_logit_mean}, std={flow_logit_std}."
            )
        elif args.flow_timestep_distribution == "uniform":
            logger.info("Rectified Flow timesteps sampled uniformly in [0, 1].")
        else:
            raise ValueError(f"Unknown Rectified Flow timestep distribution: {args.flow_timestep_distribution}")
        if shift_enabled:
            if args.flow_uniform_static_ratio is not None:
                flow_uniform_static_ratio = float(getattr(args, "flow_uniform_static_ratio", 0.0))
                if flow_uniform_static_ratio <= 0:
                    raise ValueError("`--flow_uniform_static_ratio` must be positive.")
                logger.info(
                    f"Applying Rectified Flow timestep shift with static ratio={flow_uniform_static_ratio}."
                )
            else:
                logger.info(
                    f"Applying resolution-dependent Rectified Flow timestep shift with base pixels={args.flow_uniform_base_pixels}."
                )

    if args.contrastive_flow_matching and not (args.v_parameterization or args.flow_model):
        raise ValueError("`--contrastive_flow_matching` requires either v-parameterization or Rectified Flow.")

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    train_util.args_set_seed(args)

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]  # will be removed in the future

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    if args.cache_latents:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    if args.protected_tags_file:
        logger.info("Injecting protected_tags_file into datasets...")
        for ds in train_dataset_group.datasets:
            ds.protected_tags_file = args.protected_tags_file
    if args.log_caption_tag_dropout:
        logger.info("Enabling caption tag dropout logging for datasets...")
        for ds in train_dataset_group.datasets:
            ds.log_caption_tag_dropout = True
    if args.log_caption_dropout:
        logger.info("Enabling caption dropout logging for datasets...")
        for ds in train_dataset_group.datasets:
            ds.log_caption_dropout = True

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        if val_dataset_group is not None:
            train_util.debug_dataset(val_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    if args.vae_reflection_padding:
        vae = model_util.use_reflection_padding(vae)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    if args.use_ramtorch_vae:
        vae = apply_ramtorch_to_module(vae, "vae", accelerator.device, vae_dtype)
        
    if args.list_unet_blocks:
        block_info = describe_unet_blocks(unet)
        accelerator.print("SDXL U-Net block mapping (index -> example parameter) with param counts and layers:")
        for idx in sorted(block_info.keys()):
            info = block_info[idx]
            layers = ", ".join(info.get("layers", [])) or "-"
            accelerator.print(f"{idx:02d}: {info['example']} (params: {info['params']:,})")
            accelerator.print(f"    layers: {layers}")
        return

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)
        if val_dataset_group is not None:
            val_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate != 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 != 0
        train_text_encoder2 = lr_te2 != 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # TextEncoderの出力をキャッシュする
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            text_encoder_output_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, False, is_weighted=args.weighted_captions
            )
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

            text_encoder1.to(accelerator.device)
            text_encoder2.to(accelerator.device)
            with accelerator.autocast():
                train_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator)
                if val_dataset_group is not None:
                    val_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator)

        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if train_unet and frozen_unet_blocks:
        accelerator.print(f"Freezing U-Net blocks: {sorted(frozen_unet_blocks)}")
        freeze_unet_blocks(unet, frozen_unet_blocks)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            trainable_params = [p for p in unet.parameters() if p.requires_grad]
            params_to_optimize.append({"params": trainable_params, "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs, frozen_unet_blocks))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    if args.fused_optimizer_groups:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each group of parameters.
        # This balances memory usage and management complexity.

        # calculate total number of parameters
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

        # split params into groups, keeping the learning rate the same for all params in a group
        # this will increase the number of groups if the learning rate is different for different params (e.g. U-Net and text encoders)
        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                # if the learning rate is different for different params, start a new group
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr

                param_group.append(p)

                # if the group has enough parameters, start a new group
                if len(param_group) == params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
                    param_group_lr = -1

        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")

    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()
    if val_dataset_group is not None:
        val_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if val_dataset_group is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group,
            shuffle=False,
            batch_size=1,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )
        val_dataloader = accelerator.prepare(val_dataloader)
        cyclic_val_dataloader = itertools.cycle(val_dataloader)
    else:
        val_dataloader, cyclic_val_dataloader = None, None

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    if args.fused_optimizer_groups:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    elif args.fused_optimizer_groups:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(args, noise_scheduler, accelerator)

    if args.edm2_loss_weighting:
        # Add to training models so accelerator.accumulate handles it if necessary
        training_models.append(edm2_model)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, [text_encoder1, text_encoder2], unet
    )

    if args.edm2_loss_weighting:
        plot_edm2_loss_weighting(args, 0, edm2_model, 1000, accelerator.device)

    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    loss_recorder = train_util.EMARecorder()
    val_loss_recorder = train_util.EMARecorder()

    if args.edm2_loss_weighting:
        loss_scaled_recorder = train_util.EMARecorder()
        loss_edm2_recorder = train_util.EMARecorder()

    def calculate_val_loss(epoch_step):
        if not train_util.calculate_val_loss_check(args, global_step, epoch_step, val_dataloader, train_dataloader):
            return None, None, {}

        _val_seed = args.validation_seed if args.validation_seed is not None else args.seed
        rng_states = switch_rng_state(int(_val_seed) if _val_seed is not None else 23, accelerator)

        timesteps_list = ast.literal_eval(args.validation_timesteps)

        accelerator.print("")
        accelerator.print("Validating バリデーション処理...")
        total_loss = 0.0

        # set eval
        for m in training_models:
            m.eval()
        
        with torch.no_grad():
            validation_steps = min(int(args.max_validation_steps), len(val_dataloader)) if args.max_validation_steps is not None else len(val_dataloader)
            val_dataloader_seed = random.randint(global_step, 0x7FFFFFFF)
            val_dataloader_state = random.Random(val_dataloader_seed).getstate()
            for val_step in tqdm(range(validation_steps), desc='Validation Steps'):
                val_original_state = random.getstate()
                random.setstate(val_dataloader_state)
                batch = next(cyclic_val_dataloader)
                val_dataloader_state = random.getstate()
                random.setstate(val_original_state)
                
                # Validation batch processing (simplified from train loop)
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        latents = torch.nan_to_num(latents, 0, out=latents)
                if args.vae_shift_factor != 0.0:
                    latents = latents - args.vae_shift_factor
                latents = latents * args.vae_scale_factor

                text_encoder_conds = []
                masks_reshaped = []
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    text_encoder_conds = text_encoder_outputs_list
                    masks_reshaped = text_encoder_outputs_list[3:]

                if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None:
                    with accelerator.autocast():
                        if args.weighted_captions:
                            input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                            encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                input_ids_list,
                                weights_list,
                            )
                        else:
                            input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                            masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                            encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                input_ids,
                                attn_masks=masks,
                            )
                        if args.full_fp16:
                            encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                    if len(text_encoder_conds) == 0:
                        text_encoder_conds = encoded_text_encoder_conds
                    else:
                        for i in range(len(encoded_text_encoder_conds)):
                            if encoded_text_encoder_conds[i] is not None:
                                text_encoder_conds[i] = encoded_text_encoder_conds[i]

                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_conds
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                batch_size = latents.shape[0]
                
                # Loop through validation timesteps
                for fixed_timestep in timesteps_list:
                    timesteps = torch.full((batch_size,), fixed_timestep, dtype=torch.long, device=latents.device)
                    
                    noise, noisy_latents, _ = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents, fixed_timesteps=timesteps, is_train=False
                    )
                    
                    noisy_latents = noisy_latents.to(weight_dtype)

                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, encoder_attention_mask=masks_reshaped[1] if len(masks_reshaped) > 1 else None)

                    latents = latents.to(torch.float64)
                    noise = noise.to(torch.float64)
                    
                    if args.flow_model:
                        target = noise - latents
                    elif args.v_parameterization:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    noise_pred = noise_pred.to(dtype=torch.float64)
                    target = target.to(dtype=torch.float64)

                    loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)
                    loss = loss.mean()
                    total_loss += loss.item()

        current_val_loss = total_loss / (validation_steps * len(timesteps_list))
        val_loss_recorder.add(current_val_loss)
        average_val_loss = val_loss_recorder.average
        
        restore_rng_state(rng_states, accelerator)
        
        # restore train mode
        for m in training_models:
            m.train()
            
        return current_val_loss, average_val_loss, {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

    current_val_loss, average_val_loss, val_logs = calculate_val_loss(0)
    if current_val_loss is not None:
        if len(accelerator.trackers) > 0:
            accelerator.log(val_logs, step=0)

    # Initialize loss accumulators for logging
    current_global_step_loss = 0.0
    current_global_step_loss_scaled = 0.0
    current_global_step_loss_edm2 = 0.0
    accumulation_counter = 0

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            accumulation_counter += 1

            if args.fused_optimizer_groups:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                if args.vae_shift_factor != 0.0:
                    latents = latents - args.vae_shift_factor
                latents = latents * args.vae_scale_factor

                text_encoder_conds = []
                masks_reshaped = []
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
                    masks_reshaped = text_encoder_outputs_list[3:]

                if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or args.train_text_encoder:
                    # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
                    with torch.set_grad_enabled(args.train_text_encoder), accelerator.autocast():
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                            encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                input_ids_list,
                                weights_list,
                            )
                        else:
                            input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                            masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                            encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                input_ids,
                                attn_masks=masks,
                            )
                        if args.full_fp16:
                            encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                    # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                    if len(text_encoder_conds) == 0:
                        text_encoder_conds = encoded_text_encoder_conds
                    else:
                        # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                        for i in range(len(encoded_text_encoder_conds)):
                            if encoded_text_encoder_conds[i] is not None:
                                text_encoder_conds[i] = encoded_text_encoder_conds[i]

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_conds
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                needs_dynamic_shift = (
                    args.flow_model and args.flow_uniform_shift and args.flow_uniform_static_ratio is None
                )
                if needs_dynamic_shift:
                    if target_size is None:
                        raise ValueError(
                            "Resolution-dependent Rectified Flow shift requires target size information in the batch."
                        )
                    pixel_counts = (target_size[:, 0] * target_size[:, 1]).to(latents.device, torch.float32)
                else:
                    pixel_counts = None

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, pixel_counts=pixel_counts)

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, encoder_attention_mask=masks_reshaped[1])

                # Upcast for grokking
                latents = latents.to(torch.float64)
                noise = noise.to(torch.float64)

                if args.flow_model:
                    target = noise - latents
                elif args.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                # Upcast for grokking
                noise_pred = noise_pred.to(dtype=torch.float64)
                target = target.to(dtype=torch.float64)

                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                if (
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))
                    if args.contrastive_flow_matching and latents.size(0) > 1:
                        # CRITICAL FIX: Add .detach() to prevent gradients flowing through negative samples
                        negative_latents = latents.roll(1, 0).detach()
                        negative_noise = noise.roll(1, 0).detach()
                        with torch.no_grad():
                            if getattr(args, "flow_model", False):
                                target_negative = negative_noise - negative_latents
                            else:
                                target_negative = noise_scheduler.get_velocity(negative_latents, negative_noise, timesteps)

                        # Handle cast for CFM
                        target_negative = target_negative.to(dtype=torch.float64)

                        loss_contrastive = torch.nn.functional.mse_loss(
                            noise_pred, target_negative, reduction="none"
                        )
                        # Store CFM component for logging (before applying lambda)
                        #loss_cfm = loss_contrastive.mean([1, 2, 3]).mean().detach()
                        loss = loss - float(args.cfm_lambda) * loss_contrastive
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)


                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
                else:
                    per_pixel_loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "mean", huber_c, scale=float(args.loss_scale))
                    if args.contrastive_flow_matching and latents.size(0) > 1:
                        negative_latents = latents.roll(1, 0)
                        negative_noise = noise.roll(1, 0)
                        with torch.no_grad():
                            if args.flow_model:
                                target_negative = negative_noise - negative_latents
                            else:
                                target_negative = noise_scheduler.get_velocity(negative_latents, negative_noise, timesteps)

                        # Cast to float64 (Double Precision) for Grokking
                        target_negative = target_negative.to(dtype=torch.float64)

                        loss_contrastive = torch.nn.functional.mse_loss(
                            noise_pred, target_negative, reduction="none"
                        )
                        per_pixel_loss = per_pixel_loss - float(args.cfm_lambda) * loss_contrastive
                    loss = per_pixel_loss

                loss = loss.mean()

                pre_scaling_loss = loss.detach()  # for logging (pre-EDM2 scale)
                edm2_loss_pre_scale = pre_scaling_loss  # capture before EDM2 for logging

                if args.edm2_loss_weighting:
                    loss, loss_scaled = edm2_model(loss, timesteps)
                    loss_scaled = loss_scaled.mean()  # for logging
                else:
                    loss_scaled = None

                if loss.ndim != 0:
                    loss = loss.mean()

                accelerator.backward(loss)

                if args.use_ramtorch:
                    torch.cuda.synchronize()

                edm2_loss = edm2_loss_pre_scale  # pre-scale loss for EDM2 logging
                loss = pre_scaling_loss

                # Sync EDM2 gradients explicitly across GPUs (for DDP and DeepSpeed compatibility)
                # Must happen before any gradient clipping
                if args.edm2_loss_weighting and accelerator.sync_gradients:
                    for param in edm2_model.parameters():
                        if param.grad is not None:
                            param.grad = accelerator.reduce(param.grad, reduction="mean")

                if not (args.fused_backward_pass or args.fused_optimizer_groups):
                    if accelerator.sync_gradients:
                        if args.max_grad_norm != 0.0 or args.edm2_loss_weighting:
                            accelerator.unscale_gradients()

                        if args.max_grad_norm != 0.0:
                            params_to_clip = []
                            for m in training_models:
                                # Skip EDM2 model - it has its own gradient clipping with potentially different norm
                                if args.edm2_loss_weighting and m is edm2_model:
                                    continue
                                params_to_clip.extend(m.parameters())
                            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if args.edm2_loss_weighting:
                            # Apply gradient clipping for EDM2 (with separate grad norm if specified)
                            edm2_grad_norm = (args.edm2_loss_weighting_max_grad_norm
                                             if args.edm2_loss_weighting_max_grad_norm is not None
                                             else args.max_grad_norm)
                            if edm2_grad_norm != 0.0:
                                edm2_params = list(accelerator.unwrap_model(edm2_model).parameters())
                                torch.nn.utils.clip_grad_norm_(edm2_params, edm2_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

                if args.edm2_loss_weighting:
                    edm2_optimizer.step()
                    edm2_lr_scheduler.step()
                    edm2_optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    tokenizers,
                    [text_encoder1, text_encoder2],
                    unet,
                )

                # Validation during steps
                current_val_loss, average_val_loss, val_logs = calculate_val_loss(step)

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

                        if args.edm2_loss_weighting:
                            loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step, "_edm2_loss_weights")
                            loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
                            accelerator.print(f"saving edm2 loss weights: {loss_weights_file}")
                            accelerator.unwrap_model(edm2_model).save_weights(loss_weights_file, edm2_model.dtype, None)
                            
                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no, "_edm2_loss_weights")
                                remove_loss_weights_file = os.path.join(args.output_dir, remove_loss_weights_ckpt_name)
                                if os.path.exists(remove_loss_weights_file):
                                    os.remove(remove_loss_weights_file)

                if plot_edm2_loss_weighting_check(args, global_step):
                    plot_edm2_loss_weighting(args, global_step, edm2_model, 1000, accelerator.device)

            current_global_step_loss += loss.detach().item()  # 平均なのでbatch sizeは関係ないはず

            if args.edm2_loss_weighting:
                current_global_step_loss_scaled += loss_scaled.detach().item()
                current_global_step_loss_edm2 += edm2_loss.detach().item()

            if accelerator.sync_gradients:
                avg_loss = current_global_step_loss / accumulation_counter
                logs = {"loss": avg_loss}
                
                if current_val_loss is not None:
                    logs.update(val_logs)

                if args.edm2_loss_weighting:
                    avg_loss_scaled = current_global_step_loss_scaled / accumulation_counter
                    avg_loss_edm2 = current_global_step_loss_edm2 / accumulation_counter
                    logs["loss/scaled"] = avg_loss_scaled
                    logs["loss/edm2"] = avg_loss_edm2
                    logs["lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                accelerator.log(logs, step=global_step)
                
                # Recorders
                loss_recorder.add(avg_loss)
                if args.edm2_loss_weighting:
                    loss_scaled_recorder.add(avg_loss_scaled)
                    loss_edm2_recorder.add(avg_loss_edm2)
                
                # Reset accumulators
                current_global_step_loss = 0.0
                if args.edm2_loss_weighting:
                    current_global_step_loss_scaled = 0.0
                    current_global_step_loss_edm2 = 0.0
                accumulation_counter = 0

            # Reset val logs for next steps
            current_val_loss, average_val_loss, val_logs = None, None, {}

            avr_loss: float = loss_recorder.average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

                if args.edm2_loss_weighting:
                    if (epoch + 1) % args.save_every_n_epochs == 0:
                        loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1, "_edm2_loss_weights")
                        loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
                        accelerator.print(f"saving edm2 loss weights: {loss_weights_file}")
                        accelerator.unwrap_model(edm2_model).save_weights(loss_weights_file, edm2_model.dtype, None)

                        remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                        if remove_epoch_no is not None:
                            remove_loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no, "_edm2_loss_weights")
                            remove_loss_weights_file = os.path.join(args.output_dir, remove_loss_weights_ckpt_name)
                            if os.path.exists(remove_loss_weights_file):
                                os.remove(remove_loss_weights_file)

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    if is_main_process:
        # Save EDM2 loss weights before releasing accelerator
        if args.edm2_loss_weighting:
            loss_weights_ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as, "_edm2_loss_weights")
            loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
            accelerator.print(f"saving edm2 loss weights: {loss_weights_file}")
            accelerator.unwrap_model(edm2_model).save_weights(loss_weights_file, edm2_model.dtype, None)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--vae_reflection_padding",
        action="store_true",
        help="switch VAE convolutions to reflection padding (improves border quality for some custom VAEs) / VAEの畳み込みを反射パディングに切り替える",
    )
    parser.add_argument(
        "--vae_custom_scale",
        type=float,
        default=None,
        help="override the latent scaling factor applied after VAE encode (default matches SDXL) / VAEエンコード後のスケーリング係数を上書きする",
    )
    parser.add_argument(
        "--vae_custom_shift",
        type=float,
        default=None,
        help="apply a constant latent shift before scaling (e.g. Flux-style offset) / スケーリング前に潜在表現へ定数シフトを適用する",
    )
    parser.add_argument(
        "--disable_cross_attn_mask",
        action="store_true",
        help="disable SDXL cross-attention masking so padded tokens are treated as normal tokens / SDXLのcross-attentionマスク処理を無効化する",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--list_unet_blocks",
        action="store_true",
        help="print SDXL U-Net block indices with example parameter names, then exit / U-Netブロックの番号とサンプルのパラメータ名を表示して終了",
    )
    parser.add_argument(
        "--freeze_unet_blocks",
        type=str,
        default=None,
        help="comma-separated block indices to freeze in the U-Net (0=time/label embed, 1-9=input, 10-12=middle, 13-21=output, 22=out conv) / U-Net内で学習しないブロック番号を指定（カンマ区切り）",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )

    parser.add_argument(
        "--flow_model",
        action="store_true",
        help="enable Rectified Flow training objective instead of standard diffusion / 通常の拡散ではなくRectified Flowで学習する",
    )
    parser.add_argument(
        "--flow_use_ot",
        action="store_true",
        help="pair latents and noise with cosine optimal transport when using Rectified Flow / Rectified Flow使用時にOTでlatentとノイズを対応付ける",
    )
    parser.add_argument(
        "--flow_timestep_distribution",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="sampling distribution over Rectified Flow sigmas (default: logit_normal) / Rectified Flowのシグマの分布（デフォルトlogit_normal）",
    )
    parser.add_argument(
        "--flow_logit_mean",
        type=float,
        default=0.0,
        help="mean of the logit-normal distribution when using Rectified Flow / Rectified Flowでlogit-normal分布を用いるときの平均値",
    )
    parser.add_argument(
        "--flow_logit_std",
        type=float,
        default=1.0,
        help="stddev of the logit-normal distribution when using Rectified Flow / Rectified Flowでlogit-normal分布を用いるときの標準偏差",
    )
    parser.add_argument(
        "--flow_uniform_shift",
        action="store_true",
        help="apply resolution-dependent shift to Rectified Flow timesteps (SD3-style) / Rectified Flowタイムステップに解像度依存のシフトを適用する",
    )
    parser.add_argument(
        "--flow_uniform_base_pixels",
        type=float,
        default=1024.0 * 1024.0,
        help="reference pixel count used for the resolution-dependent timestep shift / タイムステップシフトで使用する基準ピクセル数",
    )
    parser.add_argument(
        "--flow_uniform_static_ratio",
        type=float,
        default=None,
        help="use a fixed sqrt(m/n) ratio (e.g. 2.5) for Rectified Flow timestep shift; overrides resolution-based shift / 一定のsqrt(m/n)比率（例:2.5）でRectified Flowタイムステップをシフトする（解像度依存シフトを上書き）",
    )
    parser.add_argument(
        "--contrastive_flow_matching",
        action="store_true",
        help="Enable Contrastive Flow Matching (ΔFM) objective. Works with v-parameterization or Rectified Flow.",
    )
    parser.add_argument(
        "--cfm_lambda",
        type=float,
        default=0.05,
        help="Lambda weight for the contrastive term in ΔFM loss (default: 0.05).",
    )
    parser.add_argument(
        "--use_zero_cond_dropout",
        type=bool,
        default=False,
        help="For full caption dropout, use zero conditioning instead of empty caption"
    )

    parser.add_argument(
        "--edm2_loss_weighting",
        action="store_true",
        help="Use EDM2 loss weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_laplace",
        action="store_true",
        help="Use EDM2 loss weighting to calculate timestep sampling using laplace.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer",
        type=str,
        default="torch.optim.AdamW",
        help="Fully qualified optimizer class name to use with the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer_lr",
        type=float,
        default=2e-2,
        help="Learning rate as a float for the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer_args",
        type=str,
        default=r"{'weight_decay': 0, 'betas': (0.9,0.999)}",
        help="A JSON object as a string of optimizer args for the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler",
        action="store_true",
        help="Use lr scheduler with EDM2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_warmup_percent",
        type=float,
        default=0.1,
        help="Percent of training steps to use for warmup.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_constant_percent",
        type=float,
        default=0.1,
        help="Percent of training steps to maintain constant LR before decay.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph",
        action="store_true",
        help="Enable generation of graph images that show the loss weighting per timestep.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_every_x_steps",
        type=int,
        default=20,
        help="Every x steps generate a graph image.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_output_dir",
        type=str,
        default=None,
        help="""The parent directory where loss weighting graph images should be stored, 
        with sub directories automatically created and named after the model's defined name.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_y_limit",
        type=int,
        default=None,
        help="""Set the max limit of the y axis, if not set, uses dynamic scaling of the y-axis, which can make it harder to follow. 
        6 is a good value for v-pred + ztsnr without any augmentation (i.e. low min snr gamma, debiased loss, or scaled v-pred loss). 
        If any of the noted augmentations are used, weighting values can reach ~100-150.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_y_scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="""Select between linear or log scaling for the y-axis.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_num_channels",
        type=int,
        default=128,
        help="The number of channels used by for the loss weighting module. Additional channels allows for greater granularity in the weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_initial_weights",
        type=str,
        default=None,
        help="The full filepath to initial weights and state of edm2 weighting model to use instead of random.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_decay_scaling",
        type=float,
        default=1.0,
        help="A scaling factor to apply to the decay rate of the edm2_loss_weighting_lr_scheduler, lower values result in slower decay, higher values result in faster decay.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_weighting",
        action="store_true",
        help="If edm2 loss scaling weights are weighted by importance, which is based using a specific min snr gamma value and SNR for the given timestep. " \
        "Default behavior when edm2_loss_weighting_importance_weighting is enabled is to disable normal min snr gamma and debiased loss if enabled." \
        "It is not advised to stack with either, as there is a possiblity of loss curving to 0 as SNR approaches 0." \
        "If you still wish to, set edm2_loss_weighting_importance_weighting_safety_override=True at your own risk."
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_weighting_max",
        type=float,
        default=10.0,
        help="The max loss weighting/scaling to apply when using edm2 importance weighting, has no effect otherwise.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_min_snr_gamma",
        type=float,
        default=1.0,
        help="The min snr gamma used for edm2 importance weighting as a heuristic, has no effect if not using importance weighting. " \
        "Not related to the typical application of min snr gamma.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_weighting_safety_override",
        action="store_true",
        help="At your own risk, you may set this to true to ALLOW stacking debiased loss and/or typical min snr gamma with EDM2 using importance weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_max_grad_norm",
        type=float,
        default=None,
        help="Maximum gradient norm for EDM2 loss weighting model. If not specified, uses --max_grad_norm value. Set to 0 to disable clipping for EDM2. / EDM2損失重み付けモデルの最大勾配ノルム。指定しない場合は--max_grad_normの値を使用。0に設定するとEDM2のクリッピングを無効化。"
    )

    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します",
    )

    parser.add_argument(
        "--validation_timesteps",
        type=str,
        default=r"[50, 350, 500, 650, 950]",
        help="A list of timesteps to use for each validation step."
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)