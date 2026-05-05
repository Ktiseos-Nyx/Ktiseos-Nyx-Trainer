import gc
import importlib
import argparse
import math
import os
import typing
from typing import Any, List, Union, Optional
import sys
import random
import time
import json
from multiprocessing import Value
import numpy as np
import ast
import itertools

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device
from library.edm2_loss_utils import prepare_edm2_loss_weighting, plot_edm2_loss_weighting_check, plot_edm2_loss_weighting
from library.ramtorch_util import apply_ramtorch_to_module

init_ipex()


from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, sai_model_spec, strategy_base, strategy_sd, sai_model_spec
from library.strategy_sdxl import SdxlTextEncodingStrategy

import library.train_util as train_util
from library.train_util import DreamBoothDataset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

# T-LoRA timestep-dependent rank masking support
try:
    from lycoris.modules.tlora import set_timestep_mask, clear_timestep_mask, compute_timestep_mask, compute_timestep_mask_batch
    TLORA_AVAILABLE = True
except ImportError:
    TLORA_AVAILABLE = False

setup_logging()
import logging

logger = logging.getLogger(__name__)


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False
        self.latent_shift = 0.0

        # T-LoRA timestep-dependent rank masking config
        self.tlora_enabled = False
        self.tlora_max_rank = 0
        self.tlora_min_rank = 1
        self.tlora_mask_alpha = 1.0
        self.tlora_max_timestep = 1000

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
        edm2_lr_scheduler=None,
        current_loss_scaled=None, 
        average_loss_scaled=None, 
        current_loss_edm2=None, 
        average_loss_edm2=None, 
        current_val_loss=None,
        average_val_loss=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if current_loss_scaled is not None:
            logs["loss/current_scaled"] = current_loss_scaled
            logs["loss/average_scaled"] = average_loss_scaled

        if current_loss_edm2 is not None:
            logs["loss/current_edm2"] = current_loss_edm2
            logs["loss/average_edm2"] = average_loss_edm2

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        if current_val_loss is not None:
            logs["loss/current_val_loss"] = current_val_loss                      
            logs["loss/average_val_loss"] = average_val_loss

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else 1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{i}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().startswith("Prodigy".lower()):
                opt = lr_scheduler.optimizers[-1] if hasattr(lr_scheduler, "optimizers") else optimizer
                if opt is not None:
                    logs[f"lr/d*lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["lr"]
                    if "effective_lr" in opt.param_groups[i]:
                        logs[f"lr/d*eff_lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["effective_lr"]

        if edm2_lr_scheduler is not None:
            logs[f"lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        train_dataset_group.verify_bucket_reso_steps(64)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        if args.use_ramtorch:
            logger.info("Applying RamTorch to SD model.")
            unet = apply_ramtorch_to_module(unet, "unet", accelerator.device, weight_dtype)
            text_encoder = apply_ramtorch_to_module(text_encoder, "clip_l", accelerator.device, weight_dtype)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, List[nn.Module]]:
        raise NotImplementedError()

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def get_flow_pixel_counts(self, args, batch, latents):
        return None

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, text_masks, batch, weight_dtype, **kwargs):
        noisy_latents = noisy_latents.to(weight_dtype)
        noise_pred = unet(noisy_latents, timesteps, text_conds[0], text_masks).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def all_reduce_edm2_model(self, accelerator, edm2_model):
        """Manually synchronize EDM2 model gradients across GPUs."""
        if edm2_model is None:
            return
        for param in edm2_model.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, device)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return (latents - self.latent_shift) * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        encoder_attention_mask_bias = text_encoder_masks[1] #[(1 - t.to(dtype=text_encoder_conds[0].dtype)).unsqueeze(1) * -10000.0 for t in text_encoder_masks]

        pixel_counts = None
        if hasattr(self, "get_flow_pixel_counts"):
            pixel_counts = self.get_flow_pixel_counts(args, batch, latents.device)

        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
            args, noise_scheduler, latents, fixed_timesteps=fixed_timesteps, is_train=is_train, pixel_counts=pixel_counts
        )

        # ensure the hidden state will require grad
        if is_train and args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Set T-LoRA timestep mask before the forward pass
        self.apply_tlora_mask(timesteps)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                encoder_attention_mask_bias,
                batch,
                weight_dtype,
            )

        # Clear T-LoRA mask after the forward pass
        self.clear_tlora_mask_if_needed()

        # Upcast for grokking
        latents = latents.to(torch.float64)
        noise = noise.to(torch.float64)

        if getattr(args, "flow_model", False):
            target = noise - latents
        elif args.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        args,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        encoder_attention_mask_bias,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None, noise

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        if args.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        pass

    def is_text_encoder_not_needed_for_training(self, args):
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    def setup_tlora_masking(self, net_kwargs, network_dim, noise_scheduler):
        """
        Initialize T-LoRA timestep masking if the algo is tlora.

        Reads tlora_min_rank and tlora_mask_alpha from network_args.
        Must be called after the network is created.
        """
        algo = (net_kwargs.get("algo", "lora") or "lora").lower()
        if algo != "tlora":
            return
        if not TLORA_AVAILABLE:
            logger.warning("T-LoRA requested but lyco_tlora is not available. Skipping T-LoRA setup.")
            return


        self.tlora_enabled = True
        self.tlora_max_rank = int(network_dim) if network_dim is not None else 4
        tlora_min_rank = net_kwargs.get("tlora_min_rank", None)
        if tlora_min_rank is None:
            self.tlora_min_rank = int(math.ceil(self.tlora_max_rank * 0.5))
        else:
            self.tlora_min_rank = int(tlora_min_rank)
        self.tlora_min_rank = max(0, min(self.tlora_min_rank, self.tlora_max_rank))
        self.tlora_mask_alpha = float(net_kwargs.get("tlora_mask_alpha", 1.0))
        self.tlora_max_timestep = noise_scheduler.config.num_train_timesteps
        logger.info(
            f"T-LoRA masking enabled: max_rank={self.tlora_max_rank}, "
            f"min_rank={self.tlora_min_rank}, mask_alpha={self.tlora_mask_alpha}, "
            f"max_timestep={self.tlora_max_timestep}"
        )

    def apply_tlora_mask(self, timesteps: torch.Tensor):
        """
        Compute and set the T-LoRA timestep mask for the current batch.

        Computes a per-sample mask so each sample in the batch gets a rank
        mask matching its own noise level, avoiding the bias of a single
        shared mask (e.g. max timestep penalizing low-noise samples).
        For batch_size=1 (common in LoRA training), this is equivalent to
        computing a single mask.
        """
        if not self.tlora_enabled:
            return

        if timesteps.numel() == 1:
            # Fast path for batch_size=1: avoid tensor ops
            mask = compute_timestep_mask(
                timestep=int(timesteps.item()),
                max_timestep=self.tlora_max_timestep,
                max_rank=self.tlora_max_rank,
                min_rank=self.tlora_min_rank,
                alpha=self.tlora_mask_alpha,
            )
        else:
            # Per-sample masks: shape (B, max_rank)
            mask = compute_timestep_mask_batch(
                timesteps=timesteps,
                max_timestep=self.tlora_max_timestep,
                max_rank=self.tlora_max_rank,
                min_rank=self.tlora_min_rank,
                alpha=self.tlora_mask_alpha,
            )
        set_timestep_mask(mask)

    def clear_tlora_mask_if_needed(self):
        """Clear the T-LoRA mask after the forward pass."""
        if not self.tlora_enabled:
            return
        clear_timestep_mask()

    # endregion

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
        edm2_model=None
    ) -> torch.Tensor:
        """
        Process a batch for the network
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(device=accelerator.device))
            else:
                # latentに変換
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(device=accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(args, vae, chunk.to(device=accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

        text_encoder_conds = []
        masks_reshaped = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
            if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                masks_reshaped = text_encoder_outputs_list[3:]

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                # Get the text embedding for conditioning
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                        masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                        encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids,
                            attn_masks=masks,
                        )
                    else:
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids
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

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting, noise = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            masks_reshaped,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

        # Cast to float64 (Double Precision) for Grokking
        noise_pred = noise_pred.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)

        if is_train:
            if args.differential_guidance:
                target = noise_pred + (float(args.differential_guidance_scale) * (target - noise_pred))

            huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
            loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))
            if weighting is not None:
                loss = loss * weighting

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
        else:
                loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch

        if is_train:
            loss_weights = batch["loss_weights"]  # 各sampleごとのweight
            loss = loss * loss_weights
            loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        if is_train and args.loss_multiplier:
            loss.mul_(float(args.loss_multiplier) if args.loss_multiplier is not None else 1.0)

        # For logging
        pre_scaling_loss = loss.mean()

        if is_train and args.edm2_loss_weighting:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        return loss.mean(), pre_scaling_loss, loss_scaled
    
    def process_val_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        train_text_encoder=True,
        train_unet=True,
        timesteps_list: list = [50, 350, 500, 650, 950]
    ) -> torch.Tensor:
        """
        Process a batch for the network to determine val loss
        """
        total_loss = 0.0 
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(device=accelerator.device))
            else:
                # latentに変換
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(device=accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        chunk = self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                        list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

            text_encoder_conds = []
            masks_reshaped = []
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
                if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                    masks_reshaped = text_encoder_outputs_list[3:]

            if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
                with accelerator.autocast():
                    # Get the text embedding for conditioning
                    if args.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids_list,
                            weights_list,
                        )
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                        if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                            masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                            encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                input_ids,
                                attn_masks=masks,
                            )
                        else:
                            encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                input_ids
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

            batch_size = latents.shape[0]
            for fixed_timesteps in timesteps_list:
                timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)
                # sample noise, call unet, get target
                noise_pred, target, _, _, _ = self.get_noise_pred_and_target(
                    args,
                    accelerator,
                    noise_scheduler,
                    latents,
                    batch,
                    text_encoder_conds,
                    masks_reshaped,
                    unet,
                    network,
                    weight_dtype,
                    train_unet,
                    fixed_timesteps=timesteps,
                    is_train=False,
                )

                # Cast to float64 (Double Precision) for Grokking
                noise_pred = noise_pred.to(dtype=torch.float64)
                target = target.to(dtype=torch.float64)

                loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)
                loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch
                loss = loss.mean()
                total_loss += loss

        average_loss = total_loss / len(timesteps_list)    

        return average_loss

    def cast_text_encoder(self, args):
        return True  # default for other than HunyuanImage

    def cast_vae(self, args):
        return True  # default for other than HunyuanImage

    def cast_unet(self, args):
        return True  # default for other than HunyuanImage

    def switch_rng_state(self, val_seed: int, accelerator) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        cpu_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        if accelerator.device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state()
        elif accelerator.device.type == "xpu":
            gpu_rng_state = torch.xpu.get_rng_state()
        elif accelerator.device.type == "mps":
            gpu_rng_state = torch.cuda.get_rng_state()
        else:
            gpu_rng_state = None

        random.seed(val_seed)
        np.random.seed(val_seed)
        torch.manual_seed(val_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(val_seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)

    def restore_rng_state(self, rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple], accelerator):
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
                torch.cuda.set_rng_state(gpu_rng_state)

    def calculate_val_loss(self, 
                           global_step,
                           epoch_step,
                           train_dataloader,
                           val_loss_recorder,
                           val_dataloader,
                           cyclic_val_dataloader,
                           network, 
                           tokenize_strategy, 
                           text_encoders, 
                           text_encoding_strategy, 
                           unet, 
                           vae, 
                           noise_scheduler, 
                           vae_dtype, 
                           weight_dtype, 
                           accelerator, 
                           args, 
                           epoch,
                           batch=None,
                           train_text_encoder=True):
        if not train_util.calculate_val_loss_check(args,global_step,epoch_step,val_dataloader,train_dataloader):
            return None, None, None
        
        if batch is not None:
            self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)
   
        rng_states = self.switch_rng_state(int(args.validation_seed) if args.validation_seed else 23, accelerator)

        timesteps_list = ast.literal_eval(args.validation_timesteps)
              
        accelerator.print("") 
        accelerator.print("Validating バリデーション処理...")
        total_loss = 0.0
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
                loss = self.process_val_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, 
                                              weight_dtype, accelerator, args, text_encoding_strategy, tokenize_strategy, 
                                              train_text_encoder=train_text_encoder,
                                              timesteps_list=timesteps_list)
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)   
                     
        average_val_loss: float = val_loss_recorder.average
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

        self.restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss, logs


    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        train_util.set_torch_cuda_reduced_precision(args)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        if getattr(args, "flow_model", False):
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
            distribution = getattr(args, "flow_timestep_distribution", "logit_normal")
            if distribution == "logit_normal":
                flow_logit_std = float(getattr(args, "flow_logit_std", 1.0))
                flow_logit_mean = float(getattr(args, "flow_logit_mean", 0.0))
                if flow_logit_std == 0:
                    raise ValueError("`--flow_logit_std` must be non-zero.")
                logger.info(
                    "Rectified Flow timesteps sampled from logit-normal distribution with "
                    f"mean={flow_logit_mean}, std={flow_logit_std}."
                )
            elif distribution == "uniform":
                logger.info("Rectified Flow timesteps sampled uniformly in [0, 1].")
            else:
                raise ValueError(f"Unknown Rectified Flow timestep distribution: {distribution}")

            if shift_enabled:
                if args.flow_uniform_static_ratio is not None:
                    flow_uniform_static_ratio = float(getattr(args, "flow_uniform_static_ratio", 0.0))
                    if flow_uniform_static_ratio <= 0:
                        raise ValueError("`--flow_uniform_static_ratio` must be positive.")
                    logger.info(
                        f"Rectified Flow timestep shift uses static ratio={flow_uniform_static_ratio}."
                    )
                else:
                    logger.info(
                        f"Rectified Flow timestep shift uses base pixels={args.flow_uniform_base_pixels}."
                    )

        if args.contrastive_flow_matching and not (args.v_parameterization or getattr(args, "flow_model", False)):
            raise ValueError("`--contrastive_flow_matching` requires either v-parameterization or Rectified Flow.")

        if getattr(args, "vae_custom_scale", None) is not None:
            try:
                self.vae_scale_factor = float(args.vae_custom_scale)
            except (TypeError, ValueError):
                raise ValueError("`--vae_custom_scale` must be a valid number")
            logger.info(f"Using custom VAE scale factor: {self.vae_scale_factor}")
        if getattr(args, "vae_custom_shift", None) is not None:
            try:
                self.latent_shift = float(args.vae_custom_shift)
            except (TypeError, ValueError):
                raise ValueError("`--vae_custom_shift` must be a valid number")
            logger.info(f"Using custom VAE shift factor: {self.latent_shift}")
        else:
            self.latent_shift = 0.0

        args.vae_scale_factor = self.vae_scale_factor
        args.vae_shift_factor = self.latent_shift

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        train_util.args_set_seed(args)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
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
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

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

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                train_util.debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
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

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)  # may change some args

        # acceleratorを準備する
        logger.info(f"preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        logger.info(f"prepared accelerator on {accelerator.device}")
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if self.cast_vae(args) else None

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        if vae_dtype is None:
            vae_dtype = vae.dtype
            logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

        if getattr(args, "vae_reflection_padding", False):
            vae = model_util.use_reflection_padding(vae)

        if args.use_ramtorch_vae:
            vae = apply_ramtorch_to_module(vae, "vae", accelerator.device, vae_dtype)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # prepare dataset for latents caching if needed
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

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None and not isinstance(args.base_weights, list):
            args.base_weights = [args.base_weights]

        if args.base_weights_multiplier is not None and not isinstance(args.base_weights_multiplier, list):
            args.base_weights_multiplier = [float(x) for x in [args.base_weights_multiplier]]

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        network_has_multiplier = hasattr(network, "set_multiplier")

        # TODO remove `hasattr` by setting up methods if not defined in the network like below  (hacky but will work):
        # if not hasattr(network, "prepare_network"):
        #    network.prepare_network = lambda args: None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.use_ramtorch_network:
            #move all network weights to cpu first as base device
            network = network.to("cpu")
            logger.info("Applying RamTorch to network/lora.")
            network = apply_ramtorch_to_module(network, "network/lora", accelerator.device)
            # Make sure the rest of the network is moved to the accelerator.device
            network = network.to(accelerator.device)

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        (
            optimizer_name, 
            optimizer_args, 
            optimizer, 
            optimizer_train_fn, 
            optimizer_eval_fn, 
            lr_descriptions, 
            text_encoder_lr
         ) = train_util.prepare_optimizer(args, network, accelerator)

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
            pin_memory=args.pin_data_loader_memory or args.pin_memory,
        )

        if val_dataset_group is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group if val_dataset_group is not None else [],
                shuffle=False,
                batch_size=1,
                collate_fn=collator,
                num_workers=n_workers,
                persistent_workers=args.persistent_data_loader_workers,
                pin_memory=args.pin_data_loader_memory or args.pin_memory,
            )

        if val_dataset_group is not None:
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
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn

            if not args.fp8_base_unet:
                accelerator.print("enable fp8 training for Text Encoder.")
            te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn

            # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
            # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

            # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
            # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
            logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
            unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

        unet.requires_grad_(False)
        if self.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            flags = self.get_text_encoders_train_flags(args, text_encoders)
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if flags[0] else None,
                text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                # default implementation is:  unet = accelerator.prepare(unet)
                unet = self.prepare_unet_with_accelerator(args, accelerator, unet)  # accelerator does some magic here
            else:
                # move to device because unet is not prepared by accelerator
                unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(args) else None)
            if train_text_encoder:
                text_encoders = [
                    (accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
                ]
                if len(text_encoders) > 1:
                    text_encoder = text_encoders
                else:
                    text_encoder = text_encoders[0]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
            training_model = network

        if val_dataset_group is not None:
            val_dataloader = accelerator.prepare(val_dataloader)
            cyclic_val_dataloader = itertools.cycle(val_dataloader)
        else:
            val_dataloader, cyclic_val_dataloader = None, None

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(
            f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
        )
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_scale": args.huber_scale,
            "ss_huber_c": args.huber_c,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_fp8_base_unet": bool(args.fp8_base_unet),
            "ss_validation_seed": args.validation_seed,
            "ss_validation_split": float(args.validation_split),
            "ss_max_validation_steps": args.max_validation_steps,
            "ss_validate_every_n_epochs": args.validate_every_n_epochs,
            "ss_validate_every_n_steps": args.validate_every_n_steps,
            "ss_resize_interpolation": args.resize_interpolation,
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "skip_image_resolution": dataset.skip_image_resolution,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                    "resize_interpolation": dataset.resize_interpolation,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "random_crop_padding_percent": float(getattr(subset, "random_crop_padding_percent", 0.05)),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                        "resize_interpolation": subset.resize_interpolation,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        subset_metadata["is_val"] = subset.is_val
                        if subset.is_reg or subset.is_val:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            val_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    if subset.is_reg:
                        info = reg_dataset_dirs_info
                    elif subset.is_val:
                        info = val_dataset_dirs_info
                    else:
                        info = dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_random_crop_padding_percent": float(getattr(args, "random_crop_padding_percent", 0.05)),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_multires_training": bool(getattr(dataset, "multires_training", False)),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_skip_image_resolution": dataset.skip_image_resolution,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata["ss_vae_scale_factor"] = self.vae_scale_factor
        metadata["ss_vae_shift_factor"] = self.latent_shift
        metadata["ss_vae_reflection_padding"] = getattr(args, "vae_reflection_padding", False)

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by gradient_accumulation_steps (dataloader len is already divided by num_processes)
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        resumed_step = initial_step
        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                epoch_to_start = resumed_step // update_steps_per_epoch
                # Calculate exactly how many batches to skip in the starting epoch
                initial_step = (resumed_step % update_steps_per_epoch) * args.gradient_accumulation_steps
                logger.info(f"skipping {epoch_to_start} epochs and {initial_step} batches / {epoch_to_start}エポックと{initial_step}バッチをスキップします")
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = resumed_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = resumed_step

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        # Initialize T-LoRA timestep masking if applicable
        self.setup_tlora_masking(net_kwargs, args.network_dim, noise_scheduler)

        edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(args, noise_scheduler, accelerator)

        train_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = train_util.EMARecorder()
        val_loss_recorder = train_util.EMARecorder()

        if args.edm2_loss_weighting:
            loss_scaled_recorder = train_util.EMARecorder()
            loss_edm2_recorder = train_util.EMARecorder()

        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start_for_network = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False, dtype_override=None):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, dtype_override or save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # if text_encoder is not needed for training, delete it to save memory.
        # TODO this can be automated after SDXL sample prompt cache is implemented
        if self.is_text_encoder_not_needed_for_training(args):
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        current_val_loss, average_val_loss, val_logs = None, None, {}
        keys_scaled, mean_norm, maximum_norm = None, None, None
        mean_grad_norm, mean_combined_norm = None, None
        max_mean_logs = {}
        current_global_step_loss = 0.0
        current_global_step_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        average_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        current_global_step_loss_edm2 = 0.0 if args.edm2_loss_weighting else None
        average_loss_edm2 = 0.0 if args.edm2_loss_weighting else None
        avr_loss = 0.0
        accumulation_counter = 0

        # For --sample_at_first
        if train_util.sample_images_check(args, 0, global_step) or train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
            with torch.no_grad():
                #Switch network to eval mode
                accelerator.unwrap_model(network).eval()
                if args.gradient_checkpointing:
                    accelerator.unwrap_model(unet).eval()
                    for t_enc in text_encoders:
                        accelerator.unwrap_model(t_enc).eval()

                optimizer_eval_fn()
                self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                if train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
                    current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(
                        global_step, 0, train_dataloader, val_loss_recorder, val_dataloader, 
                        cyclic_val_dataloader, network, tokenize_strategy, 
                        text_encoders, text_encoding_strategy, unet, vae, noise_scheduler, 
                        vae_dtype, weight_dtype, accelerator, args, 0, None, train_text_encoder)
                #Switch network to train mode
                optimizer_train_fn()
                accelerator.unwrap_model(network).train()
                if args.gradient_checkpointing:
                    accelerator.unwrap_model(unet).train()
                    for t_enc in text_encoders:
                        accelerator.unwrap_model(t_enc).train()

        if plot_edm2_loss_weighting_check(args, global_step):
            plot_edm2_loss_weighting(args, global_step, edm2_model, 1000, accelerator.device)

        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            logs = self.generate_step_logs(
                args,
                current_global_step_loss,
                avr_loss,
                lr_scheduler,
                lr_descriptions,
                optimizer,
                keys_scaled,
                mean_norm,
                maximum_norm,
                mean_grad_norm,
                mean_combined_norm,
                edm2_lr_scheduler,
                current_global_step_loss_scaled,
                average_loss_scaled,
                current_global_step_loss_edm2,
                average_loss_edm2,
                current_val_loss=current_val_loss, 
                average_val_loss=average_val_loss
            )
            # log empty object to commit the sample images to wandb
            accelerator.log(logs, step=0) 

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            logger.info(f"skipping {initial_step} batches in the first epoch / 最初の{epoch_to_start}エポック内で{initial_step}バッチをスキップします")

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(args.max_train_steps - global_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        for epoch in range(epoch_to_start, num_train_epochs):
            current_epoch.value = epoch + 1
            accelerator.print(f"\nepoch {current_epoch.value}/{num_train_epochs}\n")

            metadata["ss_epoch"] = str(current_epoch.value)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step

                if initial_step > 0:
                    initial_step -= 1
                    continue

                with train_util.determine_grad_sync_context(args, accelerator, None, training_model, edm2_model):
                    on_step_start_for_network(text_encoder, unet)

                    accumulation_counter += 1

                    # preprocess batch for each model
                    self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                    loss, pre_scaling_loss, loss_scaled = self.process_batch(
                        batch,
                        text_encoders,
                        unet,
                        network,
                        vae,
                        noise_scheduler,
                        vae_dtype,
                        weight_dtype,
                        accelerator,
                        args,
                        text_encoding_strategy,
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                        edm2_model=edm2_model,
                    )

                    if loss.ndim != 0:
                        loss = loss.mean()

                    accelerator.backward(loss)

                    if args.use_ramtorch or args.use_ramtorch_network:
                        torch.cuda.synchronize() 

                    edm2_loss = loss
                    loss = pre_scaling_loss

                    if accelerator.sync_gradients:
                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually

                        # Sync and clip EDM2 gradients
                        if args.edm2_loss_weighting:
                            self.all_reduce_edm2_model(accelerator, edm2_model)

                        if args.max_grad_norm != 0.0 or args.edm2_loss_weighting:
                            accelerator.unscale_gradients()

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if args.edm2_loss_weighting:
                            # Use edm2-specific grad norm if provided, otherwise use max_grad_norm
                            edm2_grad_norm = (args.edm2_loss_weighting_max_grad_norm
                                             if args.edm2_loss_weighting_max_grad_norm is not None
                                             else args.max_grad_norm)
                            if edm2_grad_norm != 0.0:
                                edm2_params = list(accelerator.unwrap_model(edm2_model).parameters())
                                torch.nn.utils.clip_grad_norm_(edm2_params, edm2_grad_norm)

                        #if hasattr(network, "update_grad_norms"):
                        #    network.update_grad_norms()
                        #if hasattr(network, "update_norms"):
                        #    network.update_norms()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if args.edm2_loss_weighting:
                        edm2_optimizer.step()
                        edm2_lr_scheduler.step()
                        # swap to pre_scaling_loss for logging
                        edm2_optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms and accelerator.sync_gradients:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    mean_grad_norm = None
                    mean_combined_norm = None
                    mean_norm = mean_norm.detach().item() if isinstance(mean_norm, torch.Tensor) else mean_norm
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    #if hasattr(network, "weight_norms"):
                    #    weight_norms = network.weight_norms()
                    #    mean_norm = weight_norms.mean().item() if weight_norms is not None else None
                    #    grad_norms = network.grad_norms()
                    #    mean_grad_norm = grad_norms.mean().item() if grad_norms is not None else None
                    #    combined_weight_norms = network.combined_weight_norms()
                    #    mean_combined_norm = combined_weight_norms.mean().item() if combined_weight_norms is not None else None
                    #    maximum_norm = weight_norms.max().item() if weight_norms is not None and weight_norms.numel() > 0 else None
                    #    keys_scaled = None
                    #    max_mean_logs = {}
                    # else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if (train_util.sample_images_check(args, None, global_step) or 
                        train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader) or 
                        args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0):
                        with torch.no_grad():
                            accelerator.unwrap_model(network).eval()
                            if args.gradient_checkpointing:
                                accelerator.unwrap_model(unet).eval()
                                for t_enc in text_encoders:
                                    accelerator.unwrap_model(t_enc).eval()

                            optimizer_eval_fn()
                            self.sample_images(
                                accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                            )

                            if train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader):
                                current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(global_step, step, 
                                                                                                        skipped_dataloader or train_dataloader, 
                                                                                                        val_loss_recorder, 
                                                                                                        val_dataloader, 
                                                                                                        cyclic_val_dataloader, 
                                                                                                        network,
                                                                                                        tokenize_strategy, 
                                                                                                        text_encoders, 
                                                                                                        text_encoding_strategy, 
                                                                                                        unet, 
                                                                                                        vae, 
                                                                                                        noise_scheduler, 
                                                                                                        vae_dtype, 
                                                                                                        weight_dtype, 
                                                                                                        accelerator, 
                                                                                                        args, 
                                                                                                        current_epoch.value,
                                                                                                        batch,
                                                                                                        train_text_encoder)
                            else:
                                current_val_loss, average_val_loss, val_logs = None, None, {}
                            progress_bar.unpause()

                            # 指定ステップごとにモデルを保存
                            if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                    if args.edm2_loss_weighting:
                                        loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step, "_edm2_loss_weights")
                                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, epoch, dtype_override=torch.float32)

                                    if args.save_state:
                                        train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                                    remove_step_no = train_util.get_remove_step_no(args, global_step)
                                    if remove_step_no is not None:
                                        remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                        remove_model(remove_ckpt_name)

                                        if args.edm2_loss_weighting:
                                            remove_loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no, "_edm2_loss_weights")
                                            remove_model(remove_loss_weights_ckpt_name)

                            optimizer_train_fn()
                            accelerator.unwrap_model(network).train()
                            if args.gradient_checkpointing:
                                accelerator.unwrap_model(unet).train()
                                for t_enc in text_encoders:
                                    accelerator.unwrap_model(t_enc).train()
                    else:
                        current_val_loss, average_val_loss, val_logs = None, None, None

                    # EDM2 graph generation - moved outside the sample/val/save conditional
                    if plot_edm2_loss_weighting_check(args, global_step):
                        plot_edm2_loss_weighting(args, global_step, edm2_model, 1000, accelerator.device)

                current_global_step_loss += loss.detach().item()
                if args.edm2_loss_weighting:
                    current_global_step_loss_scaled += loss_scaled.detach().item()
                    current_global_step_loss_edm2 += edm2_loss.detach().item()
                else:
                    current_global_step_loss_scaled = None
                    current_global_step_loss_edm2 = None

                if accelerator.sync_gradients:
                    loss_recorder.add(current_global_step_loss / accumulation_counter)
                    if args.edm2_loss_weighting:
                        loss_scaled_recorder.add(current_global_step_loss_scaled / accumulation_counter)
                        loss_edm2_recorder.add(current_global_step_loss_edm2 / accumulation_counter)
                        
                    avr_loss: float = loss_recorder.average
                    logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                    if is_tracking:
                        current_global_step_loss = (current_global_step_loss / accumulation_counter)
                        if args.edm2_loss_weighting:
                            current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                            average_loss_scaled: float = loss_scaled_recorder.average
                            current_global_step_loss_edm2 = (current_global_step_loss_edm2 / accumulation_counter)
                            average_loss_edm2: float = loss_edm2_recorder.average
                        else:
                            current_global_step_loss_scaled = None
                            average_loss_scaled = None
                            current_global_step_loss_edm2 = None
                            average_loss_edm2 = None

                        logs = self.generate_step_logs(
                            args,
                            current_global_step_loss,
                            avr_loss,
                            lr_scheduler,
                            lr_descriptions,
                            optimizer,
                            keys_scaled,
                            mean_norm,
                            maximum_norm,
                            mean_grad_norm,
                            mean_combined_norm,
                            edm2_lr_scheduler,
                            current_global_step_loss_scaled,
                            average_loss_scaled,
                            current_global_step_loss_edm2,
                            average_loss_edm2,
                            current_val_loss=current_val_loss, 
                            average_val_loss=average_val_loss
                        )
                        accelerator.log(logs, step=global_step)
                    current_global_step_loss = 0.0
                    if args.edm2_loss_weighting:
                        current_global_step_loss_scaled = 0.0
                        current_global_step_loss_edm2 = 0.0
                    accumulation_counter = 0

                if global_step >= args.max_train_steps:
                    break

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.average}
                accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()

            if (train_util.sample_images_check(args, current_epoch.value, global_step) or 
                args.save_every_n_epochs is not None):
                with torch.no_grad():
                    # 指定エポックごとにモデルを保存
                    optimizer_eval_fn()
                    accelerator.unwrap_model(network).eval()
                    if args.gradient_checkpointing:
                        accelerator.unwrap_model(unet).eval()
                        for t_enc in text_encoders:
                            accelerator.unwrap_model(t_enc).eval()

                    if args.save_every_n_epochs is not None:
                        saving = (current_epoch.value) % args.save_every_n_epochs == 0 and (current_epoch.value) < num_train_epochs
                        if is_main_process and saving:
                            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, current_epoch.value)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, current_epoch.value)

                            if args.edm2_loss_weighting:
                                loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, current_epoch.value, "_edm2_loss_weights")
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, current_epoch.value, dtype_override=torch.float32)

                            remove_epoch_no = train_util.get_remove_epoch_no(args, current_epoch.value)
                            if remove_epoch_no is not None:
                                remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                remove_model(remove_ckpt_name)

                                if args.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no, "_edm2_loss_weights")
                                    remove_model(remove_loss_weights_ckpt_name)

                            if args.save_state:
                                train_util.save_and_remove_state_on_epoch_end(args, accelerator, current_epoch.value)

                    self.sample_images(accelerator, args, current_epoch.value, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                    progress_bar.unpause()
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()
                    if args.gradient_checkpointing:
                        accelerator.unwrap_model(unet).train()
                        for t_enc in text_encoders:
                            accelerator.unwrap_model(t_enc).train()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            if args.edm2_loss_weighting:
                loss_weights_ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as, "_edm2_loss_weights")
                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, num_train_epochs, force_sync_upload=True, dtype_override=torch.float32)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument(
        "--disable_cross_attn_mask",
        action="store_true",
        help="Disable SDXL cross-attention masking so padded tokens participate normally / SDXLのcross-attentionマスク機能を無効化する",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
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

    parser.add_argument(
        "--use_ramtorch_network",
        action="store_true",
        help="Use RamTorch to reduce GPU memory usage by keeping network/lora linear weights in system RAM. " \
        "Requires use of optimizers that have been modified to support it, currently only SimplifiedAdEMAMix, SimplifiedAdEMAMixExM, and OCGOpt.",
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
        "--orthograd_targets",
        type=str,
        default=r"['lora_down.weight','lora_up.weight','lora_down1.weight','lora_up1.weight','lora_down2.weight','lora_up2.weight','a1.weight','a2.weight','b1.weight','b2.weight','c1.weight']",
        help="A list of strings to determine which named parameters should subject to orthgrad, based on their name containing the string."
    )

    parser.add_argument(
        "--pin_data_loader_memory",
        action="store_true",
        help="Pins dataloader memory, may speed up dataloader operations.",
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for faster GPU loading / GPU の読み込みを高速化するためのピンメモリ",
    )

    parser.add_argument(
        "--differential_guidance",
        action="store_true",
        help="Differential Guidance amplifies applies an amplification between the difference of the model prediction and the target during training to make " \
        "a new target. This may help improve convergence to the actual target. " \
        "See original code at https://github.com/ostris/ai-toolkit/commit/2e7b2d9926de40a7b9119322c1d8fc085b1283e4#diff-fb148217f864741f0e90717dc8ab38dff83a42e917a20540f65afb1c3aedaa85",
    )

    parser.add_argument(
        "--differential_guidance_scale",
        type=float,
        default=3.0,
        help="Differential Guidance Scale is used to determine the multiplier of the difference of the model prediction and the target. " \
        "--differential_guidance arg must be passed for this to be applied. " \
        "See original code at https://github.com/ostris/ai-toolkit/commit/2e7b2d9926de40a7b9119322c1d8fc085b1283e4#diff-fb148217f864741f0e90717dc8ab38dff83a42e917a20540f65afb1c3aedaa85",
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
        help="override the latent scaling factor applied after VAE encode / VAEエンコード後のスケーリング係数を上書きする",
    )
    parser.add_argument(
        "--vae_custom_shift",
        type=float,
        default=None,
        help="apply a constant latent shift before scaling (e.g. Flux-style offset) / スケーリング前に潜在表現へ定数シフトを適用する",
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
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
