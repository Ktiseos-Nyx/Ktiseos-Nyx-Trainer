"""
Bake LoRA(s) into an Anima base checkpoint (DiT + Qwen3 text encoder).

Mirrors the CLI shape of sdxl_merge_lora.py: --base_model --models --ratios --save_to.

Usage:
    python custom/anima_merge_lora.py \\
        --base_model path/to/anima_model.safetensors \\
        --text_encoder path/to/qwen3 \\
        --models path/to/lora1.safetensors path/to/lora2.safetensors \\
        --ratios 1.0 0.8 \\
        --save_to output/baked_anima.safetensors \\
        --device cpu
"""

import argparse
import logging
import os
import sys

SD_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trainer", "derrian_backend", "sd_scripts",
)
sys.path.insert(0, SD_SCRIPTS)

import torch
from safetensors.torch import save_file

from library import anima_utils
from networks import lora_anima

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake LoRA(s) into an Anima checkpoint")
    parser.add_argument("--base_model", type=str, required=True, help="Path to Anima DiT model (.safetensors)")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to Qwen3 text encoder (dir or .safetensors)")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="LoRA weight file(s)")
    parser.add_argument("--ratios", type=float, nargs="*", default=None, help="Per-LoRA multiplier (default: 1.0)")
    parser.add_argument("--save_to", type=str, required=True, help="Output path for baked DiT model")
    parser.add_argument("--save_text_encoder_to", type=str, default=None, help="Output path for baked TE (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    ratios = args.ratios
    if ratios is None:
        ratios = [1.0] * len(args.models)
    elif len(ratios) < len(args.models):
        ratios = list(ratios) + [1.0] * (len(args.models) - len(ratios))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading base DiT model from %s", args.base_model)
    dit = anima_utils.load_anima_model(
        device=device,
        dit_path=args.base_model,
        attn_mode="torch",
        split_attn=True,
        loading_device="cpu",
        dit_weight_dtype=dtype,
        fp8_scaled=False,
    )

    logger.info("Loading text encoder from %s", args.text_encoder)
    te, _tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path=args.text_encoder,
        dtype=dtype,
        device="cpu",
    )

    for i, (lora_path, mult) in enumerate(zip(args.models, ratios)):
        logger.info("Baking LoRA %d/%d: %s (ratio=%.2f)", i + 1, len(args.models), lora_path, mult)

        lora_sd = {}
        if lora_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            lora_sd = load_file(lora_path)
        else:
            lora_sd = torch.load(lora_path, map_location="cpu", weights_only=True)

        network, _weights_sd = lora_anima.create_network_from_weights(
            multiplier=mult,
            file=None,
            ae=None,
            text_encoders=[te],
            unet=dit,
            weights_sd=lora_sd,
            for_inference=True,
        )

        network.merge_to(
            text_encoders=[te],
            unet=dit,
            weights_sd=lora_sd,
            dtype=dtype,
            device="cpu",
        )

        del network, lora_sd

    logger.info("Saving baked DiT model to %s", args.save_to)
    dit_state_dict = dit.state_dict()
    anima_utils.save_anima_model(
        args.save_to,
        dit_state_dict,
        metadata={"anima_merge": "true", "format": "pt"},
        dtype=dtype,
    )

    te_save_path = args.save_text_encoder_to
    if te_save_path is None:
        base, ext = os.path.splitext(args.save_to)
        te_save_path = f"{base}_te.safetensors"

    logger.info("Saving baked text encoder to %s", te_save_path)
    te_state_dict = te.state_dict()
    prefixed_te = {f"model.{k}": v.detach().clone().to("cpu").to(dtype).contiguous() for k, v in te_state_dict.items()}
    save_file(prefixed_te, te_save_path, metadata={"anima_merge": "true", "format": "pt"})

    logger.info("Done — baked %d LoRA(s) into Anima checkpoint", len(args.models))


if __name__ == "__main__":
    main()
