from pathlib import Path
import json

from library.train_util import BucketManager
from PIL import Image
import math


def validate(args: dict) -> tuple[bool, bool, list[str], dict, dict]:
    over_errors = []
    if "args" not in args:
        over_errors.append("args is not present")
    if "dataset" not in args:
        over_errors.append("dataset is not present")
    if over_errors:
        return False, False, over_errors, {}, {}
    args_pass, args_errors, args_data = validate_args(args["args"])
    dataset_pass, dataset_errors, dataset_data = validate_dataset_args(args["dataset"])
    over_pass = args_pass and dataset_pass
    over_errors = args_errors + dataset_errors

    # Process log prefix mode regardless of initial pass status, but add errors if needed
    if "log_prefix_mode" in args_data:
        mode = args_data["log_prefix_mode"]
        if mode == "output_name":
            # Check if output_name exists and is not empty (comes from saving_args)
            if "output_name" in args_data and args_data["output_name"]:
                args_data["log_prefix"] = args_data["output_name"] + "_"
            else:
                # Error only if output_name is expected but missing/empty
                over_errors.append("Log Prefix Mode is 'Output Name', but 'Output Name' in Saving Args is missing or empty.")
                over_pass = False # Mark overall validation as failed
        elif mode == "manual":
            # Check if log_prefix exists (set by logging UI) and is not empty
            if "log_prefix" not in args_data or not args_data.get("log_prefix", ""):
                over_errors.append("Log Prefix Mode is 'Manual', but the 'Manual Prefix' is missing or empty.")
                over_pass = False # Mark overall validation as failed
        elif mode == "disabled":
            if "log_prefix" in args_data:
                # Clean up if prefix exists but mode is disabled
                del args_data["log_prefix"]
        if "log_prefix_mode" in args_data: del args_data["log_prefix_mode"]

    if "run_name_mode" in args_data:
        mode = args_data["run_name_mode"]
        if mode == "output_name":
            if "output_name" in args_data and args_data["output_name"]:
                args_data["wandb_run_name"] = args_data["output_name"]
            else:
                # Error only if output_name is expected but missing/empty
                over_errors.append("Log Prefix Mode is 'Output Name', but 'Output Name' in Saving Args is missing or empty.")
                over_pass = False # Mark overall validation as failed
        elif mode == "manual":
            if "run_name" not in args_data or not args_data.get("run_name", ""):
                over_errors.append("Log Prefix Mode is 'Manual', but the 'Manual Run Name' is missing or empty.")
                over_pass = False # Mark overall validation as failed
                args_data["wandb_run_name"] = args_data["run_name"]
        elif mode == "default":
            if "run_name" in args_data:
                del args_data["run_name"]
        if "run_name_mode" in args_data: del args_data["run_name_mode"]

    # Get num_processes from accelerate settings for multi-GPU step calculations
    accelerate = args.get("accelerate", {})
    num_processes = accelerate.get("num_processes", 1) if accelerate.get("enabled", False) else 1

    tag_data = {}
    if not over_errors:
        validate_warmup_ratio(args_data, dataset_data, num_processes)
        validate_restarts(args_data, dataset_data, num_processes)
        tag_data = validate_save_tags(dataset_data)
        validate_existing_files(args_data)
        validate_optimizer(args_data)
    sdxl = validate_sdxl(args_data)
    return over_pass, sdxl, over_errors, args_data, dataset_data, tag_data


def validate_args(args: dict) -> tuple[bool, list[str], dict]:
    # sourcery skip: low-code-quality
    passed_validation = True
    errors = []
    output_args = {}

    for key, value in args.items():
        if (value is None or 
            (isinstance(value, str) and value.strip() == '')):
            passed_validation = False
            errors.append(f"No data filled in for {key}")
            continue
        if "fa" in value and value["fa"]:
            output_args["network_module"] = "networks.lora_fa"
            del value["fa"]
        for arg, val in value.items():
            if arg == "network_args":
                vals = []
                for k, v in val.items():
                    if k == "algo":
                        output_args["network_module"] = "lycoris.kohya"
                    elif k == "unit":
                        output_args["network_module"] = "networks.dylora"
                    if k in [
                        "down_lr_weight",
                        "up_lr_weight",
                        "block_dims",
                        "block_alphas",
                        "conv_block_dims",
                        "conv_block_alphas",
                    ]:
                        for i in range(len(v)):
                            v[i] = str(v[i])
                        vals.append(f"{k}={','.join(v)}")
                        continue
                    if k == "preset" and v == "":
                        continue
                    vals.append(f"{k}={v}")
                val = vals
            if arg == "optimizer_args":
                vals = []
                for k, v in val.items():
                    if isinstance(v, str) and v.strip().lower() in ["true", "false"]:
                        v = v.strip().capitalize()
                    vals.append(f"{k}={v}")
                val = vals
            if arg == "lr_scheduler_args":
                vals = [f"{k}={v}" for k, v in val.items()]
                val = vals
            if arg == "keep_tokens_separator" and len(val) < 1:
                passed_validation = False
                errors.append("Keep Tokens Separator is an empty string")
                continue
            if (val is None or 
                (isinstance(val, str) and val.strip() == '') or 
                (isinstance(val, bool) and val == False)):
                continue
            if isinstance(val, str):
                if val.strip().lower() == "true":
                    val = True
                elif val.strip().lower() == "false":
                    continue
            output_args[arg] = val
        if "fa" in value:
            del value["fa"]

    # Anima mode is detected by the presence of the 'qwen3' key (Anima-only arg).
    # In Anima mode, pretrained_model_name_or_path holds the DiT model path,
    # and 'qwen3' / 'vae' hold the text encoder and VAE paths respectively.
    is_anima = "qwen3" in output_args
    file_inputs = [
        {"name": "pretrained_model_name_or_path", "required": True},
        {"name": "qwen3", "required": is_anima},
        {"name": "vae", "required": is_anima},
        {"name": "sample_prompts", "required": False},
        {"name": "output_dir", "required": True},
        {"name": "logging_dir", "required": False},
        {"name": "t5_tokenizer_path", "required": False},
    ]

    for file in file_inputs:
        if file["required"] and file["name"] not in output_args:
            passed_validation = False
            errors.append(f"{file['name']} is not found")
            continue
        
        # Check if argument is present
        if file["name"] in output_args:
            path_obj = Path(output_args[file["name"]])
            
            # Special handling for creating directories
            if file["name"] in ["output_dir", "logging_dir"]:
                # If it doesn't exist, check if the parent/root is valid before creating
                if not path_obj.exists():
                    # Check if the parent path exists (or the path is relative and valid)
                    if not path_obj.parent.exists():
                        passed_validation = False
                        errors.append(f"Parent path for {file['name']} '{path_obj.parent}' does not exist")
                        continue
                    
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        passed_validation = False
                        errors.append(f"Could not create directory for {file['name']}: {e}")
                        continue

            # Standard validation for other files/paths (must already exist)
            elif not path_obj.exists():
                passed_validation = False
                errors.append(f"{file['name']} input '{output_args[file['name']]}' does not exist")
                continue
            
            output_args[file["name"]] = path_obj.as_posix()
    if "network_module" not in output_args:
        if "guidance_scale" in output_args:
            output_args["network_module"] = "networks.lora_flux"
        else:
            output_args["network_module"] = "networks.lora"
    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.is_file() else {}
    if "colab" in config_dict and config_dict["colab"]:
        output_args["console_log_simple"] = True
    return passed_validation, errors, output_args


def validate_dataset_args(args: dict) -> tuple[bool, list[str], dict]:
    passed_validation = True
    errors = []
    output_args = {"general": {}, "subsets": []}

    for key, value in args.items():
        if (value is None or 
                (isinstance(value, str) and value.strip() == '')):
            passed_validation = False
            errors.append(f"No Data filled in for {key}")
            continue
        if key == "subsets":
            continue
        for arg, val in value.items():
            if (val is None or 
                (isinstance(val, str) and val.strip() == '') or 
                (isinstance(val, bool) and value == False)):
                continue
            if arg == "max_token_length" and val == 75:
                continue
            output_args["general"][arg] = val

    for item in args["subsets"]:
        sub_res = validate_subset(item)
        if not sub_res[0]:
            passed_validation = False
            errors += sub_res[1]
            continue
        output_args["subsets"].append(sub_res[2])
    return passed_validation, errors, output_args


def validate_subset(args: dict) -> tuple[bool, list[str], dict]:
    passed_validation = True
    errors = []
    output_args = {
        key: value for key, value in args.items() if not 
        (value is None or 
         (isinstance(value, str) and value.strip() == '') or 
         (isinstance(value, bool) and value == False))
         }
    name = "subset"
    if "name" in output_args:
        name = output_args["name"]
        del output_args["name"]
    if "image_dir" not in output_args or not Path(output_args["image_dir"]).exists():
        passed_validation = False
        errors.append(f"Image directory path for '{name}' does not exist")
    else:
        output_args["image_dir"] = Path(output_args["image_dir"]).as_posix()
        
    if "target_image_dir" in output_args and Path(output_args["target_image_dir"]).exists():
        output_args["target_image_dir"] = Path(output_args["target_image_dir"]).as_posix()

    return passed_validation, errors, output_args


def validate_restarts(args: dict, dataset: dict, num_processes: int = 1) -> None:
    if "lr_scheduler_num_cycles" not in args:
        return
    if "lr_scheduler_type" not in args:
        return
    if "max_train_steps" in args:
        steps = args["max_train_steps"]
    else:
        steps = calculate_steps(
            dataset,
            args["max_train_epochs"],
            args.get("gradient_accumulation_steps", 1),
            num_processes,
        )
    steps = steps // args["lr_scheduler_num_cycles"]
    args["lr_scheduler_args"].append(f"first_cycle_max_steps={steps}")
    #del args["lr_scheduler_num_cycles"]


def validate_warmup_ratio(args: dict, dataset: dict, num_processes: int = 1) -> None:
    if "warmup_ratio" not in args:
        return
    if "max_train_steps" in args:
        steps = args["max_train_steps"]
    else:
        steps = calculate_steps(
            dataset,
            args["max_train_epochs"],
            args.get("gradient_accumulation_steps", 1),
            num_processes,
        )
    steps = round(steps * args["warmup_ratio"])
    if "lr_scheduler_type" in args:
        args["lr_scheduler_args"].append(f"warmup_steps={steps // args.get('lr_scheduler_num_cycles', 1)}")
    else:
        args["lr_warmup_steps"] = steps
    del args["warmup_ratio"]


def validate_existing_files(args: dict) -> None:
    file_name = Path(f"{args['output_dir']}/{args.get('output_name', 'last')}.safetensors")
    offset = 1
    while file_name.exists():
        file_name = Path(f"{args['output_dir']}/{args.get('output_name', 'last')}_{offset}.safetensors")
        offset += 1
    if offset > 1:
        print(f"Duplicate file found, changing file name to {file_name.stem}")
        args["output_name"] = file_name.stem


def validate_sdxl(args: dict) -> bool:
    if "sdxl" not in args:
        return False
    del args["sdxl"]
    return True


def validate_save_tags(dataset: dict) -> dict:
    tags = {}
    for subset in dataset["subsets"]:
        if 'is_val' in subset and subset['is_val']:
            continue
        subset_dir = Path(subset["image_dir"])
        if not subset_dir.is_dir():
            continue
        for file in subset_dir.iterdir():
            if not file.is_file():
                continue
            if file.suffix != subset["caption_extension"]:
                continue
            get_tags_from_file(subset_dir.joinpath(file.name), tags)
    return dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))


# Standard-optimizer names that never need custom resolution.
# Keeps the LoraEasyCustomOptimizer import (and its torchao chain) off
# the critical path for AdamW, SGD, Lion, etc.
_STANDARD_OPTIMIZERS: frozenset[str] = frozenset({
    "adamw", "adamw8bit",
    "lion", "lion8bit",
    "sgd", "sgdnesterov", "sgdnesterov8bit",
    "adafactor",
    "pagedadamw", "pagedadamw32bit", "pagedlion8bit",
})


def validate_optimizer(args: dict) -> None:
    opt_type_lower = args["optimizer_type"].lower()

    if opt_type_lower in _STANDARD_OPTIMIZERS or opt_type_lower.startswith("dadapt"):
        return

    from LoraEasyCustomOptimizer import OPTIMIZERS

    if opt_type_lower in OPTIMIZERS:
        args["optimizer_type"] = f"{OPTIMIZERS[opt_type_lower].__module__}.{OPTIMIZERS[opt_type_lower].__qualname__}"
    


def get_tags_from_file(file: str, tags: dict) -> None:
    with open(file, "r", encoding="utf-8") as f:
        temp = f.read().replace(", ", ",").split(",")
        for tag in temp:
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1


def calculate_steps(
    dataset_args: dict[str, dict | list[dict]],
    num_epochs: int,
    grad_acc_steps: int = 1,
    num_processes: int = 1,
) -> int:
    general_args: dict = dataset_args["general"]
    subsets: list = dataset_args["subsets"]
    supported_types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    resolution = (
        (general_args["resolution"], general_args["resolution"])
        if isinstance(general_args["resolution"], int)
        else general_args["resolution"]
    )
    if general_args.get("enable_bucket", False):
        bucketManager = BucketManager(
            general_args.get("bucket_no_upscale", False),
            resolution,
            general_args["min_bucket_reso"],
            general_args["max_bucket_reso"],
            general_args["bucket_reso_steps"],
            general_args.get("multires_training", False),
        )
        if not general_args.get("bucket_no_upscale", False):
            bucketManager.make_buckets()
    else:
        bucketManager = BucketManager(False, resolution, None, None, None, False)
        bucketManager.set_predefined_resos([resolution])
    for subset in subsets:
        if 'is_val' in subset and subset['is_val']:
            continue
        for image in Path(subset["image_dir"]).iterdir():
            if image.suffix not in supported_types:
                continue
            with Image.open(image) as img:
                bucket_reso, _, _ = bucketManager.select_bucket(img.width, img.height)
                for _ in range(subset["num_repeats"]):
                    bucketManager.add_image(bucket_reso, image)
    steps_before_acc = sum(
        math.ceil(len(bucket) / general_args["batch_size"]) for bucket in bucketManager.buckets
    )
    return math.ceil(steps_before_acc / grad_acc_steps / num_processes) * num_epochs
