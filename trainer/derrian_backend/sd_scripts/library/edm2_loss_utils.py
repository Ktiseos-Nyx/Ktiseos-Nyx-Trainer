import importlib
import ast
from library import edm2_loss, train_util
from library.utils import setup_logging
import torch
import math
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg', non-interactive backend
import matplotlib.pyplot as plt
plt.ioff() # Explicitly turn off interactive mode
import os
import numpy as np

setup_logging()
import logging

logger = logging.getLogger(__name__)

def prepare_edm2_loss_weighting(args, noise_scheduler, accelerator):
    if args.edm2_loss_weighting:
        handle_conflicting_configuration(args)
        values = args.edm2_loss_weighting_optimizer.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]
        opti_args = ast.literal_eval(args.edm2_loss_weighting_optimizer_args)
        opti_lr = float(args.edm2_loss_weighting_optimizer_lr) if args.edm2_loss_weighting_optimizer_lr else 2e-2

        edm2_model, edm2_optimizer = edm2_loss.create_weight_MLP(noise_scheduler,
                                                                    logvar_channels=int(args.edm2_loss_weighting_num_channels) if args.edm2_loss_weighting_num_channels else 128,
                                                                    optimizer=getattr(optimizer_module, case_sensitive_optimizer_type),
                                                                    lr=opti_lr,
                                                                    optimizer_args=opti_args,
                                                                    device=accelerator.device,
                                                                    dtype=torch.float32,
                                                                    use_importance_weights=args.edm2_loss_weighting_importance_weighting,
                                                                    importance_weights_max_weight=float(args.edm2_loss_weighting_importance_weighting_max) if args.edm2_loss_weighting_importance_weighting_max is not None else 10.0,
                                                                    importance_weights_min_snr_gamma=float(args.edm2_loss_weighting_importance_min_snr_gamma) if args.edm2_loss_weighting_importance_min_snr_gamma is not None else 1.0)
        if args.edm2_loss_weighting_initial_weights:
            edm2_model.load_weights(args.edm2_loss_weighting_initial_weights)

        if args.edm2_loss_weighting_lr_scheduler:
            def InverseSqrt(
                wrap_optimizer: torch.optim.Optimizer,
                warmup_steps: int = 0,
                constant_steps: int = 0,
                decay_scaling: float = 1.0,
            ):
                def lr_lambda(current_step: int):
                    if current_step <= warmup_steps:
                        return current_step / max(1, warmup_steps)
                    else:
                        return 1 / math.sqrt(max(current_step / max(constant_steps + warmup_steps, 1), 1)**decay_scaling)
                return torch.optim.lr_scheduler.LambdaLR(optimizer=wrap_optimizer, lr_lambda=lr_lambda)

            edm2_lr_scheduler = InverseSqrt(
                edm2_optimizer,
                warmup_steps=args.max_train_steps * float(args.edm2_loss_weighting_lr_scheduler_warmup_percent) if args.edm2_loss_weighting_lr_scheduler_warmup_percent is not None else 0.05,
                constant_steps=args.max_train_steps * float(args.edm2_loss_weighting_lr_scheduler_constant_percent) if args.edm2_loss_weighting_lr_scheduler_constant_percent is not None else 0.15,
                decay_scaling=float(args.edm2_loss_weighting_lr_scheduler_decay_scaling) if args.edm2_loss_weighting_lr_scheduler_decay_scaling is not None else 1.0,
            )
        else:
            edm2_lr_scheduler = train_util.get_dummy_scheduler(edm2_optimizer)

        # Handle DeepSpeed differently: don't use accelerator.prepare() at all
        # When DeepSpeed is active, prepare() converts models to bf16 and wraps them
        # in ways that break gradient flow for separate models like EDM2
        is_deepspeed = getattr(args, 'deepspeed', False)
        if is_deepspeed:
            logger.info("DeepSpeed detected: keeping EDM2 as standalone PyTorch model (float32)")
            # Just move to device, keep as float32 for stable gradient computation
            edm2_model = edm2_model.to(accelerator.device)
            edm2_model.train()
            # Keep optimizer and scheduler as regular PyTorch objects
        else:
            edm2_lr_scheduler = accelerator.prepare(edm2_lr_scheduler)
            edm2_model, edm2_optimizer = accelerator.prepare(edm2_model, edm2_optimizer)
    else:
        edm2_optimizer = None
        edm2_lr_scheduler = None
        edm2_model = None

    return edm2_model, edm2_optimizer, edm2_lr_scheduler

def handle_conflicting_configuration(args):
    if args.edm2_loss_weighting and args.edm2_loss_weighting_importance_weighting and not args.edm2_loss_weighting_importance_weighting_safety_override:
        if args.debiased_estimation_loss:
            args.debiased_estimation_loss = False
            logger.warning("Debiased estimation loss AND EDM2 loss weighting with importance weighting are enabled. " \
            "It is not advised to use both, as there is a possiblity of loss curving to 0 as SNR approaches 0, " \
            "as such, Debiased estimation loss has been DISABLED. " \
            "You may override this behavior by setting edm2_loss_weighting_importance_weighting_safety_override=True.")

        if args.min_snr_gamma:
            logger.warning("Min snr gamma AND EDM2 loss weighting with importance weighting are enabled. " \
            "It is not advised to use both, as there is a possiblity of loss curving to 0 as SNR approaches 0, " \
            "as such, min snr gamma has been DISABLED. " \
            "You may override this behavior by setting edm2_loss_weighting_importance_weighting_safety_override=True.")
            args.min_snr_gamma = None

def plot_edm2_loss_weighting_check(args, global_step):
    return args.edm2_loss_weighting and args.edm2_loss_weighting_generate_graph and (global_step % (int(args.edm2_loss_weighting_generate_graph_every_x_steps) if args.edm2_loss_weighting_generate_graph_every_x_steps else 20) == 0 or global_step >= args.max_train_steps)

def plot_edm2_loss_weighting(args, step: int, model, num_timesteps: int = 1000, device="cpu"):
    """
    Plot the edm2 loss weighting across timesteps using the learned parameters.

    :param model: The edm2 model instance (after training). Can be DDP-wrapped.
    :param num_timesteps: Total number of timesteps to plot.
    :param device: Device to run computations on.
    """
    # Unwrap DDP model if necessary
    unwrapped_model = model.module if hasattr(model, 'module') else model

    with torch.inference_mode():
        unwrapped_model.train(False)
        timesteps = torch.arange(0, 1000, device=device, dtype=torch.long)
        learnedweights = unwrapped_model._forward(timesteps).cpu().numpy()
        lambdas = unwrapped_model.lambda_weights.cpu().numpy()
        learnedweights = lambdas/np.exp(learnedweights)
        unwrapped_model.train(True)

        # Plot the dynamic loss weights over time
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps.cpu().numpy(), learnedweights,
                label=f'Dynamic Loss Weight\nStep: {step}')
        plt.xlabel('Timesteps')
        plt.ylabel('Weight')
        plt.title('Dynamic Loss Weighting vs Timesteps')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        if args.edm2_loss_weighting_generate_graph_y_limit is not None:
            plt.ylim(top=int(args.edm2_loss_weighting_generate_graph_y_limit))
        plt.xlim(left=0, right=num_timesteps)
        plt.xticks(np.arange(0, num_timesteps+1, 100)) 
        # plt.show()
        
        try:
            os.makedirs(args.edm2_loss_weighting_generate_graph_output_dir, exist_ok=True)
            output_dir = os.path.join(args.edm2_loss_weighting_generate_graph_output_dir, args.output_name)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"weighting_step_{str(step).zfill(7)}.png"))
        except Exception as e:
            logger.warning(f"Failed to save weighting graph image. Due to: {e}")

        plt.close()