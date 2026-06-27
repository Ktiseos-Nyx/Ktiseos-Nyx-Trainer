#!/bin/bash
# Re-install torchaudio as a CUDA-matched wheel so ComfyUI can import it.
#
# The Vast/RunPod base image pre-installs torchaudio built for cu130 (or later), which won't load on
# our CUDA 12.x boxes (OSError: libcudart.so.13) -> ComfyUI dies on `import torchaudio` (audio_vae).
#
# We standardize on CUDA 12.6, so point pip straight at the cu126 wheel index and let it pick the
# matching torchaudio (currently 2.11.0+cu126). A cu126 wheel needs libcudart.so.12, which every
# CUDA 12.x box has, so this is fine across 12.1-12.8.
#
# --no-deps is REQUIRED: torchaudio's wheel declares a hard `torch==` pin (e.g. 2.11.0 -> torch 2.11)
# that conflicts with the installed torch 2.12; without --no-deps pip would try to DOWNGRADE torch.
# (This is also why torchaudio can't just be a line in requirements.txt -- --no-deps is a global pip
# flag, not something you can scope to a single requirement. Hence the separate command.)
#
# Called by fetch-restart.sh. vastai_setup.sh keeps an inline copy (its step runs pre-clone, before
# this file exists on disk) -- keep the two in sync. If the CUDA standard ever changes, update the URL.

pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    || echo "[match_torchaudio] cu126 torchaudio reinstall failed (non-fatal)"
