#!/bin/bash
# Re-match torchaudio to the box's CUDA-matching PyTorch wheel.
#
# The Vast/RunPod base image pre-installs torchaudio from the cu130 index (or later), which won't
# load on a CUDA 12.x container (OSError: libcudart.so.13) -> ComfyUI dies on `import torchaudio`
# (it imports torchaudio for audio_vae). This pins torchaudio to torch's EXACT version + the box's
# CUDA, with --no-deps so the installed torch is never disturbed.
#
# Called by fetch-restart.sh (post-pull incremental update). NOTE: vastai_setup.sh keeps an INLINE
# copy of this same logic, because its torchaudio step runs BEFORE the repo is cloned (so this
# script doesn't exist on disk yet at that point). Keep the two copies in sync.
#
# Requires: a python with torch importable on PATH (activate the venv before calling this).

if command -v python &> /dev/null; then
    # chr(46)='.' avoids embedding a quote char in this command-substitution context.
    _cu="$(python -c "import torch; v=torch.version.cuda; print(f'cu{str().join(v.split(chr(46)))}')" 2>/dev/null || echo "")"
    if [ -n "$_cu" ]; then
        # Do NOT pin to torch's version: torchaudio LAGS torch (e.g. torch 2.12.0 but torchaudio
        # tops out at 2.11.0 on cu126), so torchaudio==<torch ver> 404s. Take the latest torchaudio
        # available on the matched CUDA index instead; --no-deps keeps the installed torch untouched
        # (without it, pip could try to DOWNGRADE torch to match torchaudio's pin).
        echo "🔊 Re-matching torchaudio to $_cu (latest available on that index) ..."
        pip install --force-reinstall --no-deps torchaudio --index-url "https://download.pytorch.org/whl/$_cu" \
            || echo "[match_torchaudio] torchaudio ($_cu) reinstall failed (non-fatal)"
    else
        echo "[match_torchaudio] could not detect torch CUDA — skipping (CPU build?)"
    fi
fi
