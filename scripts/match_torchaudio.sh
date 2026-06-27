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
    # chr(46)='.' / chr(43)='+' avoid embedding quote chars in this command-substitution context.
    _cu="$(python -c "import torch; v=torch.version.cuda; print(f'cu{str().join(v.split(chr(46)))}')" 2>/dev/null || echo "")"
    _tv="$(python -c "import torch; print(torch.__version__.split(chr(43))[0])" 2>/dev/null || echo "")"
    if [ -n "$_cu" ] && [ -n "$_tv" ]; then
        echo "🔊 Matching torchaudio==$_tv to $_cu ..."
        pip install --force-reinstall --no-deps "torchaudio==$_tv" --index-url "https://download.pytorch.org/whl/$_cu" \
            || echo "[match_torchaudio] torchaudio==$_tv ($_cu) reinstall failed (non-fatal)"
    else
        echo "[match_torchaudio] could not detect torch CUDA/version — skipping (CPU build?)"
    fi
fi
