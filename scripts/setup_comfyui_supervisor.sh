#!/bin/bash
# One-shot script to add ComfyUI as its own supervisor program.
# Run on the VastAI instance:
#   bash /workspace/Ktiseos-Nyx-Trainer/scripts/setup_comfyui_supervisor.sh

set -e

mkdir -p /opt/supervisor-scripts
mkdir -p /workspace/Ktiseos-Nyx-Trainer/logs

cat > /opt/supervisor-scripts/comfyui.sh << 'EOF'
#!/bin/bash
source /venv/main/bin/activate 2>/dev/null || true

COMFYUI_PORT="${COMFYUI_PORT:-18188}"

if [ ! -d /workspace/Ktiseos-Nyx-Trainer/ComfyUI ]; then
    echo "[$(date)] ComfyUI not installed — exiting."
    exit 0
fi

cd /workspace/Ktiseos-Nyx-Trainer
echo "[$(date)] Starting ComfyUI on port $COMFYUI_PORT..."
exec python ComfyUI/main.py --port "$COMFYUI_PORT" --listen 0.0.0.0 --enable-cors-header
EOF

chmod +x /opt/supervisor-scripts/comfyui.sh
echo "✅ Created /opt/supervisor-scripts/comfyui.sh"

# Only append if the [program:comfyui] block isn't already there
if ! grep -q '\[program:comfyui\]' /etc/supervisor/conf.d/ktiseos-nyx.conf 2>/dev/null; then
    cat >> /etc/supervisor/conf.d/ktiseos-nyx.conf << 'EOF'

[program:comfyui]
command=/opt/supervisor-scripts/comfyui.sh
directory=/workspace/Ktiseos-Nyx-Trainer
autostart=true
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/Ktiseos-Nyx-Trainer/logs/comfyui.log
redirect_stderr=true
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
EOF
    echo "✅ Appended [program:comfyui] to supervisor conf"
else
    echo "ℹ️  [program:comfyui] already in conf, skipping"
fi

supervisorctl reread
supervisorctl update
supervisorctl start comfyui || true
echo ""
supervisorctl status
