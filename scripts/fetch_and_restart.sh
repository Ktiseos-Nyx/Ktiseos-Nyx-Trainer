#!/bin/bash
# Pull latest dev changes and restart all services.
# Run on the VastAI instance:
#   bash /workspace/Ecosystem_WebUI/scripts/fetch_and_restart.sh

set -e

cd /workspace/Ecosystem_WebUI

echo "🔄 Pulling latest dev changes..."
git pull origin dev

echo ""
echo "🎨 Rebuilding frontend..."
python install_frontend.py --force

echo ""
echo "🔁 Restarting services..."
supervisorctl restart ecosystem
supervisorctl restart comfyui

echo ""
echo "⏳ Waiting for services to come up..."
sleep 10

echo ""
supervisorctl status
echo ""
echo "✅ Done!"
