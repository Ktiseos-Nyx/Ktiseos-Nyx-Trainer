# VastAI Supervisor Workaround (Archived)

## The Problem
VastAI provisioning wouldn't download the repo and run the install **before** opening the container. This caused startup failures.

## The Janky Solution
Used Supervisor to handle startup process as a workaround. This got it **PARTIALLY working** but was hacky.

## Files
- `ktiseos-nyx.conf` - Supervisor config for VastAI
- `ktiseos-nyx.sh` - Startup script called by Supervisor
- `SUPERVISOR_SETUP.sh` - Setup script for Supervisor integration
- `SUPERVISOR_SETUP_COMMANDS.txt` - Qwen's suggestions for Supervisor setup

## Status
**Archived** - Not currently used. Kept for reference in case VastAI provisioning acts up again and we need to reference this approach.

## Notes
- These folders DO exist on the GPU (`/opt/supervisor-scripts/`, `/workspace/`)
- The workaround was janky but functional
- Current deployment uses `vastai_setup.sh` instead
