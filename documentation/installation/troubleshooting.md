# Troubleshooting

Troubleshooting Issues for Installation and Runtime.

---

## Troubleshooting

### Python not found

```text
'python' is not recognized as an internal or external command
```

- Windows: reinstall Python with "Add to PATH" checked, or add manually via System Properties → Environment Variables
- Linux: `sudo apt install python3.10 python3.10-venv python3-pip`
- Verify: `python --version` or `python3 --version`

### CUDA not available

```python
torch.cuda.is_available()  # returns False
```

1. Verify driver: `nvidia-smi` — if this fails, drivers aren't installed
2. Check CUDA version in `nvidia-smi` output — must be 12.1+
3. Verify PyTorch CUDA build: `python -c "import torch; print(torch.version.cuda)"`
4. If PyTorch reports CPU-only, reinstall with the correct CUDA index:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### npm install fails

```text
npm ERR! ERESOLVE unable to resolve dependency tree
```

This error should not occur on Next.js 16 / React 19. If it does, clear the npm cache and retry:
```bash
npm cache clean --force
npm install
```

### Permission denied (Windows)

- Do not install in `C:\`, `Program Files`, or any system-owned directory
- Do not install in OneDrive, Dropbox, or Google Drive synced folders
- Install in a path you own: `C:\Users\YourName\Projects\`

### Permission denied (Linux)

```bash
chmod +x install.sh start_services_local.sh restart.sh
```

### Port already in use

```bash
# Linux/macOS — find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Windows — find PID, then kill it
netstat -ano | findstr :3000
taskkill /PID <pid> /F
```

Or start on different ports:
```bash
start_services_local.bat --port 4000 --backend-port 9000
```

### Training fails immediately

1. Check GPU availability: `nvidia-smi`
2. Check VRAM: `nvidia-smi --query-gpu=memory.free,memory.total --format=csv`
3. Check logs in `logs/app_YYYYMMDD.log`
4. Run the diagnostic: `diagnose.bat` / `./diagnose.sh`
5. Open a [GitHub issue](https://github.com/UselessToys/Ecosystem_WebUI/issues) with the `diagnostics_*.txt` output

### VastAI: services not starting after reboot

Supervisor manages the services. Check status:
```bash
supervisorctl status
bash fetch-restart.sh
./fetch-restart.sh 
```



---

*For questions not covered here, open a [GitHub issue](https://github.com/UselessToys/Ecosystem_WebUI/issues).*
