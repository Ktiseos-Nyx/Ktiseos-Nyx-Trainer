# Security Policy

This document outlines our security practices and procedures for the Ktiseos-Nyx-Trainer project.

## Security Scope

### Critical Security Concerns
- **Code execution vulnerabilities**: Arbitrary command execution prevention in API endpoints
- **Malicious model downloads**: File validation and verification for HuggingFace/Civitai downloads
- **Data exfiltration**: Protection of training datasets and API credentials
- **Dependency integrity**: Package supply chain security (npm + pip)
- **File system access**: Proper permission boundaries and path validation
- **API security**: Input validation on all FastAPI endpoints
- **WebSocket security**: Protection of real-time training log streams

### Out of Scope
- Theoretical attacks requiring physical machine access
- Vulnerabilities requiring manual code modification to exploit
- Issues affecting users who intentionally bypass safety mechanisms
- Local denial of service attacks against development server instances

## Reporting Security Vulnerabilities

### Responsible Disclosure Process
1. **Private Reporting**: Use [GitHub Security Advisories](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/security/advisories/new) for confidential reporting
2. **Report Details**: Include vulnerability description, reproduction steps, and potential impact assessment
3. **Response Timeline**: Initial response within 48 hours, resolution target within 7 days
4. **Public Disclosure**: Coordinated disclosure after fix implementation and testing

### Submission Guidelines
- Provide clear reproduction steps and environment details
- Include potential impact assessment and exploitation scenarios
- Submit uncertain findings - we prefer false positives over missed vulnerabilities

## Security Implementation

### Code Security Practices
- **Input Validation**: Pydantic models with strict typing on all API endpoints; Zod schemas for frontend validation
- **Path Traversal Prevention**: Sanitization of file paths in file browser and dataset upload endpoints
- **Safe Defaults**: No automatic script execution, minimal privilege requirements
- **Dependency Management**: Version pinning via `package-lock.json` (npm) and `requirements.txt` (pip), with Dependabot monitoring
- **Error Handling**: Information disclosure prevention in error messages

### Download Security Controls
- **URL Validation**: Verification of download sources (HuggingFace Hub API, Civitai API)
- **File Integrity**: Extension and type verification for downloaded models and VAEs
- **Size Limiting**: Prevention of excessive resource consumption
- **External Scanning**: Integration recommendations for antivirus validation

### Credential Management
- **No Logging**: API keys and tokens (HuggingFace, Civitai, W&B) excluded from log files and console output
- **Client-Side Storage**: Tokens stored in browser localStorage, never persisted server-side
- **Minimal Permissions**: Least-privilege principle for API access scopes

## System Architecture Security

### Web Application Stack
- **Frontend**: Next.js 15 (React 19) on port 3000 — serves the web UI
- **Backend API**: FastAPI (Python) on port 8000 — handles all business logic
- **WebSockets**: Real-time log streaming for training, tagging, and captioning jobs
- **VastAI Deployment**: Ports mapped to 13000 (frontend) and 18000 (API) behind VastAI proxy

### External Dependencies
The system downloads and executes content from external sources:
- **HuggingFace Models**: Community-contributed model files via `huggingface_hub` API
- **Civitai Models**: Community model repository via Civitai REST API
- **PyPI Packages**: Python package dependencies via pip
- **npm Packages**: Frontend dependencies via npm registry

### Code Execution Context
- **Training Scripts**: GPU-intensive Python script execution via vendored Kohya SS backend
- **Tagging/Captioning**: ONNX model inference for WD14 tagging and BLIP/GIT captioning
- **System Commands**: File extraction, model downloads, and training subprocess management

### VastAI Deployment Considerations
- **Shared GPU Hosts**: Training runs on shared infrastructure; isolate sensitive data
- **Exposed Ports**: Frontend and API ports are publicly accessible via VastAI proxy URLs
- **Ephemeral Instances**: Training outputs should be downloaded before instance termination
- **Auto-Provisioning**: `vastai_setup.sh` installs dependencies and starts services automatically

## User Security Responsibilities

### Environment Security
- Deploy on development/training systems, not production infrastructure
- Implement appropriate network isolation and firewall rules
- Maintain current system patches and Python versions
- Verify download sources and model authenticity
- Use multi-factor authentication for external service accounts

### Best Practices

**Environment Setup:**
```bash
# Use virtual environments for the Python backend
python -m venv lora_training
source lora_training/bin/activate

# Verify user permissions
whoami  # Should not return privileged user

# Maintain system updates
sudo apt update && sudo apt upgrade  # Linux
```

**Credential Management:**
```bash
# Environment variable storage
export HF_TOKEN="your_token_here"
export CIVITAI_TOKEN="your_token_here"

# Local environment files (excluded from version control)
echo "HF_TOKEN=your_token_here" >> .env
```

**Network Security:**
- Use HTTPS for all external communications
- Consider VPN usage on untrusted networks
- On VastAI, be aware that proxy URLs are publicly accessible — do not leave instances running unattended with sensitive data
- For local development, the API and frontend are bound to localhost by default

**File System Management:**
- Use dedicated directories for training activities
- Implement regular backup procedures for valuable training outputs
- Monitor and clean temporary file accumulation
- Download completed LoRA outputs from VastAI instances before terminating them

## Supported Versions

Security updates are provided for:
- **Current Release**: Full security support and immediate updates
- **Previous Release**: Critical security fixes for 90 days
- **Legacy Versions**: No security support - upgrade recommended

## Threat Model

### Protected Against
- Malicious model files exploiting training processes
- Network-based attacks during file downloads
- Local privilege escalation through file operations
- Information disclosure through application logs
- Path traversal via file browser and upload endpoints

### Not Protected Against
- Advanced persistent threats with significant resources
- Physical access attacks on local systems
- Social engineering targeting user credentials
- Theoretical future cryptographic vulnerabilities
- Malicious actors with direct access to VastAI instance shell

## Security Roadmap

### Current Development
- Enhanced input sanitization across all API endpoints
- Pydantic validation with strict Literal types on all route models
- Automated dependency vulnerability scanning via Dependabot

### Future Considerations
- Cryptographic verification for downloaded model files
- Process sandboxing for training execution
- Authentication layer for multi-user deployments
- Rate limiting on API endpoints

## Contact Information

- **Security Issues**: [GitHub Security Advisories](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/security/advisories/new) (preferred)
- **General Questions**: [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- **Community Support**: Discord server (see README for current invite)

---

**Security Philosophy**: We aim to balance practical security with usability for machine learning workflows. Perfect security is unattainable, but we strive for robust protection against realistic threat scenarios.

**Last Updated**: March 2026
