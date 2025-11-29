# Attributions and Licenses

This project builds upon and is inspired by work from the open-source community. We are grateful to the following projects and contributors:

---

## Kohya SS GUI
**Repository:** [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
**Copyright:** Copyright (c) 2022 kohya-ss
**License:** Apache License 2.0
**Usage:** Reference implementation for WD14 tagger configuration, caption utilities, and training workflows

The following components were inspired by or reference the Kohya SS GUI implementation:
- WD14 auto-tagging parameter structure and validation
- Caption processing utilities and tag manipulation logic
- Training configuration patterns and TOML generation
- Dataset management approaches

Full license text: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Kohya Scripts (sd-scripts)
**Repository:** [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
**Copyright:** Copyright (c) 2022 kohya-ss
**License:** Apache License 2.0
**Usage:** Vendored in `trainer/derrian_backend/sd_scripts/` - Core training scripts for Stable Diffusion LoRA training

This project includes vendored copies of the Kohya training scripts for:
- SDXL, SD1.5, Flux, SD3.5, and Lumina LoRA training
- WD14 image tagging and captioning
- Network training utilities and optimizers

Full license text: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## LyCORIS
**Repository:** [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
**Copyright:** Copyright (c) 2023 KohakuBlueleaf
**License:** Apache License 2.0
**Usage:** Vendored in `trainer/derrian_backend/lycoris/` - Advanced LoRA architectures (LoCon, LoHa, LoKR, etc.)

Full license text: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Derrian's LoRA Easy Training Scripts
**Repository:** [derrian-distro/LoRA_Easy_Training_Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts)
**Copyright:** Copyright (c) 2023-2024 Derrian Distro
**License:** Apache License 2.0
**Usage:** Vendored in `trainer/derrian_backend/` - Enhanced Kohya scripts with improvements and fixes

This project vendors Derrian's distribution which includes:
- Stability improvements to Kohya scripts
- Additional model support and optimizations
- Bug fixes and compatibility updates

Full license text: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## WD14 Tagger Models
**Repository:** [SmilingWolf on Hugging Face](https://huggingface.co/SmilingWolf)
**Models Used:**
- `wd-vit-large-tagger-v3` (Recommended)
- `wd-vit-tagger-v3`
- `wd-v1-4-swinv2-tagger-v2`
- `wd-v1-4-convnext-tagger-v2`
- `wd-v1-4-vit-tagger-v2`

**License:** CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial)
**Usage:** Pre-trained models for automatic image tagging and captioning

---

## Additional Dependencies

### Backend
- **FastAPI** - MIT License
- **PyTorch** - BSD-style License
- **Transformers (Hugging Face)** - Apache License 2.0
- **Pydantic** - MIT License
- **ONNX Runtime** - MIT License

### Frontend
- **Next.js** - MIT License
- **React** - MIT License
- **Tailwind CSS** - MIT License
- **Radix UI** - MIT License
- **Lucide Icons** - ISC License

---

## Modifications

This project makes the following modifications to the referenced works:

1. **Web-based UI**: Replaced Gradio/Jupyter interfaces with a modern Next.js frontend
2. **Service Layer Architecture**: Refactored monolithic managers into modular services
3. **Real-time Monitoring**: Added WebSocket-based job tracking and progress monitoring
4. **API-first Design**: Created RESTful API with FastAPI for frontend-backend communication
5. **Enhanced Validation**: Implemented comprehensive Pydantic validation for all operations
6. **User Experience**: Redesigned workflows for better accessibility and usability

All modifications are documented in version control with clear commit messages indicating changes from upstream sources.

---

## License Compliance

This project complies with all upstream license requirements by:

- ✅ Retaining all original copyright notices
- ✅ Including copies of all required licenses
- ✅ Documenting modifications and attributions
- ✅ Preserving license headers in vendored code
- ✅ Providing clear attribution for all dependencies

---

## Contributing

When contributing to this project, please ensure:

1. New dependencies are compatible with Apache License 2.0
2. Proper attribution is added to this file
3. License headers are preserved in all vendored code
4. Modifications to upstream code are clearly documented

---

**Last Updated:** 2025-01-29
**Maintained By:** Ktiseos-Nyx-Trainer Project

If you believe any attribution is missing or incorrect, please open an issue or pull request.
