# Anzhc Nodes & Snippets Review

**Date:** 2026-07-07
**Status:** Draft (exploratory)
**Source repos:** 5 Anzhc repos

## Summary

Most are ComfyUI inference nodes, not training features. Two items worth noting:

---

## 1. LoRA Loader with Grouped Parameter Weighting (`anzhc-utility-nodes`)

`lora_loader_node.py` implements an advanced LoRA loader with:
- **Grouped parameter weighting** — targetting Anima LoRAs specifically
- **On-the-fly merging** (SVC method) — merge LoRAs into base model at load time
- Parameter groups for different LoRA block types

**Relevance:** The SVC merging method and grouped weighting approach could
inform our LoRA merge page, particularly for Anima model merges. Our current
merge weighted endpoint already supports block-weighted merging for SD models,
but Anima-specific merge patterns (grouped by parameter type) are not covered.

**Not implementing as-is** — this is a ComfyUI runtime node, not a batch
merge utility. But the grouping strategy could influence future merge UI
design for Anima models.

---

## 2. Autotagged Tile Conditioning (`anzhc-utility-nodes`)

A novel ComfyUI node that:
1. Patches a model for Tiled Diffusion
2. Tags each tile with WDv3
3. Injects tile-specific positive conditioning
4. Keeps negative conditioning universal

**Relevance:** Inference-only. Interesting concept but not training-related.

---

## 3. Qwen2D VAE (`anzhc-qwen2d-comfyui`)

A ComfyUI node for a lightweight Qwen2D VAE that replaces the full WAN/Qwen
VAE for static-image generation. Relevant to models that use Qwen-based VAEs
(Anima, Krea 2, Qwen Image).

**Inference-only.** No training implications.

---

## 4. Remaining repos — Not relevant to training

| Repo | What it does | Relevance |
|------|-------------|-----------|
| Euler A2 Sampler | Custom KSampler with time-travel-free inference | Inference only |
| SDXL-Flux2VAE | Patches SDXL VAE for Flux architecture | Inference only |
| Anima Mod Guidance | CLIP modulation guidance for Anima models | Inference only |
| YOLO Training | Ultralytics YOLO object detection training | Different domain entirely |

## Conclusion

None of these repos warrant their own plan doc. The LoRA loader's grouped
weighting approach is the only thing worth a note — and it's a one-liner
reference for when we extend the merge page for Anima models.
