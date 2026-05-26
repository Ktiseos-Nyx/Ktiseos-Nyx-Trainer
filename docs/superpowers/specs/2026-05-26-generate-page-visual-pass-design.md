# Generate Page Visual Pass — Design

**Date:** 2026-05-26
**Status:** Approved (pending spec review)
**Scope:** `frontend/components/comfy/GenerateUI.tsx` (the connected-state generate UI). The pre-connect cards and page header in `frontend/app/comfyui/page.tsx` already carry the app's visual treatment (`BorderGlow`, `NoiseTexture`) and are out of scope except where noted.

## Goal / North Star

**"Welcoming, not corporate."** The connected generate UI is currently functional but flat. Add visual warmth using components we *already* have — no new design system. The guiding rule:

- **Glow concentrates on the emotional beats of generating:** start (Generate), during (progress), result (gallery).
- **Contrast (not glow) defines the workspace:** the dense control sections get quiet structural definition so they read as crafted panels, not a flat wall — while staying calm and noise-free.

## Components: use INSTALLED shadcn/registry components, not hand-rolled

Audited 2026-05-26. `components.json` has ~25 registries wired up (@magicui, @aceternity, @react-bits, …) and `components/ui/` already contains properly-engineered effect components. Per the project rule "always use shadcn where possible," this pass uses the **installed** components and **retires the bare-JSX duplicates** in `components/effects/` + `BorderGlow.jsx`.

| Need | USE (installed, `components/ui/`) | Retire (hand-built dupe) |
|---|---|---|
| Glowing CTA | `shiny-button.tsx` (`ShinyButton`) — `cva`+`Slot`+`forwardRef`, real `<button>`, focus ring, `disabled`, configurable gradient. Verified accessible. | `effects/gradient-border-button` (raw `<button>`) |
| Animated glow/shimmer border | `shine-border.tsx` (`ShineBorder`) — overlay; `shine` keyframe confirmed in `globals.css:106` | `effects/shimmer-border`, `BorderGlow.jsx` |
| Hover glow on cards | theme-aware Tailwind ring/glow (`hover:`+`focus-visible:`). NOT `hover-border-gradient` — on inspection it's a pill-shaped, hardcoded-black, blue-highlight component unsuited to square image thumbnails. | `effects/gradient-border-card` |
| Inset surface / texture | shadcn `Card` + theme tokens, or `texture-card.tsx` | — |
| Noise overlay | `noise-texture.tsx` — already mounted on the page; no change | — |

Other installed options available if preferred: `rainbow-button`, `backlight`, `spotlightcard`, `background-gradient`, `border-glide`.

**No new installs required** — everything needed is already in `components/ui/`. The hand-built `effects/` components and `BorderGlow.jsx` are not used by this pass; broader removal of them across the app is banked as a separate cleanup.

## Section-by-Section Treatment

### Focal points — real glow

| Spot | Location | Treatment |
|---|---|---|
| **Generate button** (hero CTA) | `GenerateUI.tsx:888` | Use the installed **`ShinyButton`** (`components/ui/shiny-button`) for the idle Generate CTA — it's a real accessible `<button>` (focus ring, `disabled`), so we keep the `ClickSpark` wrapper and `disabled`/`queued` states. Tune its gradient to the app palette (purple/pink). The **Stop** (generating) button stays a plain destructive `Button`. |
| **Generation-active state** | `GenerationProgress` (`:290`) | Wrap the active progress zone in **`ShineBorder`** (`components/ui/shine-border`, purple, slow) so "working" visibly breathes. Keep the existing `Progress` bar inside. Only renders while `isGenerating`. |
| **Gallery empty state** | `ImageGallery` (`:241`) | Replace the dashed-box placeholder with a soft **`ShineBorder`** (or `backlight`) frame + friendlier copy, so the empty canvas reads as an invitation rather than a placeholder. |
| **Gallery thumbnails (hover)** | `ImageGallery` (`:256`) | Give the thumbnail `Button`s a theme-aware **Tailwind** ring/glow on `hover:` AND `focus-visible:` (keyboard parity) so browsing feels tactile. NOT `hover-border-gradient` (unsuitable — pill/black/blue, not theme-aware). Preserve the keyboard focus/activation already in place. |

### Calm but contrastive — no glow, structural contrast only

| Spot | Treatment |
|---|---|
| **Control sections** (Prompts, Model, Size, Sampler, Post-processing, LoRA stack) | Give each group a subtle **inset surface**: faint background (theme token, e.g. `bg-muted/40`), hairline border (`border-border/40`), rounded corners, slightly more padding. Replaces the flat `<Separator>`-only division. `SectionLabel` becomes each panel's header. Contrast/grouping via elevation, never glow. |
| **Architecture switcher** (`ArchSwitcher`, top) | Keep the custom pill control; add only a gentle accent to the *active* mode. No full border. |

### Deliberately unchanged

- Prompt textareas, sampler sliders/selects, size/seed/batch inputs, model pickers, toggles — the cockpit controls stay quiet (only gain the inset-surface grouping above).
- Page header and pre-connect cards in `page.tsx`.
- The intentional meteor background (do not touch).

## Constraints

- **shadcn rule:** no raw HTML form elements; keep using `@/components/ui/*`. Glow/effect wrappers wrap shadcn components, not replace them.
- **Accessibility:** preserve the keyboard-accessible gallery `Button`s (no regression to clickable `<div>`s); effect wrappers must not strip focusability.
- **Theming:** prefer theme tokens (`bg-card`, `bg-muted`, `border-border`) over hardcoded colors so light/dark both work. Effect components that take explicit color props should use the app's existing purple/pink/cyan palette already seen on the page.
- **Functionality:** purely additive styling — no change to generation logic, state, or the submit flow.
- **Cross-platform:** no platform-specific assumptions (pure CSS/Tailwind + existing React components).

## Verification (manual, in dev server)

- Golden path: connect to ComfyUI → fill a model → Generate → watch progress → see gallery populate. Confirm glow appears at start/during/result and controls remain readable.
- Light **and** dark mode both look intentional (inset surfaces + glow).
- Keyboard: Tab to a gallery thumbnail, Enter opens the lightbox (no regression).
- Dense panel still parses easily — inset surfaces aid grouping, don't add noise.

## Out of Scope (banked separately)

- `train_llm_adapter` wiring, preset audit/rename, LyCORIS re-sync — tracked elsewhere, not part of this visual pass.
