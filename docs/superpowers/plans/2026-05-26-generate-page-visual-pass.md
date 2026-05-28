# Generate Page Visual Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ComfyUI generate page feel "welcoming, not corporate" — glow at the emotional beats (Generate, generating, gallery) via installed shadcn/registry components, and theme-aware structural contrast for the dense control sections.

**Architecture:** Pure additive styling on `frontend/components/comfy/GenerateUI.tsx`. Use the already-installed `ShinyButton` and `ShineBorder` from `components/ui/`; do NOT add new dependencies or use the hand-built `components/effects/*`. No generation/logic changes.

**Tech Stack:** Next.js 16, React 19, Tailwind v4, shadcn/ui, `motion` (motion/react). Effect components: `@/components/ui/shiny-button` (magicui), `@/components/ui/shine-border` (magicui).

---

## Testing approach (read first)

This is a **visual pass**. The frontend has **no automated UI test runner** (the `tests/` tree is Python), so do not invent one — that would violate YAGNI. Each task is verified by:

1. **Typecheck/lint:** `cd frontend && npm run lint` — expect no NEW errors (pre-existing errors in unrelated files are acceptable; do not fix them here).
2. **Manual visual check:** run `cd frontend && npm run dev`, open `http://localhost:3000/comfyui`. **GenerateUI only mounts after ComfyUI has connected at least once** (`hasEverConnected`), so ComfyUI must be running on its default port for the connected UI to render. Confirm the change in **both light and dark mode** (theme toggle in the navbar).
3. **Keyboard check** (where noted): Tab to the element, confirm focus/activation behaves.

If ComfyUI cannot be run during implementation, say so explicitly in the task notes rather than claiming visual success.

## File Structure

- **Modify only:** `frontend/components/comfy/GenerateUI.tsx` — the connected-state generate UI. All five tasks touch this one file.
- **Reuse (no edits):** `frontend/components/ui/shiny-button.tsx`, `frontend/components/ui/shine-border.tsx` (its `shine` keyframe is confirmed present in `app/globals.css:106`).
- **Do NOT touch:** `frontend/components/effects/*`, `frontend/components/BorderGlow.jsx`. They are the legacy hand-built duplicates being retired.

---

### Task 1: Generate CTA → ShinyButton

The idle Generate button is the hero action; make it glow while keeping shadcn `<Button>`-grade accessibility. The **Stop** button stays a plain destructive `Button`.

> **Integration note:** `ShinyButton`'s `className` prop is applied to its inner background layer, NOT the `<button>` element. So use `style={{ width: "100%" }}` for full width (it spreads to the button via `...props`). Its default inner bg (`bg-neutral-100 dark:bg-neutral-900`) is already theme-aware — leave it.

**Files:**
- Modify: `frontend/components/comfy/GenerateUI.tsx` (imports near line 23; Generate button near lines 883–890)

- [ ] **Step 1: Add the import**

After the existing `import { Button } from '@/components/ui/button';` (line 23), add:

```tsx
import { ShinyButton } from '@/components/ui/shiny-button';
```

- [ ] **Step 2: Replace the idle Generate button**

Find this block (the `else` branch of the `isGenerating` ternary, ~lines 883–890):

```tsx
              ) : (
                <Button className="w-full gap-2" onClick={handleGenerate} disabled={!canGenerate || !isConnected}>
                  Generate
                  {queueRemaining > 0 && (
                    <span className="ml-1 text-xs opacity-70">({queueRemaining} queued)</span>
                  )}
                </Button>
              )}
```

Replace with:

```tsx
              ) : (
                <ShinyButton
                  onClick={handleGenerate}
                  disabled={!canGenerate || !isConnected}
                  gradientFrom="#a855f7"
                  gradientTo="#ec4899"
                  style={{ width: "100%" }}
                >
                  Generate
                  {queueRemaining > 0 && (
                    <span className="ml-1 text-xs opacity-70">({queueRemaining} queued)</span>
                  )}
                </ShinyButton>
              )}
```

Leave the `<Button variant="destructive" ...>` Stop branch unchanged.

- [ ] **Step 3: Typecheck/lint**

Run: `cd frontend && npm run lint`
Expected: no new errors referencing `GenerateUI.tsx`.

- [ ] **Step 4: Visual + keyboard check**

With the dev server up and ComfyUI connected: the idle Generate button shows a purple→pink shine that tracks the pointer; it's still full-width; `disabled` (no model / disconnected) greys it out; Tab reaches it and Enter triggers generation. Confirm in light and dark.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/comfy/GenerateUI.tsx
git commit -m "feat(comfy): glowing ShinyButton for the Generate CTA"
```

---

### Task 2: Generation-active state → ShineBorder

Make "working" feel alive: an animated shine border around the progress strip while generating.

> **Note:** `ShineBorder` is an absolutely-positioned overlay (`absolute inset-0`). Its parent must be `relative overflow-hidden`.

**Files:**
- Modify: `frontend/components/comfy/GenerateUI.tsx` (imports; `GenerationProgress`, ~lines 302–313)

- [ ] **Step 1: Add the import**

After the `import { ShinyButton } ...` line from Task 1, add:

```tsx
import { ShineBorder } from '@/components/ui/shine-border';
```

- [ ] **Step 2: Wrap the progress zone**

Find (in `GenerationProgress`, ~lines 302–312):

```tsx
    <div className="flex shrink-0 flex-col gap-1.5 border-t border-border/40 px-3 py-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" />
          {currentNode ? `Node: ${currentNode}` : 'Queued…'}
        </span>
        {progress && <span>{pct}%</span>}
      </div>
      <Progress value={pct} className="h-1 bg-muted" />
    </div>
```

Replace with:

```tsx
    <div className="relative flex shrink-0 flex-col gap-1.5 overflow-hidden border-t border-border/40 px-3 py-2">
      <ShineBorder shineColor={["#a855f7", "#ec4899", "#38bdf8"]} duration={8} borderWidth={1} />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" />
          {currentNode ? `Node: ${currentNode}` : 'Queued…'}
        </span>
        {progress && <span>{pct}%</span>}
      </div>
      <Progress value={pct} className="h-1 bg-muted" />
    </div>
```

- [ ] **Step 3: Typecheck/lint**

Run: `cd frontend && npm run lint`
Expected: no new errors.

- [ ] **Step 4: Visual check**

Start a generation; the progress strip at the bottom of the gallery shows a slow purple/pink/cyan shine tracing its border, and disappears when generation ends (the component already returns `null` when `!isGenerating`). Confirm in light and dark.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/comfy/GenerateUI.tsx
git commit -m "feat(comfy): ShineBorder on the active generation progress strip"
```

---

### Task 3: Gallery empty state → ShineBorder frame + friendlier copy

Turn the placeholder into an invitation.

**Files:**
- Modify: `frontend/components/comfy/GenerateUI.tsx` (`ImageGallery` empty branch, ~lines 241–250)

- [ ] **Step 1: Replace the empty-state markup**

Find:

```tsx
  if (images.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground/50 p-8 text-center">
        <div className="h-16 w-16 rounded-2xl border-2 border-dashed border-muted-foreground/20 flex items-center justify-center">
          <span className="text-2xl">🎨</span>
        </div>
        <p className="text-sm">Generated images will appear here</p>
      </div>
    );
  }
```

Replace with:

```tsx
  if (images.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center">
        <div className="relative overflow-hidden rounded-2xl bg-card/40 px-8 py-10">
          <ShineBorder shineColor={["#a855f7", "#ec4899", "#38bdf8"]} duration={10} borderWidth={1} />
          <div className="relative flex flex-col items-center gap-3">
            <span className="text-3xl">🎨</span>
            <p className="text-sm font-medium text-muted-foreground">Your canvas is ready</p>
            <p className="text-xs text-muted-foreground/60">Generated images will appear here</p>
          </div>
        </div>
      </div>
    );
  }
```

(Uses the `ShineBorder` import added in Task 2.)

- [ ] **Step 2: Typecheck/lint**

Run: `cd frontend && npm run lint`
Expected: no new errors.

- [ ] **Step 3: Visual check**

With ComfyUI connected and no images yet, the gallery shows a softly shimmering rounded card with the new copy. Confirm in light and dark (the `bg-card/40` should read on both).

- [ ] **Step 4: Commit**

```bash
git add frontend/components/comfy/GenerateUI.tsx
git commit -m "feat(comfy): welcoming ShineBorder gallery empty state"
```

---

### Task 4: Thumbnail hover + focus glow (theme-aware Tailwind)

The spec originally named `hover-border-gradient`, but on inspection it's a pill-shaped, hardcoded-black, blue-highlight component unsuited to square image thumbnails. Use a theme-aware Tailwind glow instead, applied on **both** hover and `focus-visible` so keyboard users get the same highlight.

**Files:**
- Modify: `frontend/components/comfy/GenerateUI.tsx` (thumbnail `Button`, ~lines 256–261)

- [ ] **Step 1: Update the thumbnail className**

Find:

```tsx
          <Button
            key={`${img.filename}-${i}`}
            variant="ghost"
            onClick={() => setSelected(img)}
            className="aspect-square h-auto cursor-pointer overflow-hidden rounded-xl border border-border/50 p-0 transition-all hover:border-primary/50 hover:shadow-md"
          >
```

Replace the `className` value with:

```tsx
          <Button
            key={`${img.filename}-${i}`}
            variant="ghost"
            onClick={() => setSelected(img)}
            className="aspect-square h-auto cursor-pointer overflow-hidden rounded-xl border border-border/50 p-0 transition-all hover:border-primary/60 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.5),0_4px_20px_-2px_hsl(var(--primary)/0.35)] focus-visible:border-primary/60 focus-visible:shadow-[0_0_0_1px_hsl(var(--primary)/0.5),0_4px_20px_-2px_hsl(var(--primary)/0.35)]"
          >
```

- [ ] **Step 2: Typecheck/lint**

Run: `cd frontend && npm run lint`
Expected: no new errors.

- [ ] **Step 3: Visual + keyboard check**

Hovering a thumbnail shows a primary-colored ring + soft glow; **Tab**bing to a thumbnail shows the *same* glow (keyboard parity); Enter still opens the lightbox. Confirm in light and dark.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/comfy/GenerateUI.tsx
git commit -m "feat(comfy): theme-aware hover+focus glow on gallery thumbnails"
```

---

### Task 5: Inset-surface contrast for control sections

Give the dense left-panel control groups quiet definition (depth, not glow) so they read as crafted panels instead of a flat wall. Introduce one small `Section` wrapper and apply it to each group, removing the now-redundant `<Separator />` dividers.

**Files:**
- Modify: `frontend/components/comfy/GenerateUI.tsx` (add `Section` near `SectionLabel` ~line 112; wrap sections in the left `ScrollArea`, ~lines 556–874; remove `<Separator />` dividers)

- [ ] **Step 1: Add the `Section` wrapper component**

Immediately after the `SectionLabel` component definition (~line 118), add:

```tsx
function Section({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-3 rounded-lg border border-border/40 bg-muted/30 p-3">
      {children}
    </div>
  );
}
```

- [ ] **Step 2: Wrap each control group and drop the dividers**

In the left panel's `ScrollArea > div.space-y-5`, each group currently looks like `<div className="space-y-N"> <SectionLabel>…</SectionLabel> … </div>` separated by `<Separator />`. For EACH of these groups, change the outer wrapper to `<Section>` and delete the `<Separator />` between groups. The groups (identified by their `SectionLabel`/header text) are:

1. **Workflow** (Architecture switcher)
2. **Prompts**
3. **Model — ANIMA** and **Model — SDXL** (two conditional blocks; wrap each)
4. **Size**
5. **Sampler**
6. **Post-processing** (SDXL-only block)
7. **LoRAs**

Example transformation — Prompts group, from:

```tsx
              {/* Prompts */}
              <div className="space-y-3">
                <SectionLabel>Prompts</SectionLabel>
                {/* …textareas… */}
              </div>

              <Separator />
```

to:

```tsx
              {/* Prompts */}
              <Section>
                <SectionLabel>Prompts</SectionLabel>
                {/* …textareas… */}
              </Section>
```

(The `<Separator />` that followed it is removed — the surface provides the separation.) Apply the same wrap to every group listed above. Keep the inner content of each group exactly as-is. Leave the outer `<div className="space-y-5 p-4">` container as the spacing between `Section`s.

- [ ] **Step 3: Remove the now-unused Separator import (only if no `<Separator />` remain)**

Run: `cd frontend && npm run lint`. If lint flags `Separator` as unused, remove its import (`import { Separator } from '@/components/ui/separator';`, ~line 34). If any `<Separator />` legitimately remains, leave the import.

- [ ] **Step 4: Visual check**

With ComfyUI connected: each control group now sits in its own subtly-bordered, faintly-filled panel; the panel reads as grouped, not glowing, and stays easy to scan. Confirm dense controls (sliders, selects, inputs) remain legible in light and dark.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/comfy/GenerateUI.tsx
git commit -m "feat(comfy): inset-surface contrast for generate-page control sections"
```

---

## Self-Review

- **Spec coverage:** Generate CTA glow (Task 1 ✓), generation-active shimmer (Task 2 ✓), gallery empty-state glow + copy (Task 3 ✓), thumbnail hover — *re-scoped* from `hover-border-gradient` to Tailwind glow with rationale (Task 4 ✓), calm-but-contrastive control sections (Task 5 ✓), retire hand-built effects / no new installs (honored throughout ✓). Architecture-switcher "gentle active accent" from the spec is intentionally folded into Task 5's surface treatment rather than a separate change, to avoid over-styling the custom pill.
- **Placeholder scan:** none — every code step shows complete JSX; Task 5's repetition is a single fully-specified wrapper applied to an enumerated list.
- **Type/name consistency:** `ShinyButton`, `ShineBorder`, `Section` used consistently; imports defined before use (Task 1 → ShinyButton, Task 2 → ShineBorder, Task 5 → Section).
