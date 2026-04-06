# Visual Effects Component Library

A collection of reusable, copy-paste React/Next.js visual effect components with no external dependencies beyond the basics.

## 📋 Requirements

All components require:
- **Next.js 14+** (App Router with `'use client'`)
- **React 18+**
- **TypeScript**
- **Tailwind CSS**
- **`cn` utility** for className merging (see below)

### Setting Up `cn` Utility

The components use a `cn` utility function for merging Tailwind classes.

**If you already use shadcn/ui:** You're all set! These components are fully compatible and use the same `cn` utility.

**If you don't have shadcn/ui:** Create `/lib/utils.ts`:

```typescript
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

Then install dependencies:
```bash
npm install clsx tailwind-merge
```

**Compatibility:** These components work perfectly alongside shadcn/ui components and follow the same patterns. They're designed to integrate seamlessly into shadcn/ui projects!

## 📦 Installation

**No registry needed!** Just copy-paste the component files you want into your project.

1. Create `/components/effects/` directory in your project
2. Copy the component files you need
3. Add required CSS keyframes to your `globals.css` (see below)
4. Import and use!

### Required Global CSS

Add these keyframes to your `app/globals.css` or `styles/globals.css`:

```css
@theme inline {
  @keyframes shimmer {
    0% {
      background-position: -200% 0;
    }
    100% {
      background-position: 200% 0;
    }
  }

  @keyframes gradient {
    0%, 100% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
  }
}
```

---

## 🎨 Text Effects

### TextGlow

Adds glowing text shadow effects with customizable colors and intensity.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { TextGlow } from '@/components/effects'

<TextGlow color="purple" intensity="intense">
  Glowing Text
</TextGlow>
```

**Props:**
- `color`: `'purple' | 'blue' | 'green' | 'pink'` (default: `'purple'`)
- `intensity`: `'subtle' | 'medium' | 'intense' | 'extreme'` (default: `'medium'`)
- `className`: Optional Tailwind classes

---

### TextOutline

Creates outlined text with customizable stroke color, width, and fill.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { TextOutline } from '@/components/effects'

<TextOutline strokeColor="#ffffff" strokeWidth={2}>
  OUTLINED
</TextOutline>
```

**Props:**
- `strokeColor`: CSS color string (default: `'#ffffff'`)
- `strokeWidth`: Number in pixels (default: `2`)
- `fillColor`: CSS color string (default: `'transparent'`)
- `className`: Optional Tailwind classes

---

### NeonText

Neon sign effect with optional flickering animation.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { NeonText } from '@/components/effects'

<NeonText color="cyan" flicker={true}>
  NEON SIGN
</NeonText>
```

**Props:**
- `color`: `'cyan' | 'pink' | 'purple' | 'green'` (default: `'cyan'`)
- `flicker`: Boolean to enable flickering (default: `false`)
- `className`: Optional Tailwind classes

---

### GlitchText

CSS-based glitch effect with RGB channel separation.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { GlitchText } from '@/components/effects'

<GlitchText intensity="wild">
  SYSTEM FAILURE
</GlitchText>
```

**Props:**
- `intensity`: `'subtle' | 'medium' | 'wild'` (default: `'medium'`)
- `className`: Optional Tailwind classes

---

### MatrixGlitchText

Constantly cycling character substitution with Japanese katakana, Cyrillic, and symbols. True Matrix effect!

**Dependencies:** Standard requirements only

**Font Support:** Uses Unicode characters (Katakana, Cyrillic) - 99%+ browser support, no special fonts needed.

**Usage:**
```tsx
import { MatrixGlitchText } from '@/components/effects'

<MatrixGlitchText glitchSpeed="fast">
  NEURAL INTERFACE
</MatrixGlitchText>
```

**Props:**
- `glitchSpeed`: `'slow' | 'medium' | 'fast'` (default: `'medium'`)
  - `slow`: 150ms per frame
  - `medium`: 80ms per frame
  - `fast`: 40ms per frame
- `className`: Optional Tailwind classes

**How it works:**
- Each character independently has a 40% chance to swap to a random glitch character every frame
- Glitch characters include: Japanese Katakana, Cyrillic letters, symbols, numbers
- Always displays with green glow for Matrix aesthetic

---

## 🃏 Card & Container Effects

### HolographicCard

Animated holographic shimmer effect that travels across the card.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { HolographicCard } from '@/components/effects'

<HolographicCard speed="medium">
  <div className="p-6">
    <h4>Card Content</h4>
  </div>
</HolographicCard>
```

**Props:**
- `speed`: `'slow' | 'medium' | 'fast'` (default: `'medium'`)
- `className`: Optional Tailwind classes

---

### ShimmerBorder

Animated light traveling around card borders.

**Dependencies:**
- Standard requirements
- **Requires `shimmer` keyframes in globals.css** (see Required Global CSS above)

**Usage:**
```tsx
import { ShimmerBorder } from '@/components/effects'

<ShimmerBorder color="purple" speed="medium" borderWidth={2}>
  <div className="p-6">
    <h4>Content</h4>
  </div>
</ShimmerBorder>
```

**Props:**
- `color`: `'purple' | 'blue' | 'rainbow'` (default: `'purple'`)
- `speed`: `'slow' | 'medium' | 'fast'` (default: `'medium'`)
- `borderWidth`: Number in pixels (default: `2`)
- `className`: Optional Tailwind classes

**Note:** If border isn't animating, ensure shimmer keyframes are in your globals.css.

---

### MagneticCard

Interactive card that follows your mouse cursor with magnetic attraction.

**Dependencies:** Standard requirements only (uses React hooks)

**Usage:**
```tsx
import { MagneticCard } from '@/components/effects'

<MagneticCard strength="strong">
  <div className="p-6 bg-slate-800 rounded-lg">
    <h4>Magnetic Content</h4>
  </div>
</MagneticCard>
```

**Props:**
- `strength`: `'subtle' | 'medium' | 'strong'` (default: `'medium'`)
  - `subtle`: 0.15 multiplier
  - `medium`: 0.25 multiplier
  - `strong`: 0.4 multiplier
- `className`: Optional Tailwind classes

**How it works:**
- Calculates mouse position relative to card center
- Applies transform translation based on mouse distance × strength
- Smoothly returns to center when mouse leaves

---

### GradientCard

Tasteful gradient overlays with multiple color themes and intensity levels.

**Dependencies:** Standard requirements only

**Usage:**
```tsx
import { GradientCard } from '@/components/effects'

<GradientCard variant="dusk" intensity="medium">
  <div className="p-6">
    <h4>Card Content</h4>
  </div>
</GradientCard>
```

**Props:**
- `variant`: `'dusk' | 'ocean' | 'watermelon' | 'cotton-candy' | 'shadow'` (default: `'dusk'`)
- `intensity`: `'subtle' | 'medium' | 'vibrant'` (default: `'subtle'`)
- `className`: Optional Tailwind classes

**Color Themes:**
- **Dusk**: Purple/Blue gradient
- **Ocean**: Blue/Teal/Cyan (3-color blend)
- **Watermelon**: Pink/Green
- **Cotton Candy**: Pastel Pink/Blue/Purple
- **Shadow**: No gradient, just depth shadows

---

## 🌌 Background Effects

### AnimatedGradientBg

Slowly animated background gradients for full sections or pages.

**Dependencies:**
- Standard requirements
- **Requires `gradient` keyframes in globals.css** (see Required Global CSS above)

**Usage:**
```tsx
import { AnimatedGradientBg } from '@/components/effects'

<AnimatedGradientBg variant="dusk" speed="slow">
  <div className="p-8">
    <h1>Content Here</h1>
  </div>
</AnimatedGradientBg>
```

**Props:**
- `variant`: `'dusk' | 'cotton-candy' | 'watermelon' | 'ocean'` (default: `'dusk'`)
- `speed`: `'slow' | 'medium' | 'fast'` (default: `'slow'`)
  - `slow`: 15 second animation
  - `medium`: 10 second animation
  - `fast`: 5 second animation
- `className`: Optional Tailwind classes

**Note:** Background must have `backgroundSize: '200% 200%'` for animation to work (already included).

---

## 🚀 Usage Examples

### Import Individual Components

```tsx
import { TextGlow } from '@/components/effects/text-glow'
import { MagneticCard } from '@/components/effects/magnetic-card'
```

### Import from Barrel Export

Create `/components/effects/index.ts`:

```typescript
// Text Effects
export { TextGlow } from './text-glow'
export { TextOutline } from './text-outline'
export { NeonText } from './neon-text'
export { GlitchText } from './glitch-text'
export { MatrixGlitchText } from './matrix-glitch-text'

// Card/Container Effects
export { HolographicCard } from './holographic-card'
export { ShimmerBorder } from './shimmer-border'
export { MagneticCard } from './magnetic-card'
export { GradientCard } from './gradient-card'

// Background Effects
export { AnimatedGradientBg } from './animated-gradient-bg'
```

Then import multiple at once:

```tsx
import { TextGlow, MagneticCard, HolographicCard } from '@/components/effects'
```

---

## 🎨 Combining Effects

Effects can be combined for more complex visuals:

```tsx
<MagneticCard strength="strong">
  <GradientCard variant="ocean" intensity="vibrant">
    <div className="p-8">
      <TextGlow color="blue" intensity="intense">
        <h1 className="text-4xl font-bold">
          Combined Effects!
        </h1>
      </TextGlow>
    </div>
  </GradientCard>
</MagneticCard>
```

---

## 🐛 Troubleshooting

### Animations not working

1. **Check globals.css** - Ensure shimmer/gradient keyframes are added
2. **Check Tailwind config** - Make sure Tailwind is processing your components
3. **Restart dev server** - After adding keyframes, restart Next.js

### TypeScript errors

1. **Check imports** - Ensure `@/` path alias is configured in `tsconfig.json`:
   ```json
   {
     "compilerOptions": {
       "paths": {
         "@/*": ["./*"]
       }
     }
   }
   ```

### MatrixGlitchText showing boxes instead of characters

1. **Font fallback** - Rare on modern systems, but you can add explicit font stack:
   ```tsx
   <MatrixGlitchText
     className="font-['Courier_New','Noto_Sans_Mono',monospace]"
   >
     Text
   </MatrixGlitchText>
   ```

### MagneticCard not moving

1. **Ensure proper nesting** - The card itself moves, so children need proper styling
2. **Check for conflicting CSS** - Other transforms might override the effect
3. **Try stronger strength** - Use `strength="strong"` to see more obvious movement

---

## 🙏 Credits & Inspiration

These components were inspired by various sources in the web development community:

- **HolographicCard shimmer effect**: Inspired by [Uiverse.io](https://uiverse.io/Itskrish01/wise-wombat-45) and [Tailwind Shimmer Effects](https://freefrontend.com/tailwind-shimmer/)
- **Component patterns**: Built on shadcn/ui's design philosophy and best practices

Special thanks to the creators who share their work openly for the community to learn from!

---

## 📄 License

MIT - Free to use in any project, commercial or personal.

## 🤝 Contributing

Since there's no registry, just copy, modify, and make these your own! If you improve them, share with the community.

---

**No installation required • No dependencies • Just copy and use!** ✨
