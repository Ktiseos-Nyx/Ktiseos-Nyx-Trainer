# KNX Blocks Library

Reusable, composable UI blocks built on top of shadcn/ui and other component libraries.

## Philosophy

- **Components** = Individual primitives (shadcn/ui, Aceternity, etc.)
- **Blocks** = Composed sections ready to use in pages
- **Pages** = Combining blocks with content

## Usage

```tsx
import { HeroAnimated, FeatureGrid, Navbar, Footer } from '@/components/blocks';

export default function Page() {
  return (
    <>
      <HeroAnimated
        title="My App"
        description="Cool stuff"
        theme="purple-blue"
      />
      <FeatureGrid features={myFeatures} />
    </>
  );
}
```

## Available Blocks

### Navigation
- **Navbar** - Navigation bar with dropdowns and theme switcher
- **Footer** - Footer with social links and copyright

### Hero
- **HeroAnimated** - Full-screen hero with animated background beams
  - Props: `title`, `subtitle`, `description`, `features`, `ctas`, `theme`
  - Themes: `purple-blue`, `green-earth`, `custom`

### Features
- **FeatureGrid** - Grid layout for feature cards
  - Props: `title`, `features`, `ctaLabel`, `ctaHref`, `columns`
  - Columns: 2, 3, or 4

## Creating New Blocks

1. Create file in appropriate category folder
2. Export component with proper TypeScript types
3. Add to `index.ts` for easy imports
4. Update this README

## Theming

Blocks support multiple themes via props. Add new themes by:

1. Define gradient colors in block component
2. Pass theme name via props
3. Block handles styling automatically

## Cross-Project Usage

These blocks are designed to work across multiple projects:
- **KNX LoRA Trainer** (this project)
- **EcoHub Social Platform** (green-earth theme)
- **Dataset Tools Electron App** (desktop variants)

To use in another project:
1. Copy `/components/blocks/` directory
2. Copy `/components/ui/` (shadcn components)
3. Copy `components.json` registry config
4. Install dependencies from `package.json`
