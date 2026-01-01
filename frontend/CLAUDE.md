# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: ALWAYS USE SHADCN/UI COMPONENTS

**DO NOT USE RAW HTML FORM ELEMENTS!** This project uses shadcn/ui components for ALL UI elements:

- ❌ **NO** `<select>` - Use `<Select>` from `@/components/ui/select`
- ❌ **NO** `<input>` - Use `<Input>` from `@/components/ui/input`
- ❌ **NO** `<button>` - Use `<Button>` from `@/components/ui/button`
- ❌ **NO** `<textarea>` - Use `<Textarea>` from `@/components/ui/textarea`
- ✅ **YES** - Import from `components/ui/*` or use FormField wrappers

**Why:** shadcn components have built-in theme-aware styling, accessibility, and proper light/dark mode support. Raw HTML elements break in dark mode and require manual Tailwind fixes.

**See:** `components/training/fields/FormFields.tsx` for examples of proper usage.

## Repository Context

This is the **frontend** subdirectory of the Ktiseos-Nyx-Trainer project. The parent repository is a **web-based LoRA training environment** with a FastAPI backend. This frontend provides a modern Next.js web interface for training LoRA models.

**Architecture**: Next.js 15 (frontend) → FastAPI (backend) → Services (business logic) → Kohya SS (training)

**For backend/training documentation, see the parent directory's CLAUDE.md.**

## Development Commands

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```
Opens development server at http://localhost:3000

### Build for Production
```bash
npm run build
npm start
```

### Linting
```bash
npm run lint
```

## Architecture Overview

### Tech Stack
- **Framework**: Next.js 15.4.7 (App Router, React 19 RC)
- **Language**: TypeScript 5.4+
- **Styling**: Tailwind CSS v4 with CSS custom properties for theming
- **UI Components**: shadcn/ui ecosystem (Radix UI primitives)
- **API Integration**: FastAPI backend on port 8000
- **State Management**: React hooks + TanStack Query
- **WebSockets**: Native WebSocket API for real-time logs

### App Router Structure (Next.js 15)

The project uses Next.js 15's App Router with the following convention:
- `app/` - All routes and pages
- `app/layout.tsx` - Root layout with theme provider, navbar, footer
- `app/page.tsx` - Homepage with animated hero
- `app/*/page.tsx` - Individual route pages
- `components/` - Reusable React components
- `lib/` - Utilities and API client
- `hooks/` - Custom React hooks

**Key Routes:**
- `/` - Homepage
- `/dataset` - Dataset uploader
- `/dataset/[name]/tags` - Tag editor with gallery
- `/dataset/[name]/auto-tag` - Auto-tagging interface
- `/files` - File manager with tree view
- `/models` - Model downloader (HuggingFace/Civitai)
- `/training` - Training configuration
- `/calculator` - LoRA step calculator
- `/utilities` - Post-training utilities
- `/settings` - Application settings

### API Client Architecture

**Centralized API client**: `lib/api.ts`

All backend communication flows through typed API modules:
```typescript
import { datasetAPI, trainingAPI, fileAPI, modelsAPI, utilitiesAPI, captioningAPI } from '@/lib/api';

// Examples
await datasetAPI.list();
await datasetAPI.tag(config);  // WD14 tagging
await captioningAPI.startBLIP(config);  // BLIP captioning
await captioningAPI.startGIT(config);   // GIT captioning
await trainingAPI.start(config);
await fileAPI.upload(file, destination);
```

**WebSocket Integration:**
- Training logs: `trainingAPI.connectLogs(jobId, onMessage, onError)`
- Tagging logs: `datasetAPI.connectTaggingLogs(jobId, onMessage, onError)`
- Captioning logs: `captioningAPI.connectLogs(jobId, onMessage, onError)`

**Environment Variables:**
- `NEXT_PUBLIC_API_URL` - Backend API base URL (default: http://localhost:8000/api)
- Backend URL automatically converted to WebSocket URL (`http://` → `ws://`, `https://` → `wss://`)

### Component Architecture

**shadcn/ui Component System:**
- All UI components in `components/ui/` from shadcn/ui ecosystem
- Multiple registries used: @diceui, @coss, @hextaui, @paceui, @kokonutui
- Components are copy-pasted and customizable (not npm packages)
- See `SHADCN_COMPONENTS.md` for full component inventory

**Custom Components:**
- `components/blocks/` - Page sections (hero, navigation, footer)
- `components/training/` - Training-specific components
- `components/effects/` - Visual effects (gradients, glitch, neon)
- Root-level components: `DatasetUploader.tsx`, `FileManager.tsx`, `FileBrowser.tsx`

**Component Development Rules:**
1. **Always use shadcn/ui components first** - Check `SHADCN_COMPONENTS.md` before building custom
2. **Accessibility is mandatory** - See `FRONTEND_GUIDELINES.md`
3. **Use `'use client'` directive** - Required for client-side interactivity (state, effects, event handlers)
4. **Prefer server components** - Use server components when no client-side features needed
5. **TypeScript required** - All components must be typed

### Theming System

**CSS Custom Properties** (defined in `app/globals.css`):
```css
:root {
  --background: ...
  --foreground: ...
  --primary: ...
  --secondary: ...
  /* etc */
}

.dark {
  --background: ...
  /* dark mode overrides */
}
```

**Usage in Tailwind:**
```tsx
<div className="bg-primary text-primary-foreground">
```

**Theme Toggle:**
```typescript
import { useTheme } from 'next-themes';
const { theme, setTheme } = useTheme();
```

## Critical Development Patterns

### API Response Handling

All API calls use centralized error handling in `lib/api.ts`:
```typescript
async function handleResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
```

### WebSocket Pattern

```typescript
const ws = trainingAPI.connectLogs(
  (data) => {
    console.log('Log:', data.message);
    console.log('Progress:', data.progress);
  },
  (error) => console.error('WS Error:', error)
);

// Cleanup on unmount
useEffect(() => {
  return () => ws.close();
}, []);
```

### File Upload Pattern

```typescript
const formData = new FormData();
formData.append('file', file);

await fileAPI.upload(file, destination);
```

### Dynamic Imports (for client-only libraries)

Some libraries (like Fabric.js for image editing) require browser environment:
```typescript
const ImageEditor = dynamic(() => import('@/components/ImageEditor'), {
  ssr: false,
  loading: () => <Spinner />
});
```

## Path Aliases

TypeScript path aliases configured in `tsconfig.json`:
```typescript
import { Button } from '@/components/ui/button';  // ✅
import Button from '../../../components/ui/button';  // ❌ Don't use relative paths
```

## Docker & Deployment

**Development:** Runs standalone via `npm run dev`

**Production (Docker Compose):**
- Frontend: Port 3000
- Backend: Port 8000
- Proxy: Backend API calls via Next.js rewrites (see `next.config.js`)

**VastAI Deployment:**
- Automated via `vastai_setup.sh` in parent directory
- Builds production bundle
- Runs on port 3000 (proxied to port 80)

## Accessibility Requirements

From `FRONTEND_GUIDELINES.md`:

1. **Semantic HTML** - Use proper elements (`<main>`, `<nav>`, `<button>`, etc.)
2. **Keyboard Navigation** - All interactive elements must work with keyboard
3. **ARIA Attributes** - Use roles and labels for custom components
4. **Image Alt Text** - All images must have descriptive `alt` or `alt=""`
5. **Form Labels** - All inputs must have associated `<label>`
6. **Color Contrast** - Ensure sufficient contrast for readability

**This is non-negotiable.** Accessibility is a core requirement, not an afterthought.

## Common Development Tasks

### Adding a New Page

1. Create `app/new-page/page.tsx`:
```typescript
export default function NewPage() {
  return <main className="container mx-auto p-8">...</main>;
}
```

2. Add navigation link in `components/blocks/navigation/navbar.tsx`

### Adding a shadcn/ui Component

```bash
npx shadcn@latest add button
# Or from custom registries:
npx shadcn@latest add @diceui/combobox
```

### Creating a New API Endpoint

1. Add interface to `lib/api.ts`:
```typescript
export interface MyResponse {
  data: string;
}
```

2. Add to appropriate API module:
```typescript
export const myAPI = {
  getData: async (): Promise<MyResponse> => {
    const response = await fetch(`${API_BASE}/my-endpoint`);
    return handleResponse(response);
  },
};
```

### Working with Training Configuration

Training config types are defined in `lib/api.ts` as `TrainingConfig` interface. This massive interface (560+ lines) defines all possible training parameters for:
- SD 1.5, SDXL, Flux, SD3, Lumina models
- LoRA types (LoRA, LoCon, LoHa, LoKR)
- Optimizers, schedulers, advanced parameters

**Key files:**
- `components/training/TrainingConfig.tsx` - Main training form
- `components/training/TrainingDefaults.tsx` - Default values
- `components/training/TrainingMonitor.tsx` - Real-time monitoring

## Known Issues & Quirks

### React 19 RC Overrides

Package.json uses React 19 RC with overrides for framer-motion compatibility:
```json
"overrides": {
  "framer-motion": {
    "react": "19.0.0-rc-66855b96-20241106",
    "react-dom": "19.0.0-rc-66855b96-20241106"
  }
}
```

### Tailwind v4 Migration

Uses new Tailwind v4 with PostCSS plugin:
- Import in `app/globals.css`: `@import "tailwindcss";`
- Config in `tailwind.config.ts` uses extended theme with CSS variables

### SSR vs Client-Side Rendering

**Server Components (default):**
- Faster initial load
- Better SEO
- No `useState`, `useEffect`, or browser APIs

**Client Components (`'use client'`):**
- Required for interactivity
- Can use hooks and browser APIs
- Mark with `'use client'` at top of file

## Project Structure

```
Ktiseos-Nyx-Trainer/           # Parent (Full web application)
├── frontend/                   # THIS DIRECTORY (Next.js)
│   ├── app/                   # Next.js routes
│   ├── components/            # React components
│   └── lib/                   # API client
├── api/                        # FastAPI backend
│   ├── main.py                # FastAPI app
│   └── routes/                # API endpoints
├── services/                   # Business logic layer
│   ├── *_service.py           # Service modules
│   └── trainers/              # Training integration
├── trainer/derrian_backend/   # Vendored Kohya SS + LyCORIS
└── CLAUDE.md                  # Parent docs
```

**Backend serves API on port 8000, frontend consumes it on port 3000.**

The frontend provides:
- Modern, accessible web UI for LoRA training
- Real-time monitoring via WebSockets
- Professional UI with accessibility
- Neurodivergent-friendly structured workflows

## Important Reminders

1. **Check parent CLAUDE.md** - Backend architecture and training logic documented there
2. **Accessibility is mandatory** - See `FRONTEND_GUIDELINES.md`
3. **Use shadcn/ui components** - Check `SHADCN_COMPONENTS.md` before building custom
4. **TypeScript everywhere** - No plain JavaScript files
5. **API client centralization** - All backend calls through `lib/api.ts`
6. **Security updates** - Currently at 0 vulnerabilities (Next.js 15.4.7)
