# Ktiseos-Nyx-Trainer Frontend

Modern Next.js web interface for LoRA training. Replaces Jupyter notebooks with a professional, accessible, bug-free UI!

## 🎉 Recent Updates

### Security Patch (v15.4.7)
- ✅ Upgraded Next.js to 15.4.7 - **0 security vulnerabilities**
- ✅ Fixed critical authorization bypass vulnerability
- ✅ Resolved 7 dependabot security alerts

### Accessibility & UI Enhancements
- ✅ Replaced Base UI components with Radix-based alternatives
- ✅ Added professional DiceUI components (tags-input, mask-input, hitbox)
- ✅ Improved ScrollArea with proper ref forwarding
- ✅ Added @paceui GitHub star counter component
- ✅ Enhanced accessibility utilities (visually-hidden, hitbox)

### New Features
- 🖼️ **Image Editor Page** - Gallery view + Fabric.js-powered image editing
  - Crop, blur, draw, and annotate images
  - Browse dataset images in gallery grid
  - Navigate between images with Previous/Next
  - Save edited images directly to filesystem
- 🏷️ **Tag Editor** - Comprehensive tag management with image gallery
- 📊 **Training Monitor** - Real-time training progress and logs

## Features

### Core Functionality
- 📁 **File Manager** - Browse, upload, delete files with drag & drop
- 🖼️ **Dataset Manager** - Upload images, auto-tag with WD14
- ✏️ **Image Editor** - Crop, blur, and edit dataset images with gallery view
- 🏷️ **Tag Editor** - Bulk tag operations, trigger word injection, visual gallery
- ⚙️ **Training UI** - Configure and monitor LoRA training
- 📊 **Real-time Logs** - WebSocket-based log streaming
- 🎨 **Modern Design** - Tailwind CSS with comprehensive light/dark theme support
- ♿ **Accessibility** - Screen reader support, keyboard navigation, ARIA labels

### UI Components Library

**shadcn/ui Ecosystem:**
- Multiple component registries (@aceternity, @kokonutui, @magicui, @coss, @hextaui, @paceui, etc.)
- Over 50+ pre-built accessible components
- Customizable with Tailwind CSS
- Full TypeScript support

**Key Components:**
- **DiceUI**: Combobox, Tags Input, Mask Input, Hitbox, Visually Hidden
- **Coss**: ScrollArea, Textarea (Radix-based)
- **HextaUI**: Enhanced Alert Dialog
- **PaceUI**: GitHub Star Counter with GSAP animations
- **Image Editor**: Professional editing with @ozdemircibaris/react-image-editor

## Development Setup

### Prerequisites

- Node.js 18.18.0+ (required for Next.js 15)
- Backend API running on port 8000
- npm or pnpm package manager

### Install Dependencies

```bash
cd frontend
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js 15 App Router
│   ├── layout.tsx         # Root layout with theme provider
│   ├── page.tsx           # Homepage with hero animation
│   ├── about/            # About page
│   ├── calculator/       # LoRA step calculator
│   ├── dashboard/        # Main dashboard
│   ├── dataset/          # Dataset management
│   │   ├── page.tsx      # Dataset uploader
│   │   ├── tags/         # Tag editor with gallery
│   │   └── editor/       # Image editor with gallery (NEW!)
│   ├── docs/             # Documentation
│   ├── files/            # File manager
│   ├── models/           # Model downloader
│   ├── settings/         # Application settings
│   ├── training/         # Training configuration
│   ├── utilities/        # Post-training utilities
│   └── globals.css       # Global styles & theme variables
├── components/            # React components
│   ├── blocks/           # Page blocks (hero, nav, footer)
│   ├── effects/          # Visual effects components
│   ├── kibo-ui/          # Kibo UI components
│   ├── training/         # Training-specific components
│   ├── ui/               # shadcn/ui component library
│   ├── Breadcrumbs.tsx   # Navigation breadcrumbs
│   ├── DatasetUploader.tsx  # Dataset upload interface
│   ├── FileBrowser.tsx   # File browser component
│   ├── FileManager.tsx   # File manager with tree view
│   └── theme-provider.tsx  # Dark/light theme provider
├── lib/                   # Utilities
│   ├── api.ts            # API client with typed endpoints
│   ├── utils.ts          # Helper functions
│   └── compose-refs.ts   # Ref composition utility
├── hooks/                # Custom React hooks
│   ├── useSettings.ts    # Settings management
│   └── use-download-file.ts  # File download hook
├── public/               # Static assets
├── components.json       # shadcn/ui configuration
├── package.json          # Dependencies
├── tailwind.config.ts    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

## Pages Overview

### Dataset Management
- **`/dataset`** - Upload and manage training datasets
- **`/dataset/tags`** - Edit tags with visual gallery, bulk operations
- **`/dataset/editor`** - Edit images with gallery view (crop, blur, annotate)

### Training
- **`/training`** - Configure training parameters
- **`/calculator`** - Calculate optimal training steps
- **`/utilities`** - Post-training utilities (resize, convert)

### Files & Models
- **`/files`** - File manager with tree view
- **`/models`** - Download models from HuggingFace/Civitai

### General
- **`/`** - Homepage with animated hero
- **`/dashboard`** - Main dashboard
- **`/settings`** - Application settings
- **`/docs`** - Documentation
- **`/about`** - About page

## API Integration

The frontend connects to the FastAPI backend on port 8000.

Environment variables (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

All API calls are centralized in `lib/api.ts` with full TypeScript types.

### API Modules

```typescript
import { datasetAPI, trainingAPI, filesAPI, modelsAPI } from '@/lib/api';

// Dataset operations
await datasetAPI.list();
await datasetAPI.getImagesWithTags(datasetPath);
await datasetAPI.updateTags(imagePath, tags);

// Training operations
await trainingAPI.start(config);
await trainingAPI.stop();
trainingAPI.connectLogs(onLog, onError);

// File operations
await filesAPI.list(path);
await filesAPI.read(path);
await filesAPI.write(path, content);

// Model operations
await modelsAPI.download(url, savePath);
```

## Component Development

### Using shadcn/ui Components

```bash
# Add components from shadcn registry
npx shadcn@latest add button

# Add from custom registries
npx shadcn@latest add @diceui/combobox
npx shadcn@latest add @coss/scroll-area
```

### Creating Custom Components

```typescript
'use client'; // For client-side interactivity

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export default function MyComponent() {
  const [state, setState] = useState('');

  return (
    <Card>
      <Button onClick={() => setState('clicked')}>
        {state || 'Click me'}
      </Button>
    </Card>
  );
}
```

## Styling & Theming

### Tailwind CSS

Uses Tailwind v4 utility classes:

```tsx
<div className="p-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90">
  Button
</div>
```

### Theme System

Full light/dark mode support with CSS variables:

```tsx
import { useTheme } from 'next-themes';

function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
      Toggle Theme
    </button>
  );
}
```

Theme colors defined in `app/globals.css` using CSS custom properties.

## Image Editor Usage

The new image editor combines gallery browsing with powerful editing:

```typescript
// Features available:
- Gallery grid view of all dataset images
- Click to select image for editing
- Crop with custom aspect ratios
- Blur sensitive areas
- Draw shapes and annotations
- Undo/Redo support
- Previous/Next navigation
- Save directly to filesystem
```

### Editor Tools

1. **Crop** - Resize to training resolutions (512x512, 1024x1024)
2. **Blur** - Hide watermarks, faces, sensitive content
3. **Draw** - Add annotations and shapes
4. **Text** - Add text labels
5. **Shapes** - Circles, rectangles, lines

## WebSocket Usage

For real-time updates (training logs, progress):

```typescript
import { trainingAPI } from '@/lib/api';

const ws = trainingAPI.connectLogs(
  (data) => {
    console.log('Log:', data.message);
    console.log('Progress:', data.progress);
  },
  (error) => console.error('WS Error:', error)
);

// Cleanup: ws.close();
```

## Deployment

### VastAI (Automated)

The `vastai_setup.sh` script automatically:
1. Installs Node.js 18+ dependencies
2. Builds the production app
3. Starts on port 3000 (with proxy on port 80)

### Docker Deployment

```bash
# Build image
docker build -t ktiseos-nyx-frontend .

# Run container
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000/api \
  ktiseos-nyx-frontend
```

### Manual Deployment

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Troubleshooting

### "Cannot connect to API"

Check:
1. Backend is running: `curl http://localhost:8000/api/health`
2. CORS is enabled in FastAPI
3. Environment variable: `echo $NEXT_PUBLIC_API_URL`
4. Firewall rules allow port 8000

### "Module not found"

```bash
rm -rf node_modules .next
npm install
npm run dev
```

### Styles not loading

```bash
npm run build
```

Tailwind requires a build step for production.

### Image editor not loading

The image editor uses dynamic imports to avoid SSR issues:
- Fabric.js requires browser environment
- Component loads only on client side
- Check browser console for canvas errors

### Type errors after component installation

```bash
# Regenerate TypeScript types
npm run build

# If using VS Code, reload window
Cmd/Ctrl + Shift + P → "Developer: Reload Window"
```

## Contributing

1. Create feature branch from `main`
2. Make changes with TypeScript
3. Test locally: `npm run build && npm run dev`
4. Ensure 0 build errors and warnings
5. Submit PR with clear description

### Code Style

- Use TypeScript for all components
- Follow existing naming conventions
- Use `'use client'` directive for interactive components
- Prefer server components when possible
- Use shadcn/ui components for consistency
- Add JSDoc comments for complex functions

## Security

- ✅ Regular dependency updates via dependabot
- ✅ No known security vulnerabilities
- ✅ HTTPS in production (via VastAI proxy)
- ✅ API authentication (when backend implements it)
- ✅ Input sanitization on forms
- ✅ XSS protection via React
- ✅ CSRF protection via FastAPI

## Performance

- **Build size**: ~100KB shared JS (gzipped)
- **First Load JS**: ~100-180KB per page
- **Image optimization**: Next.js automatic optimization
- **Code splitting**: Automatic per-page splitting
- **Dynamic imports**: Heavy components (editor) load on demand

## Tech Stack

- **Framework**: Next.js 15.4.7 (App Router)
- **Language**: TypeScript 5.4+
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui + Radix UI
- **Image Editing**: Fabric.js (@ozdemircibaris/react-image-editor)
- **Icons**: Lucide React + Tabler Icons
- **Animations**: Framer Motion
- **Theme**: next-themes
- **API Client**: Fetch API with TypeScript
- **WebSockets**: Native WebSocket API

---

**Much better than Jupyter widgets!** 🚀

Built with ❤️ by Bowen (with Claude Code assistance)
