# Ktiseos-Nyx-Trainer Frontend

Modern Next.js web interface for LoRA training. Replaces Jupyter notebooks with a professional, bug-free UI!

## 🎉 No More Race Conditions!

This Next.js app uses:
- **WebSockets** for real-time updates (no polling!)
- **Proper state management** (no widget chaos!)
- **Async file uploads** with progress tracking
- **Built-in file manager** (goodbye Jupyter Lab!)

## Features

- 📁 **File Manager** - Browse, upload, delete files with drag & drop
- 🖼️ **Dataset Manager** - Upload images, auto-tag with WD14
- ⚙️ **Training UI** - Configure and monitor LoRA training
- 📊 **Real-time Logs** - WebSocket-based log streaming
- 🎨 **Modern Design** - Tailwind CSS with dark mode support

## Development Setup

### Prerequisites

- Node.js 18+ (already in VastAI base image!)
- Backend API running on port 8000

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
├── app/                    # Next.js 14 App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Homepage
│   ├── files/            # File manager page
│   ├── dataset/          # Dataset management
│   ├── training/         # Training UI
│   └── globals.css       # Global styles
├── components/            # React components
│   └── FileManager.tsx   # File browser component
├── lib/                   # Utilities
│   └── api.ts            # API client
├── public/               # Static assets
├── package.json          # Dependencies
└── tsconfig.json         # TypeScript config
```

## API Integration

The frontend connects to the FastAPI backend on port 8000.

Environment variables (`.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

All API calls are centralized in `lib/api.ts`.

## Adding New Pages

1. Create directory in `app/`: `app/mypage/`
2. Add `page.tsx`:
```typescript
export default function MyPage() {
  return <div>My Page</div>;
}
```
3. Link from homepage or navigation

## Component Development

Use TypeScript for all components:

```typescript
'use client'; // For client-side interactivity

import { useState } from 'react';

export default function MyComponent() {
  const [state, setState] = useState('');

  return <div>{state}</div>;
}
```

## Styling

Uses Tailwind CSS utility classes:

```tsx
<div className="p-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
  Button
</div>
```

## WebSocket Usage

For real-time updates (training logs, progress):

```typescript
import { trainingAPI } from '@/lib/api';

const ws = trainingAPI.connectLogs(
  (data) => console.log('Log:', data),
  (error) => console.error('WS Error:', error)
);

// Later: ws.close();
```

## Deployment

### VastAI (Automated)

The `vastai_setup.sh` script automatically:
1. Installs Node.js dependencies
2. Builds the production app
3. Starts on port 3000

### Manual Deployment

```bash
# Build
npm run build

# Start
npm start
```

Or with Docker:
```bash
docker build -t ktiseos-nyx-frontend .
docker run -p 3000:3000 ktiseos-nyx-frontend
```

## Troubleshooting

### "Cannot connect to API"

Check:
1. Backend is running on port 8000
2. CORS is enabled in FastAPI
3. Environment variable is set

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

Tailwind requires a build step.

## Contributing

1. Create feature branch
2. Make changes
3. Test locally
4. Submit PR

---

**Much better than Jupyter widgets!** 🚀
