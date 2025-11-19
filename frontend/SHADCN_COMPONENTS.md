# shadcn/ui Components

✅ **NOW PROPERLY SET UP!**

## What's Included

### Components (`components/ui/`)

1. **Button** - `components/ui/button.tsx`
   - Variants: default, destructive, outline, secondary, ghost, link
   - Sizes: default, sm, lg, icon

2. **Input** - `components/ui/input.tsx`
   - Text input with proper styling
   - Supports all HTML input types

3. **Card** - `components/ui/card.tsx`
   - Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter

4. **Label** - `components/ui/label.tsx`
   - Form labels with proper accessibility

5. **Progress** - `components/ui/progress.tsx`
   - Progress bars with animations

### Utilities

- **`lib/utils.ts`** - `cn()` function for className merging
- **`components.json`** - shadcn configuration

## Usage Examples

### Button

```tsx
import { Button } from "@/components/ui/button"

// Default button
<Button>Click me</Button>

// Variants
<Button variant="destructive">Delete</Button>
<Button variant="outline">Cancel</Button>
<Button variant="ghost">Ghost</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="lg">Large</Button>
<Button size="icon"><Icon /></Button>

// With loading state
<Button disabled={loading}>
  {loading ? "Loading..." : "Submit"}
</Button>
```

### Input

```tsx
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

<div>
  <Label htmlFor="email">Email</Label>
  <Input
    id="email"
    type="email"
    placeholder="you@example.com"
    value={email}
    onChange={(e) => setEmail(e.target.value)}
  />
</div>
```

### Card

```tsx
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Card description goes here</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Card content</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

### Progress

```tsx
import { Progress } from "@/components/ui/progress"

<Progress value={progress} />  // 0-100
```

### Using cn() for className

```tsx
import { cn } from "@/lib/utils"

<div className={cn(
  "base-classes",
  isActive && "active-classes",
  "more-classes"
)}>
  Content
</div>
```

## Refactoring Existing Components

### Before (Raw Tailwind):

```tsx
<button className="flex items-center gap-2 bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-600 disabled:opacity-50">
  Click me
</button>
```

### After (shadcn Button):

```tsx
import { Button } from "@/components/ui/button"

<Button className="gap-2">
  Click me
</Button>
```

### Before (Raw Input):

```tsx
<input
  type="text"
  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
  placeholder="Enter text"
/>
```

### After (shadcn Input):

```tsx
import { Input } from "@/components/ui/input"

<Input placeholder="Enter text" />
```

## Next Steps

To fully migrate to shadcn, update:

1. **TrainingConfig.tsx** - Replace raw buttons/inputs
2. **TrainingMonitor.tsx** - Use Card component
3. **DatasetUploader.tsx** - Use Button, Progress
4. **FileManager.tsx** - Use Card, Button

## Adding More Components

Need more shadcn components? Add them to `components/ui/`:

Common ones to add:
- Select (dropdowns)
- Tabs
- Dialog (modals)
- Alert
- Badge
- Checkbox
- RadioGroup
- Switch
- Textarea
- Tooltip

## Dependencies

All required Radix UI packages are in `package.json`:
- @radix-ui/react-slot
- @radix-ui/react-label
- @radix-ui/react-progress
- @radix-ui/react-select
- @radix-ui/react-tabs

Run `npm install` to install them!

---

**shadcn/ui is NOW properly set up!** 🎉

The components are ready to use - just import and go!
