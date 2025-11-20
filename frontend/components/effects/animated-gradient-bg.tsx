import { cn } from '@/lib/utils'

interface AnimatedGradientBgProps {
  children: React.ReactNode
  variant?: 'dusk' | 'cotton-candy' | 'watermelon' | 'ocean'
  speed?: 'slow' | 'medium' | 'fast'
  className?: string
}

export function AnimatedGradientBg({
  children,
  variant = 'dusk',
  speed = 'slow',
  className
}: AnimatedGradientBgProps) {
  const speeds = {
    slow: 'animate-[gradient_15s_ease_infinite]',
    medium: 'animate-[gradient_10s_ease_infinite]',
    fast: 'animate-[gradient_5s_ease_infinite]',
  }

  // Animated gradients (need background-size: 200% for animation to work)
  const gradients = {
    dusk: 'bg-gradient-to-br from-purple-900 via-blue-950 to-slate-950',
    'cotton-candy': 'bg-gradient-to-br from-pink-400/40 via-blue-400/35 via-purple-400/40 to-slate-900',
    watermelon: 'bg-gradient-to-br from-pink-950 via-green-950 to-slate-950',
    ocean: 'bg-gradient-to-br from-blue-900 via-teal-800 via-cyan-900 to-slate-950',
  }

  return (
    <div className={cn('relative overflow-hidden', className)}>
      {/* Animated gradient background */}
      <div
        className={cn(
          'absolute inset-0',
          gradients[variant],
          speeds[speed]
        )}
        style={{
          backgroundSize: '200% 200%'
        }}
      />

      {/* Content layer */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  )
}
