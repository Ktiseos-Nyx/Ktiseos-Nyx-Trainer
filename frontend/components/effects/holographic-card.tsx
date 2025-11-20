/**
 * HolographicCard - Sleek shimmer effect for cards
 *
 * Inspired by:
 * - https://uiverse.io/Itskrish01/wise-wombat-45
 * - https://freefrontend.com/tailwind-shimmer/
 */

import { cn } from '@/lib/utils'

interface HolographicCardProps {
  children: React.ReactNode
  speed?: 'slow' | 'medium' | 'fast'
  className?: string
}

export function HolographicCard({
  children,
  speed = 'medium',
  className
}: HolographicCardProps) {
  const speeds = {
    slow: '2000ms',
    medium: '1000ms',
    fast: '600ms',
  }

  return (
    <div
      className={cn(
        'group relative rounded-2xl overflow-hidden',
        'backdrop-blur-xl border-2 border-purple-500/30',
        'bg-gradient-to-br from-purple-900/40 via-slate-900/60 to-slate-950/80',
        'shadow-2xl hover:shadow-purple-500/30 hover:shadow-2xl',
        'hover:scale-[1.02] hover:-translate-y-1 active:scale-95',
        'transition-all duration-500 ease-out',
        'hover:border-purple-400/60',
        className
      )}
    >
      {/* Sliding shimmer effect - inspired by Uiverse.io */}
      <div
        className={cn(
          'absolute inset-0',
          'bg-gradient-to-r from-transparent via-purple-400/30 to-transparent',
          '-translate-x-full group-hover:translate-x-full',
          'transition-transform ease-out'
        )}
        style={{
          transitionDuration: speeds[speed]
        }}
      />

      {/* Ambient glow overlay */}
      <div
        className={cn(
          'absolute inset-0 rounded-2xl',
          'bg-gradient-to-r from-purple-500/10 via-purple-400/20 to-purple-500/10',
          'opacity-0 group-hover:opacity-100',
          'transition-opacity duration-500'
        )}
      />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  )
}
