/**
 * GradientBorderCard - Animated gradient border card with glow effect
 *
 * Inspired by:
 * - https://uiverse.io/nima-mollazadeh/terrible-panda-97
 */

import { cn } from '@/lib/utils'

interface GradientBorderCardProps {
  children: React.ReactNode
  gradient?: 'neon' | 'sunset' | 'ocean' | 'fire' | 'purple'
  className?: string
}

export function GradientBorderCard({
  children,
  gradient = 'neon',
  className
}: GradientBorderCardProps) {
  const gradients = {
    neon: 'linear-gradient(45deg, #0ce39a, #69007f, #fc0987)',
    sunset: 'linear-gradient(45deg, #f59e0b, #ef4444, #ec4899)',
    ocean: 'linear-gradient(45deg, #06b6d4, #3b82f6, #8b5cf6)',
    fire: 'linear-gradient(45deg, #f97316, #dc2626, #7c2d12)',
    purple: 'linear-gradient(45deg, #a855f7, #ec4899, #8b5cf6)',
  }

  return (
    <div
      className={cn(
        'group relative rounded-lg',
        className
      )}
      style={{
        background: gradients[gradient],
      }}
    >
      {/* Inner dark background */}
      <div
        className="absolute inset-[2px] bg-slate-900 rounded-[7px] transition-opacity duration-500 group-hover:opacity-70"
      />

      {/* Glow effect on hover */}
      <div
        className="absolute inset-0 rounded-lg opacity-0 transition-opacity duration-500 group-hover:opacity-100"
        style={{
          background: gradients[gradient],
          filter: 'blur(20px)',
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  )
}
