/**
 * GradientBorderButton - Animated gradient border with glow effect
 *
 * Inspired by:
 * - https://uiverse.io/nima-mollazadeh/terrible-panda-97
 */

'use client'

import { cn } from '@/lib/utils'

interface GradientBorderButtonProps {
  children: React.ReactNode
  gradient?: 'neon' | 'sunset' | 'ocean' | 'fire'
  onClick?: () => void
  className?: string
  disabled?: boolean
}

export function GradientBorderButton({
  children,
  gradient = 'neon',
  onClick,
  className,
  disabled = false
}: GradientBorderButtonProps) {
  const gradients = {
    neon: 'linear-gradient(45deg, #0ce39a, #69007f, #fc0987)',
    sunset: 'linear-gradient(45deg, #f59e0b, #ef4444, #ec4899)',
    ocean: 'linear-gradient(45deg, #06b6d4, #3b82f6, #8b5cf6)',
    fire: 'linear-gradient(45deg, #f97316, #dc2626, #7c2d12)',
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'group relative px-6 py-3.5 rounded-lg text-white font-semibold text-lg cursor-pointer',
        'transition-all duration-300',
        disabled && 'opacity-50 cursor-not-allowed',
        className
      )}
      style={{
        background: gradients[gradient],
      }}
    >
      {/* Inner dark background */}
      <span
        className={cn(
          'absolute inset-[1px] bg-slate-900 rounded-[9px] transition-opacity duration-500',
          !disabled && 'group-hover:opacity-70'
        )}
      />

      {/* Glow effect on hover */}
      <span
        className={cn(
          'absolute inset-0 rounded-lg opacity-0 transition-opacity duration-500 group-hover:opacity-100',
          disabled && 'group-hover:opacity-0'
        )}
        style={{
          background: gradients[gradient],
          filter: 'blur(20px)',
        }}
      />

      {/* Text content */}
      <span className="relative z-10">{children}</span>
    </button>
  )
}
