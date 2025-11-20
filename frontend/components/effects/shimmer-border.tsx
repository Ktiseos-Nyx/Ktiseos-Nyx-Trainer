import { cn } from '@/lib/utils'

interface ShimmerBorderProps {
  children: React.ReactNode
  color?: 'purple' | 'blue' | 'rainbow'
  speed?: 'slow' | 'medium' | 'fast'
  borderWidth?: number
  className?: string
}

export function ShimmerBorder({
  children,
  color = 'purple',
  speed = 'medium',
  borderWidth = 2,
  className
}: ShimmerBorderProps) {
  const gradients = {
    purple: 'linear-gradient(90deg, transparent, #a855f7, transparent)',
    blue: 'linear-gradient(90deg, transparent, #3b82f6, transparent)',
    rainbow: 'linear-gradient(90deg, #ef4444, #f59e0b, #10b981, #3b82f6, #8b5cf6, #ef4444)',
  }

  const speeds = {
    slow: '4s',
    medium: '2s',
    fast: '1s',
  }

  return (
    <div className={cn('relative rounded-lg overflow-hidden', className)}>
      {/* Animated border */}
      <div
        className="absolute inset-0"
        style={{
          padding: `${borderWidth}px`,
          background: gradients[color],
          backgroundSize: '200% 100%',
          animation: `shimmer ${speeds[speed]} linear infinite`,
          WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
        }}
      />

      {/* Content */}
      <div className="relative z-10 bg-slate-900 rounded-lg">
        {children}
      </div>
    </div>
  )
}
