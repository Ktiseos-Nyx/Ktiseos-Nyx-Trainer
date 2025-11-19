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
    slow: '8s',
    medium: '4s',
    fast: '2s',
  }

  return (
    <div className={cn('relative overflow-hidden rounded-lg', className)}>
      {/* Animated holographic background layer */}
      <div
        className="absolute inset-0 opacity-50"
        style={{
          background: 'linear-gradient(45deg, transparent 30%, rgba(168,85,247,0.3), transparent 70%)',
          backgroundSize: '200% 200%',
          animation: `holographic ${speeds[speed]} linear infinite`,
        }}
      />

      {/* Content layer */}
      <div className="relative z-10 bg-slate-900/80 backdrop-blur-sm rounded-lg">
        {children}
      </div>

      <style jsx>{`
        @keyframes holographic {
          0% { background-position: 0% 0%; }
          25% { background-position: 100% 100%; }
          50% { background-position: 200% 0%; }
          75% { background-position: 100% -100%; }
          100% { background-position: 0% 0%; }
        }
      `}</style>
    </div>
  )
}
