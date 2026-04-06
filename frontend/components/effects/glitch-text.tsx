import { cn } from '@/lib/utils'

interface GlitchTextProps {
  children: React.ReactNode
  intensity?: 'subtle' | 'medium' | 'wild'
  className?: string
}

export function GlitchText({
  children,
  intensity = 'medium',
  className
}: GlitchTextProps) {
  const offsets = {
    subtle: { x: 2, y: 2 },
    medium: { x: 4, y: 4 },
    wild: { x: 8, y: 8 },
  }

  const offset = offsets[intensity]

  return (
    <span className={cn('relative inline-block', className)}>
      {/* Main text */}
      <span className="relative z-10">{children}</span>

      {/* Cyan glitch layer (top half) */}
      <span
        className="absolute top-0 left-0 text-cyan-400 opacity-70"
        style={{
          clipPath: 'polygon(0 0, 100% 0, 100% 45%, 0 45%)',
          animation: `glitch-1-${intensity} 3s infinite`,
        }}
        aria-hidden="true"
      >
        {children}
      </span>

      {/* Pink glitch layer (bottom half) */}
      <span
        className="absolute top-0 left-0 text-pink-400 opacity-70"
        style={{
          clipPath: 'polygon(0 55%, 100% 55%, 100% 100%, 0 100%)',
          animation: `glitch-2-${intensity} 3s infinite reverse`,
        }}
        aria-hidden="true"
      >
        {children}
      </span>

      {/* Dynamic keyframes */}
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes glitch-1-${intensity} {
            0%, 100% { transform: translate(0); }
            20% { transform: translate(-${offset.x}px, ${offset.y}px); }
            40% { transform: translate(-${offset.x}px, -${offset.y}px); }
            60% { transform: translate(${offset.x}px, ${offset.y}px); }
            80% { transform: translate(${offset.x}px, -${offset.y}px); }
          }

          @keyframes glitch-2-${intensity} {
            0%, 100% { transform: translate(0); }
            20% { transform: translate(${offset.x}px, -${offset.y}px); }
            40% { transform: translate(${offset.x}px, ${offset.y}px); }
            60% { transform: translate(-${offset.x}px, -${offset.y}px); }
            80% { transform: translate(-${offset.x}px, ${offset.y}px); }
          }
        `
      }} />
    </span>
  )
}
