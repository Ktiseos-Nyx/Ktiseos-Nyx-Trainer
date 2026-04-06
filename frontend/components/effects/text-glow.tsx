import { cn } from '@/lib/utils'

interface TextGlowProps {
  children: React.ReactNode
  color?: 'purple' | 'blue' | 'green' | 'pink' | 'orange'
  intensity?: 'subtle' | 'medium' | 'intense' | 'extreme'
  className?: string
}

const glowColors = {
  purple: 'rgba(168,85,247',
  blue: 'rgba(59,130,246',
  green: 'rgba(34,197,94',
  pink: 'rgba(236,72,153',
  orange: 'rgba(251,146,60',
}

export function TextGlow({
  children,
  color = 'purple',
  intensity = 'medium',
  className
}: TextGlowProps) {
  const baseColor = glowColors[color]

  const shadows = {
    subtle: `0 0 10px ${baseColor},0.6)`,
    medium: `0 0 10px ${baseColor},0.8), 0 0 20px ${baseColor},0.5)`,
    intense: `0 0 10px ${baseColor},1), 0 0 20px ${baseColor},0.8), 0 0 30px ${baseColor},0.5)`,
    extreme: `0 0 10px ${baseColor},1), 0 0 20px ${baseColor},1), 0 0 30px ${baseColor},0.8), 0 0 40px ${baseColor},0.5)`,
  }

  return (
    <span
      className={cn(className)}
      style={{ textShadow: shadows[intensity] }}
    >
      {children}
    </span>
  )
}
