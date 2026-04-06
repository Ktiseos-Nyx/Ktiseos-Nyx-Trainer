import { cn } from '@/lib/utils'

interface NeonTextProps {
  children: React.ReactNode
  color?: 'purple' | 'blue' | 'pink' | 'green' | 'cyan'
  flicker?: boolean
  className?: string
}

const neonColors = {
  purple: {
    text: 'text-purple-400',
    glow: 'rgba(168,85,247',
  },
  blue: {
    text: 'text-blue-400',
    glow: 'rgba(59,130,246',
  },
  pink: {
    text: 'text-pink-400',
    glow: 'rgba(236,72,153',
  },
  green: {
    text: 'text-green-400',
    glow: 'rgba(34,197,94',
  },
  cyan: {
    text: 'text-cyan-400',
    glow: 'rgba(34,211,238',
  },
}

export function NeonText({
  children,
  color = 'purple',
  flicker = false,
  className
}: NeonTextProps) {
  const { text, glow } = neonColors[color]

  const style: React.CSSProperties = {
    textShadow: `
      0 0 7px ${glow},1),
      0 0 10px ${glow},1),
      0 0 21px ${glow},1),
      0 0 42px ${glow},0.8),
      0 0 82px ${glow},0.6),
      0 0 92px ${glow},0.4)
    `,
  }

  return (
    <span
      className={cn(
        text,
        'font-bold',
        flicker && 'animate-pulse',
        className
      )}
      style={style}
    >
      {children}
    </span>
  )
}
