import { cn } from '@/lib/utils'

interface GradientCardProps {
  children: React.ReactNode
  variant?: 'dusk' | 'cotton-candy' | 'watermelon' | 'ocean' | 'shadow'
  intensity?: 'subtle' | 'medium' | 'vibrant'
  className?: string
}

export function GradientCard({
  children,
  variant = 'dusk',
  intensity = 'subtle',
  className
}: GradientCardProps) {
  // Subtle gradient backgrounds
  const gradients = {
    dusk: {
      subtle: 'bg-gradient-to-br from-purple-950/30 via-blue-950/30 to-slate-950/30',
      medium: 'bg-gradient-to-br from-purple-900/40 via-blue-900/40 to-slate-900/40',
      vibrant: 'bg-gradient-to-br from-purple-800/50 via-blue-800/50 to-slate-800/50',
    },
    'cotton-candy': {
      subtle: 'bg-gradient-to-br from-pink-200/15 via-blue-200/15 via-purple-200/15 to-slate-950/30',
      medium: 'bg-gradient-to-br from-pink-300/20 via-blue-300/20 via-purple-300/20 to-slate-900/35',
      vibrant: 'bg-gradient-to-br from-pink-400/25 via-blue-400/25 via-purple-400/25 to-slate-800/40',
    },
    watermelon: {
      subtle: 'bg-gradient-to-br from-pink-950/25 via-green-950/20 to-slate-950/30',
      medium: 'bg-gradient-to-br from-pink-900/35 via-green-900/25 to-slate-900/40',
      vibrant: 'bg-gradient-to-br from-pink-800/40 via-green-800/30 to-slate-800/45',
    },
    ocean: {
      subtle: 'bg-gradient-to-br from-blue-950/30 via-teal-950/25 via-cyan-950/25 to-slate-950/30',
      medium: 'bg-gradient-to-br from-blue-900/40 via-teal-900/35 via-cyan-900/35 to-slate-900/40',
      vibrant: 'bg-gradient-to-br from-blue-800/45 via-teal-800/40 via-cyan-800/40 to-slate-800/45',
    },
    shadow: {
      subtle: 'bg-slate-900/50',
      medium: 'bg-slate-900/70',
      vibrant: 'bg-slate-800/85',
    },
  }

  // Subtle border accents
  const borders = {
    dusk: 'border-purple-500/20',
    'cotton-candy': 'border-pink-300/25',
    watermelon: 'border-pink-500/20',
    ocean: 'border-cyan-500/20',
    shadow: 'border-slate-700/50',
  }

  // Gentle shadow enhancements
  const shadows = {
    dusk: {
      subtle: 'shadow-lg shadow-purple-500/5',
      medium: 'shadow-xl shadow-purple-500/10',
      vibrant: 'shadow-2xl shadow-purple-500/15',
    },
    'cotton-candy': {
      subtle: 'shadow-lg shadow-pink-300/5',
      medium: 'shadow-xl shadow-pink-300/10',
      vibrant: 'shadow-2xl shadow-pink-300/15',
    },
    watermelon: {
      subtle: 'shadow-lg shadow-pink-500/5',
      medium: 'shadow-xl shadow-pink-500/10',
      vibrant: 'shadow-2xl shadow-pink-500/15',
    },
    ocean: {
      subtle: 'shadow-lg shadow-cyan-500/5',
      medium: 'shadow-xl shadow-cyan-500/10',
      vibrant: 'shadow-2xl shadow-cyan-500/15',
    },
    shadow: {
      subtle: 'shadow-2xl shadow-black/30',
      medium: 'shadow-2xl shadow-black/50',
      vibrant: 'shadow-2xl shadow-black/70',
    },
  }

  return (
    <div
      className={cn(
        'rounded-lg border backdrop-blur-sm transition-all duration-300',
        gradients[variant][intensity],
        borders[variant],
        shadows[variant][intensity],
        'hover:scale-[1.02] hover:shadow-2xl',
        className
      )}
    >
      {children}
    </div>
  )
}
