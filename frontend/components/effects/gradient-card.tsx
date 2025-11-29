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
  // Theme-aware simple backgrounds (toned down from corporate synthwave)
  const gradients = {
    dusk: {
      subtle: 'bg-card/50 backdrop-blur-sm',
      medium: 'bg-card/70 backdrop-blur-sm',
      vibrant: 'bg-card backdrop-blur-sm',
    },
    'cotton-candy': {
      subtle: 'bg-card/50 backdrop-blur-sm',
      medium: 'bg-card/70 backdrop-blur-sm',
      vibrant: 'bg-card backdrop-blur-sm',
    },
    watermelon: {
      subtle: 'bg-card/50 backdrop-blur-sm',
      medium: 'bg-card/70 backdrop-blur-sm',
      vibrant: 'bg-card backdrop-blur-sm',
    },
    ocean: {
      subtle: 'bg-card/50 backdrop-blur-sm',
      medium: 'bg-card/70 backdrop-blur-sm',
      vibrant: 'bg-card backdrop-blur-sm',
    },
    shadow: {
      subtle: 'bg-card/50 backdrop-blur-sm',
      medium: 'bg-card/70 backdrop-blur-sm',
      vibrant: 'bg-card backdrop-blur-sm',
    },
  }

  // Simple theme-aware borders
  const borders = {
    dusk: 'border-border',
    'cotton-candy': 'border-border',
    watermelon: 'border-border',
    ocean: 'border-border',
    shadow: 'border-border',
  }

  // Simple accessible shadows
  const shadows = {
    dusk: {
      subtle: 'shadow-sm',
      medium: 'shadow-md',
      vibrant: 'shadow-lg',
    },
    'cotton-candy': {
      subtle: 'shadow-sm',
      medium: 'shadow-md',
      vibrant: 'shadow-lg',
    },
    watermelon: {
      subtle: 'shadow-sm',
      medium: 'shadow-md',
      vibrant: 'shadow-lg',
    },
    ocean: {
      subtle: 'shadow-sm',
      medium: 'shadow-md',
      vibrant: 'shadow-lg',
    },
    shadow: {
      subtle: 'shadow-sm',
      medium: 'shadow-md',
      vibrant: 'shadow-lg',
    },
  }

  return (
    <div
      className={cn(
        'rounded-lg border transition-all duration-200',
        gradients[variant][intensity],
        borders[variant],
        shadows[variant][intensity],
        className
      )}
    >
      {children}
    </div>
  )
}
