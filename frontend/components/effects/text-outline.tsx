import { cn } from '@/lib/utils'

interface TextOutlineProps {
  children: React.ReactNode
  strokeColor?: string
  strokeWidth?: number
  fillColor?: 'transparent' | 'gradient' | string
  className?: string
}

export function TextOutline({
  children,
  strokeColor = '#ffffff',
  strokeWidth = 2,
  fillColor = 'transparent',
  className
}: TextOutlineProps) {
  const style: React.CSSProperties = {
    WebkitTextStroke: `${strokeWidth}px ${strokeColor}`,
    textStroke: `${strokeWidth}px ${strokeColor}`,
    color: fillColor === 'transparent' ? 'transparent' : fillColor,
  }

  return (
    <span
      className={cn('font-bold', className)}
      style={style}
    >
      {children}
    </span>
  )
}
