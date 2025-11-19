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
  const style = {
    WebkitTextStroke: `${strokeWidth}px ${strokeColor}`,
    color: fillColor === 'transparent' ? 'transparent' : fillColor,
  } as React.CSSProperties

  return (
    <span
      className={cn('font-bold', className)}
      style={style}
    >
      {children}
    </span>
  )
}
