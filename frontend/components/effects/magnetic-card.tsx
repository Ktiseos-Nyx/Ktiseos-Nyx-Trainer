'use client'

import { cn } from '@/lib/utils'
import { useRef, useState } from 'react'

interface MagneticCardProps {
  children: React.ReactNode
  strength?: 'subtle' | 'medium' | 'strong'
  className?: string
}

export function MagneticCard({
  children,
  strength = 'medium',
  className
}: MagneticCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isHovering, setIsHovering] = useState(false)

  const strengths = {
    subtle: 0.15,
    medium: 0.25,
    strong: 0.4,
  }

  const multiplier = strengths[strength]

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return

    const rect = cardRef.current.getBoundingClientRect()
    const centerX = rect.left + rect.width / 2
    const centerY = rect.top + rect.height / 2

    const deltaX = (e.clientX - centerX) * multiplier
    const deltaY = (e.clientY - centerY) * multiplier

    setPosition({ x: deltaX, y: deltaY })
  }

  const handleMouseLeave = () => {
    setIsHovering(false)
    setPosition({ x: 0, y: 0 })
  }

  return (
    <div
      ref={cardRef}
      className={cn('transition-transform duration-300 ease-out will-change-transform', className)}
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
      }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={handleMouseLeave}
    >
      {children}
    </div>
  )
}
