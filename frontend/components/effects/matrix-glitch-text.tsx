'use client'

import { cn } from '@/lib/utils'
import { useEffect, useState } from 'react'

interface MatrixGlitchTextProps {
  children: string
  glitchSpeed?: 'slow' | 'medium' | 'fast'
  className?: string
}

// Character sets for glitching
const GLITCH_CHARS = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ!@#$%^&*АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ0123456789'

export function MatrixGlitchText({
  children,
  glitchSpeed = 'medium',
  className
}: MatrixGlitchTextProps) {
  const [displayText, setDisplayText] = useState(children)

  const speeds = {
    slow: 150,
    medium: 80,
    fast: 40,
  }

  const glitchInterval = speeds[glitchSpeed]

  useEffect(() => {
    // Constantly cycle random characters
    const interval = setInterval(() => {
      const glitched = children
        .split('')
        .map((char) => {
          // Keep spaces
          if (char === ' ') return char

          // Each character has independent random chance to show glitch char
          if (Math.random() > 0.6) {
            return GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)]
          }

          // Otherwise show original character
          return char
        })
        .join('')

      setDisplayText(glitched)
    }, glitchInterval)

    return () => clearInterval(interval)
  }, [children, glitchInterval])

  return (
    <span
      className={cn(
        'inline-block font-mono',
        className
      )}
      style={{
        textShadow: '0 0 10px rgba(34, 197, 94, 0.5), 0 0 20px rgba(34, 197, 94, 0.3)',
        color: '#22c55e', // Green-400
      }}
    >
      {displayText}
    </span>
  )
}
