import { useEffect, useState } from "react"
import { animate, motion, useMotionValue, useTransform } from "motion/react"

interface UsageCardProps {
  title: string
  percentage: number
  pillCount?: number
  accentColor?: string
  icon?: React.ReactNode
}

function useAnimatedCounter(target: number, duration = 1.2) {
  const count = useMotionValue(0)
  const rounded = useTransform(count, (v) => Math.round(v))
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    const controls = animate(count, target, {
      duration,
      ease: [0.16, 1, 0.3, 1],
    })
    const unsub = rounded.on("change", setDisplay)
    return () => {
      controls.stop()
      unsub()
    }
  }, [target])

  return display
}

function Pill({
  active,
  index,
  accent,
}: {
  active: boolean
  index: number
  accent: string
}) {
  return (
    <motion.div
      initial={{ scaleY: 0.3, opacity: 0 }}
      animate={{ scaleY: 1, opacity: 1 }}
      transition={{
        delay: index * 0.04,
        duration: 0.45,
        ease: [0.16, 1, 0.3, 1],
      }}
      style={{ originY: 1, flex: 1 }}
      className="relative h-full overflow-hidden rounded-xs"
    >
      {/* Track */}
      <div className="absolute inset-0 rounded-xs bg-black/5 dark:bg-white/5" />

      {/* Fill */}
      <motion.div
        className="absolute inset-0 rounded-xs"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: active ? 1 : 0 }}
        transition={{
          delay: index * 0.055 + 0.25,
          duration: 0.4,
          ease: [0.16, 1, 0.3, 1],
        }}
        style={{
          originX: 0,
          background: active
            ? `linear-gradient(90deg, ${accent}bb, ${accent})`
            : "transparent",
          boxShadow: active
            ? `0 0 6px ${accent}99, 0 0 12px ${accent}44`
            : "none",
        }}
      />

      {/* Shimmer */}
      {active && (
        <motion.div
          className="absolute inset-0 rounded-xs"
          style={{
            background: `linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.16) 50%, transparent 100%)`,
            backgroundSize: "200% 100%",
          }}
          animate={{ backgroundPositionX: ["0%", "200%"] }}
          transition={{
            repeat: Infinity,
            duration: 2,
            ease: "linear",
            delay: index * 0.08,
          }}
        />
      )}
    </motion.div>
  )
}

export default function UsageCard({
  title,
  percentage,
  pillCount = 10,
  accentColor = "#00f5ff",
  icon,
}: UsageCardProps) {
  const display = useAnimatedCounter(percentage)
  const activePills = Math.round((percentage / 100) * pillCount)
  const accent = percentage >= 75 ? "#ff2d55" : accentColor

  return (
    <div
      className="relative w-64 select-none"
      style={{ fontFamily: "'IBM Plex Mono', 'Courier New', monospace" }}
    >
      {/* Outer glow */}
      <div
        className="pointer-events-none absolute -inset-px rounded-2xl opacity-20 blur-sm dark:opacity-30"
        style={{
          background: `linear-gradient(135deg, ${accent}66, transparent 60%)`,
        }}
      />

      {/* Card */}
      <div className="relative overflow-hidden rounded-[8px] border border-zinc-200 bg-white dark:border-white/7 dark:bg-zinc-950">
        <div className="flex flex-col gap-4 p-4">
          {/* Top row */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-black dark:text-white/40">
              {icon}
              <span className="text-[11px] tracking-widest uppercase">
                {title}
              </span>
            </div>

            <div className="flex items-baseline gap-[3px]">
              <motion.span
                className="text-xl leading-none font-semibold"
                style={{ color: accent, textShadow: `0 0 12px ${accent}88` }}
              >
                {display}
              </motion.span>
              <span className="text-[10px] text-black dark:text-white/25">
                %
              </span>
            </div>
          </div>

          {/* Pills */}
          <div className="flex gap-[4px]" style={{ height: 32 }}>
            {Array.from({ length: pillCount }).map((_, i) => (
              <Pill
                key={i}
                index={i}
                active={i < activePills}
                accent={accent}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
