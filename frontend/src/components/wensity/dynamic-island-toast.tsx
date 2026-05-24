"use client";

/**
 * DynamicIslandToast
 * ──────────────────
 * A faithful re-creation of Apple's iOS 17+ Dynamic Island.
 *
 * The real DI is not a notification toast — it's a *shape-shifting hardware
 * cutout* at the top of the screen that morphs to host live activities. We
 * model that here as a state machine over three layout modes:
 *
 *   • compact   — the resting stadium pill (≈126×37) with optional
 *                 leading + trailing "ear" accessories on either side of
 *                 the camera area.
 *   • split     — TWO pills with a magnetic gap between them, used when
 *                 there are two active live activities (e.g. timer + music).
 *                 The merge / separate uses a gooey SVG blob filter so
 *                 they smoosh together exactly like iOS.
 *   • expanded  — a tall rounded rectangle that drops down below the
 *                 hardware pill on long-press / tap. All trailing UI from
 *                 the compact view glides into place via shared layoutIds.
 *
 * Physics: every shape morph uses one tuned Apple-grade spring
 * (stiffness 700, damping 38, mass 1.05). Content cross-fades on a faster
 * spring with a 6px blur enter/exit so there's never a hard pop.
 *
 * Variants (all backward-compatible with the previous registry preview):
 *   compact     · text · upload · audio · call · ringer
 *   timer       — countdown / chronograph live activity
 *   navigation  — turn-by-turn maps banner with next maneuver
 *   airpods     — AirPods connected / battery
 *   charging    — battery % during plug-in
 *   faceid      — Face ID scanning indicator
 */

import * as React from "react";
import {
  AnimatePresence,
  LayoutGroup,
  MotionConfig,
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
  type Transition,
} from "framer-motion";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════════════════════════════════
   TYPES
   ═══════════════════════════════════════════════════════════════════════ */

export type DynamicIslandVariant =
  | "compact"
  | "text"
  | "upload"
  | "audio"
  | "call"
  | "ringer"
  | "timer"
  | "navigation"
  | "airpods"
  | "charging"
  | "faceid";

interface BaseModel {
  id: string;
  /** Auto-collapse the expanded view after this many ms of inactivity. */
  duration?: number;
}

export type DynamicIslandToast =
  | (BaseModel & { variant: "compact"; title?: string; message?: string })
  | (BaseModel & { variant: "text"; title: string; message: string; sender?: string })
  | (BaseModel & { variant: "upload"; title: string; progress: number /* 0..1 */ })
  | (BaseModel & { variant: "audio"; title: string; message?: string })
  | (BaseModel & { variant: "call"; caller: string; message?: string })
  | (BaseModel & { variant: "ringer"; muted?: boolean })
  | (BaseModel & { variant: "timer"; label?: string; remainingMs: number; totalMs: number })
  | (BaseModel & { variant: "navigation"; instruction: string; distance: string; street?: string })
  | (BaseModel & { variant: "airpods"; model?: string; battery: number /* 0..1 */ })
  | (BaseModel & { variant: "charging"; battery: number /* 0..1 */; eta?: string })
  | (BaseModel & { variant: "faceid"; status?: "scanning" | "success" | "fail" });

export interface DynamicIslandToastProps {
  /** The active toast. Pass `null` to dismiss. */
  toast: DynamicIslandToast | null;
  /**
   * Optional second activity. When present, the island enters split-mode:
   * two pills with a magnetic gooey gap, just like iOS when two live
   * activities are running concurrently.
   */
  secondary?: DynamicIslandToast | null;
  /** Render expanded by default (useful for hero previews). */
  defaultExpanded?: boolean;
  /** Notified whenever the user expands or collapses the island. */
  onExpandedChange?: (expanded: boolean) => void;
  /** Fired when the island auto-dismisses or the user closes it. */
  onDismiss?: () => void;
  /**
   * If true, position absolutely inside the nearest positioned ancestor
   * (top-center). If false, position fixed to the viewport like real iOS.
   */
  inline?: boolean;
  className?: string;
}

/* ═══════════════════════════════════════════════════════════════════════
   PHYSICS — tuned to match the real iOS Dynamic Island morph
   ═══════════════════════════════════════════════════════════════════════ */

/** Shell morph: stiff, slightly overshooting — like a soft rubber pill. */
const ISLAND_SPRING: Transition = {
  type: "spring",
  stiffness: 700,
  damping: 38,
  mass: 1.05,
};

/** Content swaps: snappier, with a subtle settle. */
const CONTENT_SPRING: Transition = {
  type: "spring",
  stiffness: 540,
  damping: 36,
  mass: 0.65,
};

const DynamicIslandReducedMotionContext = React.createContext(false);

function useDynamicIslandReducedMotion() {
  return React.useContext(DynamicIslandReducedMotionContext);
}

/* ═══════════════════════════════════════════════════════════════════════
   ROOT
   ═══════════════════════════════════════════════════════════════════════ */

export function DynamicIslandToast({
  toast,
  secondary = null,
  defaultExpanded = false,
  onExpandedChange,
  onDismiss,
  inline = false,
  className,
}: DynamicIslandToastProps) {
  const [expanded, setExpandedState] = React.useState(defaultExpanded);
  const reduceMotion = useReducedMotion() ?? false;

  const setExpanded = React.useCallback(
    (next: boolean) => {
      setExpandedState(next);
      onExpandedChange?.(next);
    },
    [onExpandedChange]
  );

  // Reset expansion when the primary toast identity changes.
  const lastIdRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!toast) return;
    if (lastIdRef.current !== toast.id) {
      lastIdRef.current = toast.id;
      if (!defaultExpanded) setExpandedState(false);
    }
  }, [toast, defaultExpanded]);

  // Auto-dismiss after `duration` ms of inactivity (compact only).
  React.useEffect(() => {
    if (!toast?.duration || expanded) return;
    const id = window.setTimeout(() => onDismiss?.(), toast.duration);
    return () => window.clearTimeout(id);
  }, [toast, expanded, onDismiss]);

  // Long-press to expand (matches real DI gesture).
  const longPressRef = React.useRef<number | null>(null);
  const startLongPress = () => {
    if (longPressRef.current) window.clearTimeout(longPressRef.current);
    longPressRef.current = window.setTimeout(() => setExpanded(true), 380);
  };
  const cancelLongPress = () => {
    if (longPressRef.current) {
      window.clearTimeout(longPressRef.current);
      longPressRef.current = null;
    }
  };

  const isSplit = !!secondary && !expanded;

  return (
    <MotionConfig reducedMotion="user">
      <DynamicIslandReducedMotionContext.Provider value={reduceMotion}>
        {/* Gooey filter — the liquid-merge effect between split pills */}
        <GooeyDefs />

        <div
          aria-live="polite"
          className={cn(
            "pointer-events-none z-50 flex justify-center",
            inline ? "absolute inset-x-0 top-3" : "fixed inset-x-0 top-2",
            className
          )}
        >
          <LayoutGroup id="dynamic-island">
            <div
              className="pointer-events-none relative flex items-start justify-center gap-2"
              style={{ filter: isSplit ? "url(#di-goo)" : undefined }}
            >
              <AnimatePresence mode="popLayout" initial={false}>
                {toast ? (
                  <IslandShell
                    key="primary"
                    toast={toast}
                    expanded={expanded}
                    split={isSplit}
                    onTap={() => setExpanded(!expanded)}
                    onPressStart={startLongPress}
                    onPressEnd={cancelLongPress}
                  />
                ) : null}

                {isSplit && secondary ? (
                  <SecondaryPill key={`secondary-${secondary.id}`} toast={secondary} />
                ) : null}
              </AnimatePresence>
            </div>
          </LayoutGroup>
        </div>
      </DynamicIslandReducedMotionContext.Provider>
    </MotionConfig>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   PRIMARY SHELL — the morphing hardware pill
   ═══════════════════════════════════════════════════════════════════════ */

function IslandShell({
  toast,
  expanded,
  split,
  onTap,
  onPressStart,
  onPressEnd,
}: {
  toast: DynamicIslandToast;
  expanded: boolean;
  split: boolean;
  onTap: () => void;
  onPressStart: () => void;
  onPressEnd: () => void;
}) {
  const contentKey = `${toast.id}-${toast.variant}-${expanded ? "x" : split ? "s" : "c"}`;
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.button
      type="button"
      layout
      layoutId="island-shell"
      onClick={onTap}
      onPointerDown={onPressStart}
      onPointerUp={onPressEnd}
      onPointerCancel={onPressEnd}
      onPointerLeave={onPressEnd}
      aria-expanded={expanded}
      aria-label={`Dynamic Island: ${toast.variant}${expanded ? ", expanded" : ""}`}
      transition={reduceMotion ? { duration: 0.01 } : ISLAND_SPRING}
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -22, scale: 0.55 }}
      animate={reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
      exit={
        reduceMotion
          ? { opacity: 0, transition: { duration: 0.01 } }
          : { opacity: 0, y: -16, scale: 0.65, transition: { duration: 0.18 } }
      }
      whileTap={reduceMotion ? undefined : { scale: 0.97 }}
      style={{
        // Real DI is a *true* matte black, with a faint top highlight that
        // hints at the OLED specular reflection along the cutout edge.
        background:
          "radial-gradient(140% 220% at 50% -10%, #1a1a1c 0%, #060607 55%, #000 100%)",
        boxShadow: expanded
          ? "0 1px 0 rgba(255,255,255,0.05) inset, 0 30px 80px -20px rgba(0,0,0,0.6), 0 8px 24px -10px rgba(0,0,0,0.5)"
          : "0 1px 0 rgba(255,255,255,0.06) inset, 0 14px 30px -14px rgba(0,0,0,0.55), 0 4px 10px -6px rgba(0,0,0,0.45)",
      }}
      className={cn(
        "pointer-events-auto relative isolate overflow-hidden",
        "select-none text-left text-white outline-none",
        "ring-1 ring-white/[0.05] focus-visible:ring-2 focus-visible:ring-white/30",
        // Border-radius animates via `layout`; base it on a stadium.
        "rounded-[28px]"
      )}
    >
      {/* Faux camera lens — only a hint, sits dead-center of the pill at rest */}
      {!expanded && toast.variant === "compact" ? <CameraLens /> : null}

      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div
          key={contentKey}
          layout
          initial={reduceMotion ? { opacity: 0 } : { opacity: 0, filter: "blur(6px)" }}
          animate={reduceMotion ? { opacity: 1 } : { opacity: 1, filter: "blur(0px)" }}
          exit={
            reduceMotion
              ? { opacity: 0, transition: { duration: 0.01 } }
              : { opacity: 0, filter: "blur(6px)", transition: { duration: 0.14 } }
          }
          transition={reduceMotion ? { duration: 0.01 } : CONTENT_SPRING}
          className="relative"
        >
          <IslandBody toast={toast} expanded={expanded} />
        </motion.div>
      </AnimatePresence>
    </motion.button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SECONDARY (split) PILL
   ═══════════════════════════════════════════════════════════════════════ */

function SecondaryPill({ toast }: { toast: DynamicIslandToast }) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.div
      layout
      layoutId={`island-secondary-${toast.id}`}
      transition={reduceMotion ? { duration: 0.01 } : ISLAND_SPRING}
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.4, x: -28 }}
      animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, x: 0 }}
      exit={
        reduceMotion
          ? { opacity: 0, transition: { duration: 0.01 } }
          : { opacity: 0, scale: 0.4, x: -22, transition: { duration: 0.18 } }
      }
      className="pointer-events-auto grid h-[37px] min-w-[37px] place-items-center rounded-full px-2.5 text-white"
      style={{
        background:
          "radial-gradient(140% 220% at 50% -10%, #1a1a1c 0%, #060607 55%, #000 100%)",
        boxShadow:
          "0 1px 0 rgba(255,255,255,0.06) inset, 0 14px 30px -14px rgba(0,0,0,0.55)",
      }}
    >
      <SecondaryBody toast={toast} />
    </motion.div>
  );
}

function SecondaryBody({ toast }: { toast: DynamicIslandToast }) {
  switch (toast.variant) {
    case "timer": {
      const pct = clamp01(1 - toast.remainingMs / Math.max(1, toast.totalMs));
      return (
        <div className="flex items-center gap-1.5">
          <ProgressRing value={pct} size={20} stroke={2.5} />
          <span className="font-display text-[12px] font-semibold tabular-nums">
            {formatTimer(toast.remainingMs)}
          </span>
        </div>
      );
    }
    case "audio":
      return <Waveform color="#a3e635" />;
    case "call":
      return <IconPhone />;
    case "ringer":
      return (toast.muted ?? true) ? <IconBellOff /> : <IconBell />;
    case "navigation":
      return (
        <div className="flex items-center gap-1.5">
          <IconArrowTurn />
          <span className="text-[12px] font-semibold">{toast.distance}</span>
        </div>
      );
    case "charging":
      return (
        <div className="flex items-center gap-1">
          <IconBolt />
          <span className="text-[12px] font-semibold">{Math.round(toast.battery * 100)}%</span>
        </div>
      );
    case "airpods":
      return <IconAirpods />;
    case "faceid":
      return <IconFaceId />;
    case "upload":
      return <ProgressRing value={toast.progress} size={20} stroke={2.5} />;
    default:
      return <Pulse />;
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   STATE MACHINE — primary content per variant × per (compact | expanded)
   ═══════════════════════════════════════════════════════════════════════ */

function IslandBody({ toast, expanded }: { toast: DynamicIslandToast; expanded: boolean }) {
  switch (toast.variant) {
    case "compact":     return <CompactState     toast={toast} expanded={expanded} />;
    case "text":        return <TextState        toast={toast} expanded={expanded} />;
    case "upload":      return <UploadState      toast={toast} expanded={expanded} />;
    case "audio":       return <AudioState       toast={toast} expanded={expanded} />;
    case "call":        return <CallState        toast={toast} expanded={expanded} />;
    case "ringer":      return <RingerState      toast={toast} expanded={expanded} />;
    case "timer":       return <TimerState       toast={toast} expanded={expanded} />;
    case "navigation":  return <NavigationState  toast={toast} expanded={expanded} />;
    case "airpods":     return <AirpodsState     toast={toast} expanded={expanded} />;
    case "charging":    return <ChargingState    toast={toast} expanded={expanded} />;
    case "faceid":      return <FaceIdState      toast={toast} expanded={expanded} />;
  }
}

/* ────────────────────────────── compact ─────────────────────────────── */

function CompactState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "compact" }>;
  expanded: boolean;
}) {
  if (!expanded) {
    // Resting state — empty pill that frames the camera. Width is tuned to
    // the real DI footprint (~126px wide, 37px tall).
    return <div className="h-[37px] w-[126px]" />;
  }
  return (
    <div className="px-6 py-4 w-[300px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-9 w-9 place-items-center rounded-full bg-white/10"
        >
          <Pulse />
        </motion.span>
        <div className="min-w-0">
          <Title>{toast.title ?? "Live Activity"}</Title>
          <Sub>{toast.message ?? "Tap to open the active session."}</Sub>
        </div>
      </Row>
    </div>
  );
}

/* ─────────────────────────────── text ───────────────────────────────── */

function TextState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "text" }>;
  expanded: boolean;
}) {
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-4 gap-2.5 max-w-[330px]">
        <motion.span
          layoutId="island-leading"
          className="grid h-7 w-7 place-items-center rounded-full bg-gradient-to-br from-blue-400 to-blue-600 text-[10px] font-bold uppercase"
        >
          {(toast.sender ?? toast.title)[0]}
        </motion.span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-semibold leading-tight text-white">
            {toast.title}
          </div>
          <div className="truncate text-[11.5px] leading-tight text-white/65">
            {toast.message}
          </div>
        </div>
        <motion.span
          layoutId="island-trailing"
          className="grid h-6 w-6 place-items-center rounded-full bg-white/10"
        >
          <IconChat />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[360px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-11 w-11 place-items-center rounded-full bg-gradient-to-br from-blue-400 to-blue-600 text-[13px] font-bold uppercase"
        >
          {(toast.sender ?? toast.title)[0]}
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.title}</Title>
          <Sub className="line-clamp-3 whitespace-normal text-[12.5px]">
            {toast.message}
          </Sub>
        </div>
        <motion.span
          layoutId="island-trailing"
          className="grid h-9 w-9 place-items-center rounded-full bg-white/10"
        >
          <IconChat />
        </motion.span>
      </Row>
    </div>
  );
}

/* ────────────────────────────── upload ──────────────────────────────── */

function UploadState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "upload" }>;
  expanded: boolean;
}) {
  const reduceMotion = useDynamicIslandReducedMotion();
  const pct = Math.round(clamp01(toast.progress) * 100);
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2.5">
        <motion.span
          layoutId="island-leading"
          className="grid h-7 w-7 place-items-center rounded-full bg-white/10"
        >
          <IconFile />
        </motion.span>
        <span className="max-w-[160px] truncate text-[12.5px] font-semibold text-white">
          {toast.title}
        </span>
        <motion.span layoutId="island-trailing" className="grid h-7 w-7 place-items-center">
          <ProgressRing value={toast.progress} size={26} />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[360px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-11 w-11 place-items-center rounded-2xl bg-white/10"
        >
          <IconFile className="h-5 w-5" />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.title}</Title>
          <Sub>Uploading · {pct}%</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-11 w-11 place-items-center">
          <ProgressRing value={toast.progress} size={36} stroke={3} />
        </motion.span>
      </Row>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
        <motion.div
          className="h-full rounded-full bg-white"
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 90, damping: 20 }}
        />
      </div>
    </div>
  );
}

/* ─────────────────────────────── audio ──────────────────────────────── */

function AudioState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "audio" }>;
  expanded: boolean;
}) {
  const reduceMotion = useDynamicIslandReducedMotion();
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2.5 max-w-[330px]">
        <motion.span
          layoutId="island-leading"
          className="grid h-7 w-7 place-items-center overflow-hidden rounded-md bg-gradient-to-br from-rose-500 via-orange-500 to-amber-400"
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-semibold leading-tight text-white">
            {toast.title}
          </div>
          {toast.message ? (
            <div className="truncate text-[11px] leading-tight text-white/60">
              {toast.message}
            </div>
          ) : null}
        </div>
        <motion.span layoutId="island-trailing" className="grid h-6 place-items-center">
          <Waveform color="#a3e635" />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[380px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="h-12 w-12 overflow-hidden rounded-lg bg-gradient-to-br from-rose-500 via-orange-500 to-amber-400"
        />
        <div className="min-w-0 flex-1">
          <Title>{toast.title}</Title>
          <Sub>{toast.message ?? "Now playing"}</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-9 place-items-center">
          <Waveform color="#a3e635" tall />
        </motion.span>
      </Row>
      <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10">
        <motion.div
          className="h-full rounded-full bg-white"
          initial={{ width: "32%" }}
          animate={reduceMotion ? { width: "32%" } : { width: ["32%", "78%", "32%"] }}
          transition={reduceMotion ? { duration: 0 } : { duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      <Row className="mt-3 justify-center gap-6 text-white/85">
        <CtrlBtn ariaLabel="Previous"><IconPrev /></CtrlBtn>
        <CtrlBtn ariaLabel="Play / pause" big><IconPause /></CtrlBtn>
        <CtrlBtn ariaLabel="Next"><IconNext /></CtrlBtn>
      </Row>
    </div>
  );
}

/* ─────────────────────────────── call ───────────────────────────────── */

function CallState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "call" }>;
  expanded: boolean;
}) {
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2.5 max-w-[330px]">
        <motion.span
          layoutId="island-leading"
          className="relative grid h-7 w-7 place-items-center rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-[10px] font-bold"
        >
          {initials(toast.caller)}
          <PulseRing />
        </motion.span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-semibold leading-tight text-white">
            {toast.caller}
          </div>
          <div className="truncate text-[11px] leading-tight text-white/60">
            {toast.message ?? "Incoming call…"}
          </div>
        </div>
        <motion.span
          layoutId="island-trailing"
          className="grid h-7 w-7 place-items-center rounded-full bg-emerald-500 text-white"
        >
          <IconPhone />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[360px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="relative grid h-12 w-12 place-items-center rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-[13px] font-bold"
        >
          {initials(toast.caller)}
          <PulseRing big />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.caller}</Title>
          <Sub>{toast.message ?? "Incoming call"}</Sub>
        </div>
      </Row>
      <Row className="mt-4 gap-3">
        <ActionBtn className="bg-rose-500" ariaLabel="Decline">
          <IconPhoneDown /> Decline
        </ActionBtn>
        <motion.span layoutId="island-trailing" className="contents">
          <ActionBtn className="bg-emerald-500" ariaLabel="Accept">
            <IconPhone /> Accept
          </ActionBtn>
        </motion.span>
      </Row>
    </div>
  );
}

/* ────────────────────────────── ringer ──────────────────────────────── */

function RingerState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "ringer" }>;
  expanded: boolean;
}) {
  const muted = toast.muted ?? true;
  if (!expanded) {
    return (
      <Row className="px-4 py-1.5 gap-2">
        <motion.span
          layoutId="island-leading"
          className={cn("grid h-7 w-7 place-items-center rounded-full", muted ? "text-amber-400" : "text-white")}
        >
          {muted ? <IconBellOff /> : <IconBell />}
        </motion.span>
        <span className="text-[12.5px] font-semibold text-white">
          {muted ? "Silent Mode On" : "Ring Mode On"}
        </span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[320px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className={cn(
            "grid h-11 w-11 place-items-center rounded-full bg-white/10",
            muted ? "text-amber-400" : "text-white"
          )}
        >
          {muted ? <IconBellOff big /> : <IconBell big />}
        </motion.span>
        <div className="min-w-0">
          <Title>{muted ? "Silent Mode On" : "Ring Mode On"}</Title>
          <Sub>{muted ? "All calls and alerts will be silenced." : "You'll hear incoming calls and alerts."}</Sub>
        </div>
      </Row>
    </div>
  );
}

/* ─────────────────────────────── timer ──────────────────────────────── */

function TimerState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "timer" }>;
  expanded: boolean;
}) {
  const pct = clamp01(1 - toast.remainingMs / Math.max(1, toast.totalMs));
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2.5">
        <motion.span
          layoutId="island-leading"
          className="grid h-7 w-7 place-items-center rounded-full bg-amber-500/15 text-amber-400"
        >
          <IconHourglass />
        </motion.span>
        <span className="font-display text-[13px] font-semibold tabular-nums text-white">
          {formatTimer(toast.remainingMs)}
        </span>
        <motion.span layoutId="island-trailing" className="grid h-7 w-7 place-items-center">
          <ProgressRing value={pct} size={26} color="#fbbf24" />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[340px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-11 w-11 place-items-center rounded-full bg-amber-500/15 text-amber-400"
        >
          <IconHourglass big />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.label ?? "Timer"}</Title>
          <Sub className="font-display tabular-nums">{formatTimer(toast.remainingMs)} remaining</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-11 w-11 place-items-center">
          <ProgressRing value={pct} size={36} stroke={3} color="#fbbf24" />
        </motion.span>
      </Row>
      <Row className="mt-4 gap-3">
        <ActionBtn className="bg-white/10" ariaLabel="Cancel">Cancel</ActionBtn>
        <ActionBtn className="bg-amber-500 text-black" ariaLabel="Pause">Pause</ActionBtn>
      </Row>
    </div>
  );
}

/* ──────────────────────────── navigation ────────────────────────────── */

function NavigationState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "navigation" }>;
  expanded: boolean;
}) {
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-4 gap-2.5 max-w-[340px]">
        <motion.span
          layoutId="island-leading"
          className="grid h-7 w-7 place-items-center rounded-md bg-emerald-500 text-white"
        >
          <IconArrowTurn />
        </motion.span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-semibold leading-tight text-white">
            {toast.distance}
          </div>
          <div className="truncate text-[11px] leading-tight text-white/65">
            {toast.instruction}
          </div>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-7 w-7 place-items-center text-white/80">
          <IconNavTriangle />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[360px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-12 w-12 place-items-center rounded-xl bg-emerald-500 text-white"
        >
          <IconArrowTurn big />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.distance} · {toast.instruction}</Title>
          <Sub>{toast.street ?? "Continue on current road"}</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-11 w-11 place-items-center text-white/80">
          <IconNavTriangle big />
        </motion.span>
      </Row>
    </div>
  );
}

/* ────────────────────────────── airpods ─────────────────────────────── */

function AirpodsState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "airpods" }>;
  expanded: boolean;
}) {
  const pct = Math.round(clamp01(toast.battery) * 100);
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2.5">
        <motion.span layoutId="island-leading" className="grid h-7 w-7 place-items-center text-white">
          <IconAirpods />
        </motion.span>
        <span className="text-[12.5px] font-semibold text-white">{pct}%</span>
        <motion.span layoutId="island-trailing" className="grid h-4 w-7 place-items-center">
          <Battery value={toast.battery} />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[340px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-12 w-12 place-items-center rounded-full bg-white/10 text-white"
        >
          <IconAirpods big />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>{toast.model ?? "AirPods Pro"}</Title>
          <Sub>Connected · {pct}% battery</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-6 w-12 place-items-center">
          <Battery value={toast.battery} big />
        </motion.span>
      </Row>
    </div>
  );
}

/* ────────────────────────────── charging ────────────────────────────── */

function ChargingState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "charging" }>;
  expanded: boolean;
}) {
  const pct = Math.round(clamp01(toast.battery) * 100);
  if (!expanded) {
    return (
      <Row className="px-3 py-1.5 pr-3 gap-2">
        <motion.span layoutId="island-leading" className="grid h-7 w-7 place-items-center text-emerald-400">
          <IconBolt />
        </motion.span>
        <span className="text-[12.5px] font-semibold text-white">{pct}%</span>
        <motion.span layoutId="island-trailing" className="grid h-4 w-7 place-items-center">
          <Battery value={toast.battery} charging />
        </motion.span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-4 w-[320px]">
      <Row className="gap-3">
        <motion.span
          layoutId="island-leading"
          className="grid h-12 w-12 place-items-center rounded-full bg-emerald-500/15 text-emerald-400"
        >
          <IconBolt big />
        </motion.span>
        <div className="min-w-0 flex-1">
          <Title>Charging · {pct}%</Title>
          <Sub>{toast.eta ?? "Fully charged in about an hour"}</Sub>
        </div>
        <motion.span layoutId="island-trailing" className="grid h-6 w-12 place-items-center">
          <Battery value={toast.battery} big charging />
        </motion.span>
      </Row>
    </div>
  );
}

/* ─────────────────────────────── faceid ─────────────────────────────── */

function FaceIdState({
  toast,
  expanded,
}: {
  toast: Extract<DynamicIslandToast, { variant: "faceid" }>;
  expanded: boolean;
}) {
  const reduceMotion = useDynamicIslandReducedMotion();
  const status = toast.status ?? "scanning";
  const tint =
    status === "success" ? "text-emerald-400" : status === "fail" ? "text-rose-400" : "text-white";
  if (!expanded) {
    return (
      <Row className="px-4 py-1.5 gap-2">
        <motion.span
          layoutId="island-leading"
          className={cn("grid h-7 w-7 place-items-center", tint)}
          animate={status === "scanning" && !reduceMotion ? { opacity: [1, 0.4, 1] } : { opacity: 1 }}
          transition={reduceMotion ? { duration: 0 } : { duration: 1.4, repeat: status === "scanning" ? Infinity : 0 }}
        >
          <IconFaceId />
        </motion.span>
        <span className={cn("text-[12.5px] font-semibold", tint)}>
          {status === "success" ? "Unlocked" : status === "fail" ? "Try Again" : "Face ID"}
        </span>
      </Row>
    );
  }
  return (
    <div className="px-5 py-5 w-[300px] text-center">
      <motion.div
        layoutId="island-leading"
        className={cn("mx-auto grid h-16 w-16 place-items-center", tint)}
        animate={status === "scanning" && !reduceMotion ? { scale: [1, 1.06, 1] } : { scale: 1 }}
        transition={
          reduceMotion
            ? { duration: 0 }
            : { duration: 1.6, repeat: status === "scanning" ? Infinity : 0, ease: "easeInOut" }
        }
      >
        <IconFaceId huge />
      </motion.div>
      <div className="mt-3">
        <Title>
          {status === "success" ? "Unlocked" : status === "fail" ? "Face Not Recognized" : "Looking for you…"}
        </Title>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   PRIMITIVES
   ═══════════════════════════════════════════════════════════════════════ */

function Row({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex items-center", className)}>{children}</div>;
}

function Title({ children }: { children: React.ReactNode }) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 1, y: 0 } : { y: 8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={reduceMotion ? { duration: 0 } : CONTENT_SPRING}
      className="truncate text-[14px] font-semibold leading-tight text-white"
    >
      {children}
    </motion.div>
  );
}

function Sub({ children, className }: { children: React.ReactNode; className?: string }) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 1, y: 0 } : { y: 8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={reduceMotion ? { duration: 0 } : { ...CONTENT_SPRING, delay: 0.04 }}
      className={cn("truncate text-[12px] leading-tight text-white/70", className)}
    >
      {children}
    </motion.div>
  );
}

function CtrlBtn({
  children,
  ariaLabel,
  big,
}: {
  children: React.ReactNode;
  ariaLabel: string;
  big?: boolean;
}) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.span
      aria-label={ariaLabel}
      whileTap={reduceMotion ? undefined : { scale: 0.9 }}
      className={cn(
        "grid place-items-center rounded-full text-white/90",
        big ? "h-10 w-10 bg-white/10" : "h-8 w-8"
      )}
    >
      {children}
    </motion.span>
  );
}

function ActionBtn({
  children,
  ariaLabel,
  className,
}: {
  children: React.ReactNode;
  ariaLabel: string;
  className?: string;
}) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.span
      aria-label={ariaLabel}
      whileTap={reduceMotion ? undefined : { scale: 0.96 }}
      className={cn(
        "inline-flex h-10 flex-1 items-center justify-center gap-2 rounded-full px-4 text-[13px] font-semibold text-white transition-colors",
        className
      )}
    >
      {children}
    </motion.span>
  );
}

/* ─────────────────────── faux camera lens accent ────────────────────── */

function CameraLens() {
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2"
      style={{
        width: 9,
        height: 9,
        borderRadius: 999,
        background:
          "radial-gradient(circle at 35% 30%, #2a2a2e 0%, #0a0a0b 60%, #000 100%)",
        boxShadow:
          "inset 0 0 0 1px rgba(255,255,255,0.04), inset 0 1px 1px rgba(255,255,255,0.05)",
      }}
    />
  );
}

/* ───────────────────── gooey filter for split mode ──────────────────── */

function GooeyDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden focusable="false">
      <defs>
        <filter id="di-goo">
          <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
          <feColorMatrix
            in="blur"
            mode="matrix"
            values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -10"
            result="goo"
          />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
      </defs>
    </svg>
  );
}

/* ───────────────────────── progress ring ────────────────────────────── */

function ProgressRing({
  value,
  size = 28,
  stroke = 2.5,
  color = "#fff",
}: {
  value: number;
  size?: number;
  stroke?: number;
  color?: string;
}) {
  const v = clamp01(value);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const target = useMotionValue(c * (1 - v));
  React.useEffect(() => {
    target.set(c * (1 - v));
  }, [v, c, target]);
  const offset = useSpring(target, { stiffness: 120, damping: 24, mass: 0.6 });
  const dash = useTransform(offset, (o) => `${c - o} ${o}`);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={stroke} />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ strokeDasharray: dash }}
      />
    </svg>
  );
}

/* ─────────────────────────────── battery ────────────────────────────── */

function Battery({ value, big, charging }: { value: number; big?: boolean; charging?: boolean }) {
  const w = big ? 36 : 22;
  const h = big ? 14 : 10;
  const pct = clamp01(value);
  const fillColor = charging ? "#34d399" : pct < 0.2 ? "#f87171" : "#fff";
  return (
    <span className="relative inline-flex items-center" style={{ width: w + 3, height: h }}>
      <span
        className="relative block rounded-[3px] border border-white/40"
        style={{ width: w, height: h }}
      >
        <span
          className="absolute left-[1px] top-[1px] bottom-[1px] rounded-[2px] transition-[width] duration-300"
          style={{ width: `${Math.max(2, pct * (w - 2))}px`, background: fillColor }}
        />
      </span>
      <span
        className="ml-[1px] inline-block rounded-r-[2px] bg-white/40"
        style={{ width: 2, height: h - 4 }}
      />
    </span>
  );
}

/* ────────────────────────── audio waveform ──────────────────────────── */

function Waveform({ color = "#fff", tall }: { color?: string; tall?: boolean }) {
  const refs = React.useRef<Array<HTMLSpanElement | null>>([]);
  const reduceMotion = useDynamicIslandReducedMotion();

  React.useEffect(() => {
    let frame = 0;
    const setHeight = (time: number) => {
      for (let i = 0; i < refs.current.length; i++) {
        const el = refs.current[i];
        if (!el) continue;
        const phase = time / 220 + i * 0.9;
        const value = (Math.sin(phase) + 1) / 2;
        const height = tall ? 8 + value * 22 : 4 + value * 12;
        el.style.height = `${height}px`;
      }
    };

    if (reduceMotion) {
      setHeight(0);
      return;
    }

    const step = (time: number) => {
      setHeight(time);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [reduceMotion, tall]);

  return (
    <span className="flex items-end gap-[3px]">
      {Array.from({ length: 5 }).map((_, i) => (
        <span
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          style={{ background: color, width: 2.5, height: 4, borderRadius: 2 }}
        />
      ))}
    </span>
  );
}

/* ────────────────────────────── pulses ──────────────────────────────── */

function Pulse() {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.span
      className="block h-2 w-2 rounded-full bg-emerald-400"
      animate={reduceMotion ? { opacity: 1, scale: 1 } : { opacity: [1, 0.35, 1], scale: [1, 0.85, 1] }}
      transition={reduceMotion ? { duration: 0 } : { duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}

function PulseRing({ big: _big }: { big?: boolean }) {
  const reduceMotion = useDynamicIslandReducedMotion();

  return (
    <motion.span
      aria-hidden
      className="absolute inset-0 rounded-full ring-2 ring-emerald-400"
      animate={reduceMotion ? { opacity: 0.6, scale: 1 } : { opacity: [0.6, 0, 0.6], scale: [1, 1.6, 1] }}
      transition={reduceMotion ? { duration: 0 } : { duration: 1.8, repeat: Infinity, ease: "easeOut" }}
    />
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════════════════════════════════ */

function IconChat({ className }: { className?: string }) {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
      <path d="M21 12a8 8 0 1 1-3.2-6.4L21 4l-1 4.2A8 8 0 0 1 21 12Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconFile({ className }: { className?: string }) {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
      <path d="M14 3v6h6" />
    </svg>
  );
}

function IconPhone() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19.5 15.4c-1.2 0-2.3-.2-3.4-.5-.3-.1-.7 0-1 .2l-2 2c-2.6-1.3-4.7-3.5-6.1-6.1l2-2c.3-.3.4-.6.2-1-.4-1-.6-2.2-.6-3.4 0-.6-.4-1-1-1H4.2C3.5 3.6 3 4 3 4.6 3 13.7 10.3 21 19.4 21c.6 0 1-.4 1-1v-3.6c0-.6-.4-1-.9-1Z" />
    </svg>
  );
}

function IconPhoneDown() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19.5 15.4c-1.2 0-2.3-.2-3.4-.5-.3-.1-.7 0-1 .2l-2 2c-2.6-1.3-4.7-3.5-6.1-6.1l2-2c.3-.3.4-.6.2-1-.4-1-.6-2.2-.6-3.4 0-.6-.4-1-1-1H4.2C3.5 3.6 3 4 3 4.6 3 13.7 10.3 21 19.4 21c.6 0 1-.4 1-1v-3.6c0-.6-.4-1-.9-1Z" transform="rotate(135 12 12)"/>
    </svg>
  );
}

function IconBell({ big }: { big?: boolean }) {
  const s = big ? 18 : 13;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>
  );
}

function IconBellOff({ big }: { big?: boolean }) {
  const s = big ? 18 : 13;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13.7 21a1.94 1.94 0 0 1-3.4 0" />
      <path d="M18.6 13.7C19.5 16.4 21 17 21 17H6" />
      <path d="M6 8c0-.7.1-1.4.3-2M18 8a6 6 0 0 0-9.3-5" />
      <path d="M2 2l20 20" />
    </svg>
  );
}

function IconPause() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>;
}
function IconPrev() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM20 6 9 12l11 6z" /></svg>;
}
function IconNext() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM4 6l11 6L4 18z" /></svg>;
}

function IconHourglass({ big }: { big?: boolean }) {
  const s = big ? 18 : 13;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 22h14M5 2h14M17 22v-4.2a2 2 0 0 0-.59-1.42L12 12l-4.41 4.38A2 2 0 0 0 7 17.8V22M7 2v4.2c0 .53.21 1.04.59 1.42L12 12l4.41-4.38A2 2 0 0 0 17 6.2V2" />
    </svg>
  );
}

function IconArrowTurn({ big }: { big?: boolean }) {
  const s = big ? 22 : 14;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 18V9a4 4 0 0 1 4-4h9" />
      <path d="m14 1 4 4-4 4" />
    </svg>
  );
}

function IconNavTriangle({ big }: { big?: boolean }) {
  const s = big ? 18 : 13;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2 22 22 12 17 2 22z" />
    </svg>
  );
}

function IconBolt({ big }: { big?: boolean }) {
  const s = big ? 22 : 14;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="currentColor">
      <path d="M13 2 3 14h7l-1 8 10-12h-7z" />
    </svg>
  );
}

function IconAirpods({ big }: { big?: boolean }) {
  const s = big ? 22 : 14;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="currentColor">
      <path d="M7 3a4 4 0 0 0-4 4v6a3 3 0 0 0 3 3h1V8a3 3 0 0 1 3-3 4 4 0 0 0-3-2zM17 3a4 4 0 0 1 4 4v6a3 3 0 0 1-3 3h-1V8a3 3 0 0 0-3-3 4 4 0 0 1 3-2z" />
      <rect x="6.5" y="14" width="3" height="7" rx="1.5" />
      <rect x="14.5" y="14" width="3" height="7" rx="1.5" />
    </svg>
  );
}

function IconFaceId({ big, huge }: { big?: boolean; huge?: boolean }) {
  const s = huge ? 56 : big ? 22 : 14;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 8V6a2 2 0 0 1 2-2h2" />
      <path d="M16 4h2a2 2 0 0 1 2 2v2" />
      <path d="M20 16v2a2 2 0 0 1-2 2h-2" />
      <path d="M8 20H6a2 2 0 0 1-2-2v-2" />
      <path d="M9 9v2M15 9v2" />
      <path d="M9 16s1 1 3 1 3-1 3-1" />
      <path d="M12 11v3h-1" />
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════════════ */

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

function initials(name: string) {
  return name.split(" ").map((s) => s[0]).slice(0, 2).join("").toUpperCase();
}

function formatTimer(ms: number) {
  const total = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}
