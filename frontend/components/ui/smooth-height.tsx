"use client";

import * as React from "react";
import {
  motion,
  useReducedMotion,
  type HTMLMotionProps,
} from "motion/react";

import { cn } from "@/lib/utils";

type MotionTransition = NonNullable<HTMLMotionProps<"div">["transition"]>;

export type SmoothHeightProps = Omit<
  HTMLMotionProps<"div">,
  "animate" | "children" | "initial" | "transition"
> & {
  children: React.ReactNode;
  innerClassName?: string;
  transition?: MotionTransition;
};

export function SmoothHeight({
  children,
  className,
  innerClassName,
  transition,
  ...props
}: SmoothHeightProps) {
  const [height, setHeight] = React.useState<number | null>(null);
  const contentRef = React.useRef<HTMLDivElement>(null);
  const shouldReduceMotion = useReducedMotion();

  React.useEffect(() => {
    const content = contentRef.current;

    if (!content) return;

    const updateHeight = (nextHeight: number) => {
      setHeight((currentHeight) => {
        if (currentHeight === null) return nextHeight;

        return Math.abs(currentHeight - nextHeight) > 0.5
          ? nextHeight
          : currentHeight;
      });
    };

    updateHeight(content.getBoundingClientRect().height);

    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];

      if (!entry) return;

      const borderBoxSize = Array.isArray(entry.borderBoxSize)
        ? entry.borderBoxSize[0]
        : entry.borderBoxSize;

      updateHeight(
        borderBoxSize?.blockSize ?? entry.target.getBoundingClientRect().height
      );
    });

    observer.observe(content);

    return () => observer.disconnect();
  }, []);

  const resolvedTransition: MotionTransition = shouldReduceMotion
    ? { duration: 0 }
    : transition ?? { duration: 0.3, ease: [0.645, 0.045, 0.355, 1] };

  return (
    <motion.div
      {...props}
      data-slot="smooth-height"
      initial={false}
      animate={{ height: height ?? "auto" }}
      transition={resolvedTransition}
      className={cn("overflow-hidden", className)}
    >
      <div
        ref={contentRef}
        data-slot="smooth-height-content"
        className={innerClassName}
      >
        {children}
      </div>
    </motion.div>
  );
}
