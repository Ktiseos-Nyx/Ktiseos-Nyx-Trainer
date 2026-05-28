"use client";

import { Children, useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type Direction = "horizontal" | "vertical";

interface ResizablePanelsProps {
  children: React.ReactNode;
  className?: string;
  defaultSizes?: number[];
  direction?: Direction;
  maxSize?: number;
  minSize?: number;
  resizeHandleSize?: number;
}

function getEqualSizes(count: number): number[] {
  const size = 100 / count;
  return Array.from({ length: count }, () => size);
}

const DEFAULT_HANDLE_SIZE = 24;

export function ResizablePanels({
  children,
  className,
  direction = "horizontal",
  defaultSizes,
  minSize = 10,
  maxSize = 90,
  resizeHandleSize = DEFAULT_HANDLE_SIZE,
}: ResizablePanelsProps) {
  const panels = Children.toArray(children);
  const count = panels.length;
  const [sizes, setSizes] = useState<number[]>(() => {
    if (defaultSizes && defaultSizes.length === count) {
      const sum = defaultSizes.reduce((a, b) => a + b, 0);
      return defaultSizes.map((s) => (s / sum) * 100);
    }
    return getEqualSizes(count);
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<{ index: number; startSizes: number[]; startPos: number } | null>(null);

  const clamp = useCallback((value: number) => Math.max(minSize, Math.min(maxSize, value)), [minSize, maxSize]);

  const handlePointerDown = useCallback(
    (index: number) => (e: React.PointerEvent) => {
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      document.body.style.touchAction = "none";
      draggingRef.current = {
        index,
        startPos: direction === "horizontal" ? e.clientX : e.clientY,
        startSizes: [...sizes],
      };
    },
    [sizes, direction],
  );

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      const drag = draggingRef.current;
      if (!drag) {
        return;
      }
      if (!containerRef.current) {
        return;
      }

      const pos = direction === "horizontal" ? e.clientX : e.clientY;
      const delta = pos - drag.startPos;
      const rect = containerRef.current.getBoundingClientRect();
      const total = direction === "horizontal" ? rect.width : rect.height;
      const deltaPercent = (delta / total) * 100;

      const leftIdx = drag.index;
      const rightIdx = drag.index + 1;
      const leftStart = drag.startSizes[leftIdx] ?? 0;
      const rightStart = drag.startSizes[rightIdx] ?? 0;
      const newSizes = [...drag.startSizes];

      let leftNew = leftStart + deltaPercent;
      let rightNew = rightStart - deltaPercent;
      leftNew = clamp(leftNew);
      rightNew = clamp(rightNew);
      const totalAdjacent = leftStart + rightStart;
      const newTotal = leftNew + rightNew;
      if (Math.abs(newTotal - totalAdjacent) > 0.01) {
        const scale = totalAdjacent / newTotal;
        leftNew *= scale;
        rightNew *= scale;
      }
      newSizes[leftIdx] = leftNew;
      newSizes[rightIdx] = rightNew;

      setSizes(newSizes);
    };

    const handlePointerUp = () => {
      draggingRef.current = null;
      document.body.style.cursor = "";
      document.body.style.touchAction = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [direction, clamp]);

  useEffect(() => {
    if (sizes.length !== count) {
      setSizes(getEqualSizes(count));
    }
  }, [count, sizes.length]);

  const isHorizontal = direction === "horizontal";

  const elements: React.ReactNode[] = [];
  for (let i = 0; i < count; i++) {
    elements.push(
      <div key={`panel-${i}`} className="flex min-h-0 min-w-0 overflow-auto" style={{ flex: `${sizes[i] ?? 1} 1 0` }}>
        <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{panels[i]}</div>
      </div>,
    );
    if (i < count - 1) {
      elements.push(
        <div
          key={`handle-${i}`}
          className={cn(
            "flex shrink-0 touch-none items-center justify-center bg-border transition-colors hover:bg-primary/20",
            isHorizontal ? "cursor-col-resize" : "cursor-row-resize",
          )}
          style={{
            ...(isHorizontal ? { width: resizeHandleSize } : { height: resizeHandleSize }),
            touchAction: "none",
          }}
          onPointerDown={handlePointerDown(i)}
        >
          <div className={cn("rounded-full bg-muted-foreground/30", isHorizontal ? "h-8 w-0.5" : "h-0.5 w-8")} />
        </div>,
      );
    }
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex h-full min-h-0 w-full min-w-0 overflow-hidden",
        isHorizontal ? "flex-row" : "flex-col",
        className,
      )}
    >
      {elements}
    </div>
  );
}
