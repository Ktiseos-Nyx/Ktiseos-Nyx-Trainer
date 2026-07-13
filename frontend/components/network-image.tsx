"use client";

import { useCallback, useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface NetworkConnection {
  effectiveType?: "slow-2g" | "2g" | "3g" | "4g";
  downlink?: number;
  saveData?: boolean;
  addEventListener?(type: "change", listener: () => void): void;
  removeEventListener?(type: "change", listener: () => void): void;
}

function getConnection(): NetworkConnection | undefined {
  const nav = navigator as Navigator & { connection?: NetworkConnection };
  return nav.connection;
}

export type EffectiveType = "slow-2g" | "2g" | "3g" | "4g" | "unknown";

export interface NetworkStatus {
  effectiveType: EffectiveType;
  downlink: number;
  saveData: boolean;
  isOnline: boolean;
}

function useNetworkStatus(): NetworkStatus {
  const [status, setStatus] = useState<NetworkStatus>(() => {
    if (typeof window === "undefined") {
      return { downlink: 10, effectiveType: "4g", isOnline: true, saveData: false };
    }
    const conn = getConnection();
    const isOnline = navigator.onLine;
    if (!conn) {
      return { downlink: 10, effectiveType: "4g", isOnline, saveData: false };
    }
    return {
      downlink: conn.downlink ?? 10,
      effectiveType: (conn.effectiveType ?? "4g") as EffectiveType,
      isOnline,
      saveData: conn.saveData ?? false,
    };
  });

  const update = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }
    const conn = getConnection();
    const isOnline = navigator.onLine;
    if (!conn) {
      setStatus({ downlink: 10, effectiveType: "4g", isOnline, saveData: false });
      return;
    }
    setStatus({
      downlink: conn.downlink ?? 10,
      effectiveType: (conn.effectiveType ?? "4g") as EffectiveType,
      isOnline,
      saveData: conn.saveData ?? false,
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    const conn = getConnection();
    if (conn?.addEventListener) {
      conn.addEventListener("change", update);
    }
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
      if (conn?.removeEventListener) {
        conn.removeEventListener("change", update);
      }
    };
  }, [update]);

  return status;
}

type ImageTier = "full" | "medium" | "low";

function getImageTier(status: NetworkStatus): ImageTier {
  if (!status.isOnline) {
    return "low";
  }
  if (status.saveData) {
    return "low";
  }
  switch (status.effectiveType) {
    case "4g":
      return "full";
    case "3g":
      return "medium";
    default:
      return "low";
  }
}

function getLoadingStrategy(status: NetworkStatus): "eager" | "lazy" {
  if (!status.isOnline) {
    return "lazy";
  }
  if (status.saveData) {
    return "lazy";
  }
  return status.effectiveType === "4g" ? "eager" : "lazy";
}

function getBadgeLabel(status: NetworkStatus): string {
  if (!status.isOnline) {
    return "Offline";
  }
  if (status.saveData) {
    return "Data Saver";
  }
  switch (status.effectiveType) {
    case "4g":
      return "4G";
    case "3g":
      return "3G";
    case "2g":
    case "slow-2g":
      return "2G";
    default:
      return "4G";
  }
}

interface NetworkImageProps {
  src: string;
  alt: string;
  lowSrc?: string;
  mediumSrc?: string;
  width?: number;
  height?: number;
  className?: string;
  imageClassName?: string;
  offlineClassName?: string;
  badgeClassName?: string;
  showBadge?: boolean;
  loadingStrategy?: "auto" | "eager" | "lazy";
  networkOverride?: Partial<NetworkStatus>;
}

export function NetworkImage({
  src,
  alt,
  lowSrc,
  mediumSrc,
  width,
  height,
  className,
  imageClassName,
  offlineClassName,
  badgeClassName,
  showBadge = false,
  loadingStrategy = "auto",
  networkOverride,
}: NetworkImageProps) {
  const realStatus = useNetworkStatus();
  const status: NetworkStatus = networkOverride ? { ...realStatus, ...networkOverride } : realStatus;
  const tier = getImageTier(status);
  const loading = loadingStrategy === "auto" ? getLoadingStrategy(status) : loadingStrategy;

  const imageSrc = (() => {
    switch (tier) {
      case "full":
        return src;
      case "medium":
        return mediumSrc ?? src;
      case "low":
        return lowSrc ?? mediumSrc ?? src;
    }
  })();

  const isOffline = !status.isOnline;

  return (
    <div className={cn("relative overflow-hidden rounded-lg", className)}>
      {isOffline ? (
        <div
          className={cn("flex items-center justify-center bg-muted text-muted-foreground", offlineClassName)}
          style={{ height: height ?? 200, minHeight: 120, width: width ?? "100%" }}
        >
          <div className="flex flex-col items-center gap-2 text-center">
            <svg aria-hidden={true} className="size-8 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3"
              />
            </svg>
            <span className="font-medium text-sm">Offline</span>
          </div>
        </div>
      ) : (
        <img
          alt={alt}
          className={cn("h-auto w-full object-cover transition-opacity duration-300", imageClassName)}
          height={height}
          loading={loading}
          src={imageSrc}
          width={width}
        />
      )}
      {showBadge && (
        <div
          className={cn(
            "absolute top-2 right-2 rounded-md px-2 py-0.5 font-medium text-xs",
            "bg-background/80 text-foreground backdrop-blur-sm",
            isOffline && "bg-destructive/80 text-destructive-foreground",
            badgeClassName,
          )}
        >
          {getBadgeLabel(status)}
        </div>
      )}
    </div>
  );
}
