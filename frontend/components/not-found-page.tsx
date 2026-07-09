"use client";

import { motion, useReducedMotion } from "framer-motion";
import { ArrowLeft, Frown } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

interface NotFoundPageProps {
  className?: string;
  homeHref?: string;
  title?: string;
  description?: string;
  helperText?: string;
  backLabel?: string;
  icon?: React.ReactNode;
  buttonClassName?: string;
}

export function NotFoundPage({
  className,
  homeHref = "/",
  title = "404",
  description = "Oops! Page not found",
  helperText = "The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.",
  backLabel = "Back to Home",
  icon,
  buttonClassName,
}: NotFoundPageProps) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={{ opacity: shouldReduceMotion ? 1 : 0, y: shouldReduceMotion ? 0 : 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={shouldReduceMotion ? { duration: 0 } : { duration: 0.5 }}
      className={cn("flex min-h-[60svh] flex-col items-center justify-center space-y-6 text-center", className)}
    >
      <motion.div
        animate={shouldReduceMotion ? { rotate: 0 } : { rotate: [0, 5, -5, 0] }}
        transition={
          shouldReduceMotion
            ? { duration: 0 }
            : {
                duration: 2,
                ease: "easeInOut",
                repeat: Number.POSITIVE_INFINITY,
              }
        }
        className="inline-block"
      >
        {icon ?? <Frown className="mx-auto h-24 w-24 text-muted-foreground" />}
      </motion.div>
      <h1 className="font-bold text-4xl text-foreground">{title}</h1>
      <p className="text-muted-foreground text-xl">{description}</p>
      <p className="mx-auto max-w-md text-muted-foreground">{helperText}</p>
      <Link
        href={homeHref}
        className={cn(
          "mt-4 inline-flex items-center rounded-md bg-primary px-4 py-2 font-medium text-primary-foreground text-sm transition-colors hover:bg-primary/90",
          buttonClassName,
        )}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        {backLabel}
      </Link>
    </motion.div>
  );
}
