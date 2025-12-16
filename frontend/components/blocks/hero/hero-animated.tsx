'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowRight, Sparkles, Settings, LucideIcon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { BackgroundBeamsWithCollision } from '@/components/ui/background-beams-with-collision';

export interface HeroAnimatedProps {
  title: string;
  subtitle?: string;
  description: string;
  features?: Array<{
    icon: string;
    label: string;
  }>;
  ctas?: Array<{
    label: string;
    href: string;
    variant: 'primary' | 'secondary';
    icon?: LucideIcon;
  }>;
  helperText?: string;
  theme?: 'purple-blue' | 'green-earth' | 'custom';
}

export function HeroAnimated({
  title,
  subtitle,
  description,
  features = [],
  ctas = [],
  helperText,
  theme = 'purple-blue',
}: HeroAnimatedProps) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setMounted(true), 0);
    return () => clearTimeout(timer);
  }, []);

  // Use dark as default during SSR to match defaultTheme in layout
  const isDark = mounted ? resolvedTheme === 'dark' : true;

  // Theme-based gradient classes with light/dark support
  const gradients = {
    'purple-blue': {
      bg: isDark
        ? 'from-slate-950 via-purple-950 to-slate-950'
        : 'from-purple-50 via-blue-50 to-indigo-50',
      text: isDark
        ? 'from-purple-400 via-blue-400 to-purple-400'
        : 'from-purple-700 via-blue-700 to-indigo-700',
      button: 'from-purple-600 to-blue-600',
      buttonHover: 'from-purple-500 to-blue-500',
      description: isDark ? 'text-gray-300' : 'text-gray-800',
      helperText: isDark ? 'text-gray-400' : 'text-gray-600',
      pillBg: isDark ? 'bg-purple-500/20' : 'bg-purple-500/30',
      pillBorder: isDark ? 'border-purple-500/30' : 'border-purple-500/50',
      pillText: isDark ? 'text-purple-200' : 'text-purple-900',
    },
    'green-earth': {
      bg: isDark
        ? 'from-slate-950 via-green-950 to-slate-950'
        : 'from-green-50 via-emerald-50 to-teal-50',
      text: isDark
        ? 'from-green-400 via-emerald-400 to-green-400'
        : 'from-green-700 via-emerald-700 to-teal-700',
      button: 'from-green-600 to-emerald-600',
      buttonHover: 'from-green-500 to-emerald-500',
      description: isDark ? 'text-gray-300' : 'text-gray-800',
      helperText: isDark ? 'text-gray-400' : 'text-gray-600',
      pillBg: isDark ? 'bg-green-500/20' : 'bg-green-500/30',
      pillBorder: isDark ? 'border-green-500/30' : 'border-green-500/50',
      pillText: isDark ? 'text-green-200' : 'text-green-900',
    },
    custom: {
      bg: isDark
        ? 'from-slate-950 via-purple-950 to-slate-950'
        : 'from-purple-50 via-pink-50 to-blue-50',
      text: isDark
        ? 'from-purple-400 via-blue-400 to-purple-400'
        : 'from-purple-700 via-pink-700 to-blue-700',
      button: 'from-purple-600 to-blue-600',
      buttonHover: 'from-purple-500 to-blue-500',
      description: isDark ? 'text-gray-300' : 'text-gray-800',
      helperText: isDark ? 'text-gray-400' : 'text-gray-600',
      pillBg: isDark ? 'bg-purple-500/20' : 'bg-purple-500/30',
      pillBorder: isDark ? 'border-purple-500/30' : 'border-purple-500/50',
      pillText: isDark ? 'text-purple-200' : 'text-purple-900',
    },
  };

  const colors = gradients[theme];

  return (
    <BackgroundBeamsWithCollision className={`min-h-[500px] flex items-center justify-center bg-gradient-to-b ${colors.bg}`}>
      <div className="relative z-10 flex flex-col items-center justify-center text-center px-4 py-16">
        {/* Glassmorphic Container */}
        <div className={`backdrop-blur-xl rounded-3xl border-2 p-8 md:p-12 max-w-5xl ${
          isDark
            ? 'bg-slate-900/20 border-purple-500/30'
            : 'bg-white/20 border-purple-300/50 shadow-2xl'
        }`}>
          {/* Main Heading */}
          <h1 className="text-6xl md:text-8xl font-bold mb-4" suppressHydrationWarning>
            <span
              className={`bg-gradient-to-r ${colors.text} bg-clip-text text-transparent animate-gradient`}
              suppressHydrationWarning
            >
              {title}
            </span>
          </h1>

          {subtitle && (
            <h2 className={`text-3xl md:text-5xl font-semibold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {subtitle}
            </h2>
          )}

          <p className={`text-lg md:text-xl max-w-2xl mx-auto mb-8 ${colors.description}`}>
            {description}
          </p>

          {/* Feature Pills */}
          {features.length > 0 && (
            <div className="flex flex-wrap gap-2 justify-center mb-8">
              {features.map((feature, idx) => (
                <div
                  key={idx}
                  className={`px-3 py-1.5 rounded-full border text-sm backdrop-blur-sm ${colors.pillBg} ${colors.pillBorder} ${colors.pillText}`}
                >
                  {feature.icon} {feature.label}
                </div>
              ))}
            </div>
          )}

          {/* CTAs */}
          {ctas.length > 0 && (
            <div className="flex flex-col sm:flex-row gap-4 mb-4 justify-center items-center">
              {ctas.map((cta, idx) => {
                const Icon = cta.icon;

                if (cta.variant === 'primary') {
                  return (
                    <Link key={idx} href={cta.href}>
                      <button className={`group relative px-8 py-3 bg-gradient-to-r ${colors.button} text-white text-lg font-semibold rounded-lg hover:${colors.buttonHover} transition-all shadow-lg hover:shadow-xl hover:scale-105`}>
                        <span className="flex items-center gap-2">
                          {Icon && <Icon className="w-5 h-5" />}
                          {cta.label}
                          <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </span>
                      </button>
                    </Link>
                  );
                }

                return (
                  <Link key={idx} href={cta.href}>
                    <button className="px-8 py-3 bg-slate-800/50 backdrop-blur-sm border border-slate-700 text-white text-lg font-semibold rounded-lg hover:bg-slate-700/50 transition-all shadow-lg hover:shadow-xl hover:scale-105">
                      <span className="flex items-center gap-2">
                        {Icon && <Icon className="w-5 h-5" />}
                        {cta.label}
                      </span>
                    </button>
                  </Link>
                );
              })}
            </div>
          )}

          {/* Helper Text */}
          {helperText && (
            <p className={`text-xs ${colors.helperText}`}>
              {helperText}
            </p>
          )}
        </div>
        {/* End Glassmorphic Container */}
      </div>
    </BackgroundBeamsWithCollision>
  );
}
