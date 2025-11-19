'use client';

import Link from 'next/link';
import { ArrowRight, Sparkles, Settings, LucideIcon } from 'lucide-react';
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
  // Theme-based gradient classes
  const gradients = {
    'purple-blue': {
      bg: 'from-slate-950 via-purple-950 to-slate-950',
      text: 'from-purple-400 via-blue-400 to-purple-400',
      button: 'from-purple-600 to-blue-600',
      buttonHover: 'from-purple-500 to-blue-500',
    },
    'green-earth': {
      bg: 'from-slate-950 via-green-950 to-slate-950',
      text: 'from-green-400 via-emerald-400 to-green-400',
      button: 'from-green-600 to-emerald-600',
      buttonHover: 'from-green-500 to-emerald-500',
    },
    custom: {
      bg: 'from-slate-950 via-purple-950 to-slate-950',
      text: 'from-purple-400 via-blue-400 to-purple-400',
      button: 'from-purple-600 to-blue-600',
      buttonHover: 'from-purple-500 to-blue-500',
    },
  };

  const colors = gradients[theme];

  return (
    <BackgroundBeamsWithCollision className={`h-screen bg-gradient-to-b ${colors.bg}`}>
      <div className="relative z-10 flex flex-col items-center justify-center text-center px-4">
        {/* Main Heading */}
        <h1 className="text-6xl md:text-8xl font-bold mb-4">
          <span className={`bg-gradient-to-r ${colors.text} bg-clip-text text-transparent animate-gradient`}>
            {title}
          </span>
        </h1>

        {subtitle && (
          <h2 className="text-3xl md:text-5xl font-semibold text-white mb-3">
            {subtitle}
          </h2>
        )}

        <p className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto mb-8">
          {description}
        </p>

        {/* Feature Pills */}
        {features.length > 0 && (
          <div className="flex flex-wrap gap-2 justify-center mb-8">
            {features.map((feature, idx) => (
              <div
                key={idx}
                className={`px-3 py-1.5 rounded-full bg-${theme === 'purple-blue' ? 'purple' : 'green'}-500/20 border border-${theme === 'purple-blue' ? 'purple' : 'green'}-500/30 text-${theme === 'purple-blue' ? 'purple' : 'green'}-200 text-sm backdrop-blur-sm`}
              >
                {feature.icon} {feature.label}
              </div>
            ))}
          </div>
        )}

        {/* CTAs */}
        {ctas.length > 0 && (
          <div className="flex flex-col sm:flex-row gap-4 mb-4">
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
          <p className="text-xs text-gray-400">
            {helperText}
          </p>
        )}
      </div>
    </BackgroundBeamsWithCollision>
  );
}
