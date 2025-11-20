'use client';

import Link from 'next/link';

export interface Feature {
  emoji: string;
  title: string;
  description: string;
}

export interface FeatureGridProps {
  title?: string;
  features: Feature[];
  ctaLabel?: string;
  ctaHref?: string;
  columns?: 2 | 3 | 4;
}

export function FeatureGrid({
  title = 'Everything You Need',
  features,
  ctaLabel,
  ctaHref,
  columns = 3,
}: FeatureGridProps) {
  const gridCols = {
    2: 'md:grid-cols-2',
    3: 'md:grid-cols-3',
    4: 'md:grid-cols-4',
  };

  return (
    <section className="bg-background py-16">
      <div className="container mx-auto px-4 max-w-6xl">
        {title && (
          <h2 className="text-3xl font-bold text-foreground text-center mb-10">
            {title}
          </h2>
        )}

        <div className={`grid ${gridCols[columns]} gap-6 mb-12`}>
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="bg-card backdrop-blur-sm border border-border rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow"
            >
              <div className="text-4xl mb-3">{feature.emoji}</div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                {feature.title}
              </h3>
              <p className="text-muted-foreground text-sm">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {ctaLabel && ctaHref && (
          <div className="text-center">
            <Link href={ctaHref}>
              <button className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:from-purple-500 hover:to-blue-500 transition-all shadow-lg hover:shadow-xl hover:scale-105">
                {ctaLabel}
              </button>
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}
