import Link from 'next/link';
import { ChevronRight } from 'lucide-react';

interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
}

export default function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <nav className="flex items-center space-x-2 text-sm mb-6">
      {items.map((item, index) => {
        const isLast = index === items.length - 1;

        return (
          <div key={index} className="flex items-center">
            {index > 0 && (
              <ChevronRight className="w-4 h-4 mx-2 text-muted-foreground" />
            )}

            {item.href && !isLast ? (
              <Link
                href={item.href}
                className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
              >
                {item.icon}
                {item.label}
              </Link>
            ) : (
              <span className={`flex items-center gap-2 ${
                isLast
                  ? 'text-foreground font-semibold'
                  : 'text-muted-foreground'
              }`}>
                {item.icon}
                {item.label}
              </span>
            )}
          </div>
        );
      })}
    </nav>
  );
}
