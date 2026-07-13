'use client';

import { LoRAFile } from '@/lib/api';
import { utilitiesAPI } from '@/lib/api';

export type { LoRAFile } from '@/lib/api';

export type ListedFile = LoRAFile & { source?: string };

export function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-lg p-6 space-y-4">
      <h2 className="text-lg font-semibold text-foreground">{title}</h2>
      {children}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      {message}
    </div>
  );
}

export function SuccessBanner({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-green-500/50 bg-green-500/10 px-4 py-3 text-sm text-green-700 dark:text-green-400 space-y-1">
      {children}
    </div>
  );
}

export async function loadModelFiles(
  sources: Array<{ label: string; dir?: string }>,
  ext: string,
): Promise<ListedFile[]> {
  const present = sources.filter((s): s is { label: string; dir: string } => Boolean(s.dir));
  const lists = await Promise.all(
    present.map(async ({ label, dir }) => {
      const res = await utilitiesAPI.listLoraFiles(dir, ext, 'date');
      return res.success ? res.files.map((f: LoRAFile) => ({ ...f, source: label })) : [];
    }),
  );
  return lists.flat();
}

export { utilitiesAPI };
