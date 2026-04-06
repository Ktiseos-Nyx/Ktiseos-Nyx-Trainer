'use client';

import { Download } from 'lucide-react';
import type { PropsWithChildren } from 'react';
import { cn } from '@/lib/utils';
import { useDownloadFile } from '@/hooks/use-download-file';

type DownloadFileProps = PropsWithChildren<{
  fileKey: string;
  fileName?: string;
  className?: string;
}>;

export function DownloadFile({
  fileKey,
  fileName,
  children,
  className,
}: DownloadFileProps) {
  const downloadFile = useDownloadFile();

  const handleDownload = () => {
    downloadFile.mutate(
      { fileKey, fileName },
      {
        onSuccess: (url: string) => {
          console.log('Download initiated:', url);
        },
        onError: (err: Error) => {
          console.error(`Download failed: ${(err as Error).message}`);
        },
      },
    );
  };

  return (
    <button
      className={cn('flex items-center gap-2', className)}
      disabled={downloadFile.isPending}
      onClick={handleDownload}
      type="button"
    >
      {children || (
        <>
          <Download className="mr-2 h-4 w-4" />
          {downloadFile.isPending ? 'Downloading...' : 'Download'}
        </>
      )}
    </button>
  );
}
