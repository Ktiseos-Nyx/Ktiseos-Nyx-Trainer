import { useMutation } from '@tanstack/react-query';

type DownloadArgs = {
  fileKey: string;
  fileName?: string;
};

export function useDownloadFile() {
  return useMutation({
    mutationFn: async ({ fileKey, fileName }: DownloadArgs) => {
      const downloadRes = await fetch('/api/downloads/presign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: fileKey }),
      });

      if (!downloadRes.ok) throw new Error('Failed to get download URL');
      const { downloadUrl } = await downloadRes.json();

      const response = await fetch(downloadUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName || fileKey.split('/').pop() || 'download';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      return downloadUrl;
    },
  });
}
