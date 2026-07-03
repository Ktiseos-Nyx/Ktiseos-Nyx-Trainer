'use client';

import { useState } from 'react';
import { parseImageMetadata, formatMetadataForDisplay } from '@/lib/dataset-tools/parseImageMetadata';
import { Input } from '@/components/ui/input';

export function MetadataViewer() {
  const [metadata, setMetadata] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setMetadata(null);

    try {
      const result = await parseImageMetadata(file);

      if (result) {
        setMetadata(formatMetadataForDisplay(result));
      } else {
        setError('No metadata found in this image');
      }
    } catch (err) {
      setError(`Error parsing metadata: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">
          Upload Image (PNG or JPEG)
        </label>
        <Input
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileChange}
          className="cursor-pointer"
        />
      </div>

      {loading && (
        <div className="text-sm text-gray-600">Parsing metadata...</div>
      )}

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {metadata && (
        <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
          <pre className="text-xs whitespace-pre-wrap font-mono overflow-x-auto">
            {metadata}
          </pre>
        </div>
      )}
    </div>
  );
}
