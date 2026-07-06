/**
 * Civitai API Service - Node.js proxy for Civitai REST API
 * Migrated from Python api/routes/civitai.py
 *
 * REST API Reference: https://github.com/civitai/civitai/wiki/REST-API-Reference
 */

/**
 * The Civitai REST API lives on civitai.com — always.
 * civitai.red is a content mirror, not an API endpoint.
 */
const CIVITAI_API_BASE = 'https://civitai.com/api/v1';

/**
 * Read the configured Civitai download domain from user settings.
 * Used only for constructing download URLs in the popular models list
 * and similar direct-link scenarios. API calls always use civitai.com.
 */
/**
 * Read the configured Civitai download domain from user settings.
 * Used only for constructing download URLs. API calls always use civitai.com.
 */
export async function getCivitaiDownloadBase(): Promise<string> {
  try {
    const { settingsService } = await import('./settings-service');
    const settings = await settingsService.loadSettings();
    return settings.civitai_base_url || 'https://civitai.com';
  } catch {
    return 'https://civitai.com';
  }
}

/**
 * Get Civitai API headers, including API key if available
 */
export function getCivitaiHeaders(apiKey?: string): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  return headers;
}

/**
 * Get Civitai API key from settings
 * For now, we'll pass it from the frontend or environment
 */
export function getCivitaiApiKey(): string | undefined {
  return process.env.CIVITAI_API_KEY;
}

/**
 * Build URL with query parameters.
 * Always uses civitai.com — the REST API is not available on civitai.red.
 */
export function buildUrl(endpoint: string, params: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(`${CIVITAI_API_BASE}${endpoint}`);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.append(key, String(value));
    }
  }

  return url.toString();
}

/**
 * Make a request to Civitai API
 */
export async function civitaiFetch<T>(
  endpoint: string,
  params: Record<string, string | number | boolean | undefined> = {},
  apiKey?: string
): Promise<{ success: boolean; data?: T; error?: string; status?: number }> {
  try {
    const url = buildUrl(endpoint, params);
    const headers = getCivitaiHeaders(apiKey || getCivitaiApiKey());

    const response = await fetch(url, {
      method: 'GET',
      headers,
    });

    if (response.ok) {
      const data = await response.json();
      return { success: true, data };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Civitai API error: ${errorText}`,
        status: response.status,
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
      status: 503,
    };
  }
}

// ========== Type Definitions ==========

export interface CivitaiImage {
  url: string;
  nsfw: boolean;
  width: number;
  height: number;
  hash: string;
  meta?: Record<string, unknown>;
}

export interface CivitaiModelFile {
  name: string;
  id: number;
  sizeKB: number;
  type: string;
  downloadUrl: string;
}

export interface CivitaiModelVersion {
  id: number;
  modelId: number;
  name: string;
  description: string;
  downloadUrl: string;
  trainedWords: string[];
  images: CivitaiImage[];
  files: CivitaiModelFile[];
}

export interface CivitaiModel {
  id: number;
  name: string;
  description: string;
  type: string;
  nsfw: boolean;
  tags: string[];
  creator?: {
    username: string;
    image?: string;
  };
  stats: {
    downloadCount: number;
    favoriteCount: number;
    commentCount: number;
    ratingCount: number;
    rating: number;
  };
  modelVersions: CivitaiModelVersion[];
}

export interface CivitaiTag {
  name: string;
  modelCount: number;
}

export interface CivitaiModelsResponse {
  items: CivitaiModel[];
  metadata: {
    totalItems?: number;
    currentPage?: number;
    pageSize?: number;
    totalPages?: number;
    nextPage?: string;
    prevPage?: string;
    nextCursor?: string;
  };
}

export interface CivitaiTagsResponse {
  items: CivitaiTag[];
  metadata: {
    totalItems?: number;
    currentPage?: number;
    pageSize?: number;
    totalPages?: number;
  };
}
