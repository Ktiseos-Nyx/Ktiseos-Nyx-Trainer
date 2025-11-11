/**
 * API client for backend communication
 * Centralized place for all API calls
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Helper for handling API responses
async function handleResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// ========== File Operations ==========

export interface FileInfo {
  name: string;
  path: string;
  type: 'file' | 'dir';
  size: number;
  modified: number;
  is_image?: boolean;
  mime_type?: string;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  files: FileInfo[];
}

export const fileAPI = {
  list: async (path: string = '/workspace'): Promise<DirectoryListing> => {
    const response = await fetch(`${API_BASE}/files/list?path=${encodeURIComponent(path)}`);
    return handleResponse(response);
  },

  upload: async (file: File, destination: string) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(
      `${API_BASE}/files/upload?destination=${encodeURIComponent(destination)}`,
      {
        method: 'POST',
        body: formData,
      }
    );
    return handleResponse(response);
  },

  delete: async (path: string) => {
    const response = await fetch(
      `${API_BASE}/files/delete?path=${encodeURIComponent(path)}`,
      { method: 'DELETE' }
    );
    return handleResponse(response);
  },

  rename: async (oldPath: string, newName: string) => {
    const response = await fetch(`${API_BASE}/files/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_path: oldPath, new_name: newName }),
    });
    return handleResponse(response);
  },

  mkdir: async (path: string, name: string) => {
    const response = await fetch(`${API_BASE}/files/mkdir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, name }),
    });
    return handleResponse(response);
  },

  read: async (path: string) => {
    const response = await fetch(`${API_BASE}/files/read/${encodeURIComponent(path.substring(1))}`);
    return handleResponse(response);
  },

  write: async (path: string, content: string) => {
    const response = await fetch(`${API_BASE}/files/write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    return handleResponse(response);
  },
};

// ========== Dataset Operations ==========

export const datasetAPI = {
  list: async () => {
    const response = await fetch(`${API_BASE}/dataset/list`);
    return handleResponse(response);
  },

  uploadBatch: async (files: File[], datasetName: string) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(
      `${API_BASE}/dataset/upload-batch?dataset_name=${encodeURIComponent(datasetName)}`,
      {
        method: 'POST',
        body: formData,
      }
    );
    return handleResponse(response);
  },

  tag: async (datasetPath: string, model: string = 'wd14-vit-v2', threshold: number = 0.35) => {
    const response = await fetch(`${API_BASE}/dataset/tag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: datasetPath,
        model,
        threshold,
      }),
    });
    return handleResponse(response);
  },

  create: async (name: string) => {
    const response = await fetch(`${API_BASE}/dataset/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    return handleResponse(response);
  },
};

// ========== Training Operations ==========

export const trainingAPI = {
  start: async (config: any) => {
    const response = await fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  stop: async () => {
    const response = await fetch(`${API_BASE}/training/stop`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  status: async () => {
    const response = await fetch(`${API_BASE}/training/status`);
    return handleResponse(response);
  },

  // WebSocket for logs
  connectLogs: (onMessage: (data: any) => void, onError?: (error: Event) => void) => {
    const wsUrl = `ws://localhost:8000/api/training/logs`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    if (onError) {
      ws.onerror = onError;
    }

    return ws;
  },
};

// ========== Config Operations ==========

export const configAPI = {
  templates: async () => {
    const response = await fetch(`${API_BASE}/config/templates`);
    return handleResponse(response);
  },

  load: async (path: string) => {
    const response = await fetch(`${API_BASE}/config/load?path=${encodeURIComponent(path)}`);
    return handleResponse(response);
  },

  save: async (name: string, config: any) => {
    const response = await fetch(`${API_BASE}/config/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, config }),
    });
    return handleResponse(response);
  },

  defaults: async () => {
    const response = await fetch(`${API_BASE}/config/defaults`);
    return handleResponse(response);
  },
};
