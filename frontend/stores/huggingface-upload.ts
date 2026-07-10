import { create } from 'zustand';
import type { LoRAFile } from '@/lib/api';

type UploadType = 'lora' | 'dataset' | 'checkpoint';

/** Result returned by the HuggingFace upload backend endpoint. */
interface UploadResult {
  success: boolean;
  repo_id?: string;
  uploaded_files: string[];
  failed_files: string[];
  error?: string;
}

/** All form & UI state for the HuggingFace upload page. */
interface HuggingFaceUploadState {
  uploadType: UploadType;
  hfToken: string;
  owner: string;
  repoName: string;
  repoType: string;
  remoteFolder: string;
  commitMessage: string;
  createPR: boolean;
  selectedFiles: string[];
  availableFiles: LoRAFile[];
  datasetDirectory: string;
  tokenValid: boolean | null;
  useStoredToken: boolean;
  savingToken: boolean;
  tokenSaved: boolean;
  uploading: boolean;
  uploadResult: UploadResult | null;
  error: string | null;
}

/** Actions that mutate the HuggingFace upload store. */
interface HuggingFaceUploadActions {
  setUploadType: (t: UploadType) => void;
  setHfToken: (t: string) => void;
  setOwner: (o: string) => void;
  setRepoName: (n: string) => void;
  setRepoType: (t: string) => void;
  setRemoteFolder: (f: string) => void;
  setCommitMessage: (m: string) => void;
  setCreatePR: (p: boolean) => void;
  setSelectedFiles: (f: string[]) => void;
  setAvailableFiles: (f: LoRAFile[]) => void;
  setDatasetDirectory: (d: string) => void;
  setTokenValid: (v: boolean | null) => void;
  setUseStoredToken: (v: boolean) => void;
  setSavingToken: (v: boolean) => void;
  setTokenSaved: (v: boolean) => void;
  setUploading: (v: boolean) => void;
  setUploadResult: (r: UploadResult | null) => void;
  setError: (e: string | null) => void;
  toggleFileSelection: (filePath: string) => void;
  clearSelections: () => void;
  /** Reset every field to its default value. */
  reset: () => void;
}

const initial: HuggingFaceUploadState = {
  uploadType: 'lora',
  hfToken: '',
  owner: '',
  repoName: '',
  repoType: 'model',
  remoteFolder: '',
  commitMessage: 'Upload via Ktiseos-Nyx-Trainer 🤗',
  createPR: false,
  selectedFiles: [],
  availableFiles: [],
  datasetDirectory: '',
  tokenValid: null,
  useStoredToken: false,
  savingToken: false,
  tokenSaved: false,
  uploading: false,
  uploadResult: null,
  error: null,
};

export const useHuggingFaceUploadStore = create<HuggingFaceUploadState & HuggingFaceUploadActions>((set) => ({
  ...initial,

  setUploadType: (uploadType) => set({ uploadType, selectedFiles: [] }),
  setHfToken: (hfToken) => set({ hfToken, useStoredToken: false, tokenValid: null }),
  setOwner: (owner) => set({ owner }),
  setRepoName: (repoName) => set({ repoName }),
  setRepoType: (repoType) => set({ repoType }),
  setRemoteFolder: (remoteFolder) => set({ remoteFolder }),
  setCommitMessage: (commitMessage) => set({ commitMessage }),
  setCreatePR: (createPR) => set({ createPR }),
  setSelectedFiles: (selectedFiles) => set({ selectedFiles }),
  setAvailableFiles: (availableFiles) => set({ availableFiles }),
  setDatasetDirectory: (datasetDirectory) => set({ datasetDirectory }),
  setTokenValid: (tokenValid) => set({ tokenValid }),
  setUseStoredToken: (useStoredToken) => set({ useStoredToken }),
  setSavingToken: (savingToken) => set({ savingToken }),
  setTokenSaved: (tokenSaved) => set({ tokenSaved }),
  setUploading: (uploading) => set({ uploading }),
  setUploadResult: (uploadResult) => set({ uploadResult }),
  setError: (error) => set({ error }),

  toggleFileSelection: (filePath) =>
    set((state) => ({
      selectedFiles: state.selectedFiles.includes(filePath)
        ? state.selectedFiles.filter((f) => f !== filePath)
        : [...state.selectedFiles, filePath],
    })),

  clearSelections: () => set({ selectedFiles: [] }),

  reset: () => set(initial),
}));
