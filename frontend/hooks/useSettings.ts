import { useState, useEffect } from 'react';

export interface AppSettings {
  // API Configuration
  apiUrl: string;
  apiTimeout: number;

  // Training Defaults
  defaultEpochs: number;
  defaultBatchSize: number;
  defaultLearningRate: number;

  // UI Preferences
  autoRefresh: boolean;
  refreshInterval: number;
  showAdvancedOptions: boolean;

  // Training Parameter Visibility
  showAdvancedLycoris: boolean;
  showBlockwiseLR: boolean;
  showSD2Params: boolean;
  showPerformanceTuning: boolean;
  showExperimentalFeatures: boolean;

  // File Management
  autoCleanup: boolean;
  maxStorageGB: number;
}

const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: 'http://localhost:8000',
  apiTimeout: 30,
  defaultEpochs: 10,
  defaultBatchSize: 1,
  defaultLearningRate: 0.0001,
  autoRefresh: true,
  refreshInterval: 5,
  showAdvancedOptions: false,
  // Alpha: Show all advanced parameters by default
  showAdvancedLycoris: true,
  showBlockwiseLR: true,
  showSD2Params: true,
  showPerformanceTuning: true,
  showExperimentalFeatures: true,
  autoCleanup: false,
  maxStorageGB: 50,
};

export function useSettings(): AppSettings {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    // Load settings from localStorage on mount
    const loadSettings = () => {
      try {
        const stored = localStorage.getItem('ktiseos-nyx-settings');
        if (stored) {
          const parsed = JSON.parse(stored);
          setSettings({ ...DEFAULT_SETTINGS, ...parsed });
        }
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    };

    loadSettings();

    // Listen for storage changes (settings updated in another tab)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'ktiseos-nyx-settings' && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue);
          setSettings({ ...DEFAULT_SETTINGS, ...parsed });
        } catch (error) {
          console.error('Failed to parse settings update:', error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  return settings;
}
