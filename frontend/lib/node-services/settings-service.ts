/**
 * Settings Service - Node.js user settings management
 * Migrated from Python api/routes/settings.py
 *
 * Handles:
 * - User settings (HuggingFace token, Civitai API key)
 * - Settings persistence to JSON file
 * - Storage information
 */

import fs from 'fs/promises';
import path from 'path';

// ========== Types ==========

export interface UserSettings {
  huggingface_token?: string | null;
  civitai_api_key?: string | null;
}

export interface SettingsResponse {
  success: boolean;
  settings?: {
    huggingface_token?: string | null;
    civitai_api_key?: string | null;
    has_huggingface_token: boolean;
    has_civitai_api_key: boolean;
  };
  message?: string;
  error?: string;
}

export interface StorageInfo {
  success: boolean;
  storage?: {
    path: string;
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    total_gb: number;
    used_gb: number;
    free_gb: number;
    used_percent: number;
  };
  error?: string;
}

// ========== Configuration ==========

// Settings directory (in project root)
const PROJECT_ROOT = path.join(process.cwd(), '..');
const SETTINGS_DIR = path.join(PROJECT_ROOT, 'user_config');
const SETTINGS_FILE = path.join(SETTINGS_DIR, 'user_settings.json');

// ========== Helper Functions ==========

/**
 * Ensure settings directory exists with proper permissions
 */
async function ensureSettingsDir(): Promise<void> {
  try {
    // Create directory if it doesn't exist
    await fs.mkdir(SETTINGS_DIR, { recursive: true });

    // Check if writable
    try {
      await fs.access(SETTINGS_DIR, fs.constants.W_OK);
    } catch {
      // Try to fix permissions (might not work on all systems)
      try {
        await fs.chmod(SETTINGS_DIR, 0o777);
        if (await fileExists(SETTINGS_FILE)) {
          await fs.chmod(SETTINGS_FILE, 0o666);
        }
      } catch (error) {
        console.warn('Failed to fix settings directory permissions:', error);
      }
    }
  } catch (error) {
    console.error('Failed to ensure settings directory:', error);
  }
}

/**
 * Check if file exists
 */
async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Mask API token for display (show first 4 and last 4 chars)
 */
function maskToken(token?: string | null): string | null {
  if (!token || token.length < 12) {
    return null;
  }
  return `${token.substring(0, 4)}...${token.substring(token.length - 4)}`;
}

// ========== SettingsService Class ==========

export class SettingsService {
  /**
   * Load settings from JSON file
   */
  async loadSettings(): Promise<UserSettings> {
    await ensureSettingsDir();

    if (!(await fileExists(SETTINGS_FILE))) {
      return {};
    }

    try {
      const content = await fs.readFile(SETTINGS_FILE, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      console.error('Error loading settings:', error);
      return {};
    }
  }

  /**
   * Save settings to JSON file
   */
  async saveSettings(settings: UserSettings): Promise<boolean> {
    await ensureSettingsDir();

    try {
      const content = JSON.stringify(settings, null, 2);
      await fs.writeFile(SETTINGS_FILE, content, 'utf-8');
      return true;
    } catch (error) {
      console.error('Error saving settings:', error);
      return false;
    }
  }

  /**
   * Get user settings with masked tokens
   */
  async getUserSettings(): Promise<SettingsResponse> {
    try {
      const settings = await this.loadSettings();

      return {
        success: true,
        settings: {
          huggingface_token: maskToken(settings.huggingface_token),
          civitai_api_key: maskToken(settings.civitai_api_key),
          has_huggingface_token: Boolean(settings.huggingface_token),
          has_civitai_api_key: Boolean(settings.civitai_api_key),
        },
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get API keys (unmasked) for internal use
   */
  async getApiKeys(): Promise<{ huggingface_token?: string | null; civitai_api_key?: string | null }> {
    const settings = await this.loadSettings();
    return {
      huggingface_token: settings.huggingface_token,
      civitai_api_key: settings.civitai_api_key,
    };
  }

  /**
   * Update user settings
   */
  async updateUserSettings(updates: UserSettings): Promise<SettingsResponse> {
    try {
      const currentSettings = await this.loadSettings();

      // Update only provided fields
      if (updates.huggingface_token !== undefined) {
        currentSettings.huggingface_token = updates.huggingface_token;
      }

      if (updates.civitai_api_key !== undefined) {
        currentSettings.civitai_api_key = updates.civitai_api_key;
      }

      const saved = await this.saveSettings(currentSettings);

      if (!saved) {
        return {
          success: false,
          error: 'Failed to save settings',
        };
      }

      return {
        success: true,
        message: 'Settings updated successfully',
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Clear all user settings
   */
  async clearUserSettings(): Promise<SettingsResponse> {
    try {
      if (await fileExists(SETTINGS_FILE)) {
        await fs.unlink(SETTINGS_FILE);
      }

      return {
        success: true,
        message: 'Settings cleared successfully',
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Delete a specific setting key
   */
  async deleteSettingKey(key: 'huggingface_token' | 'civitai_api_key'): Promise<SettingsResponse> {
    try {
      const validKeys = ['huggingface_token', 'civitai_api_key'];
      if (!validKeys.includes(key)) {
        return {
          success: false,
          error: `Invalid key. Must be one of: ${validKeys.join(', ')}`,
        };
      }

      const currentSettings = await this.loadSettings();

      if (key in currentSettings) {
        delete currentSettings[key as keyof UserSettings];
        await this.saveSettings(currentSettings);
      }

      return {
        success: true,
        message: `Removed ${key}`,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get storage information for the current working directory
   */
  async getStorageInfo(): Promise<StorageInfo> {
    try {
      const cwd = process.cwd();
      const stats = await fs.statfs(cwd);

      const totalBytes = stats.bsize * stats.blocks;
      const freeBytes = stats.bsize * stats.bavail;
      const usedBytes = totalBytes - freeBytes;

      const gb = 1024 ** 3;

      return {
        success: true,
        storage: {
          path: cwd,
          total_bytes: totalBytes,
          used_bytes: usedBytes,
          free_bytes: freeBytes,
          total_gb: Math.round((totalBytes / gb) * 100) / 100,
          used_gb: Math.round((usedBytes / gb) * 100) / 100,
          free_gb: Math.round((freeBytes / gb) * 100) / 100,
          used_percent: Math.round((usedBytes / totalBytes) * 1000) / 10,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to get storage info: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }
}

// Export singleton instance
export const settingsService = new SettingsService();
