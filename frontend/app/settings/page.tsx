'use client'

import { useState, useEffect } from 'react'
import { Home, Save, RotateCcw, Settings, Key, Eye, EyeOff } from 'lucide-react'
import { GradientCard } from '@/components/effects'
import Breadcrumbs from '@/components/Breadcrumbs'

export default function SettingsPage() {
  // API Configuration
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [apiTimeout, setApiTimeout] = useState(30)

  // API Keys (stored on backend)
  const [huggingfaceToken, setHuggingfaceToken] = useState('')
  const [civitaiApiKey, setCivitaiApiKey] = useState('')
  const [hasHuggingfaceToken, setHasHuggingfaceToken] = useState(false)
  const [hasCivitaiApiKey, setHasCivitaiApiKey] = useState(false)
  const [showHfToken, setShowHfToken] = useState(false)
  const [showCivitaiKey, setShowCivitaiKey] = useState(false)

  // Training Defaults
  const [defaultEpochs, setDefaultEpochs] = useState(10)
  const [defaultBatchSize, setDefaultBatchSize] = useState(1)
  const [defaultLearningRate, setDefaultLearningRate] = useState(0.0001)

  // UI Preferences
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5)
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)

  // Training Parameter Visibility
  const [showAdvancedLycoris, setShowAdvancedLycoris] = useState(false)
  const [showBlockwiseLR, setShowBlockwiseLR] = useState(false)
  const [showSD2Params, setShowSD2Params] = useState(false)
  const [showPerformanceTuning, setShowPerformanceTuning] = useState(false)
  const [showExperimentalFeatures, setShowExperimentalFeatures] = useState(false)

  // File Management
  const [autoCleanup, setAutoCleanup] = useState(false)
  const [maxStorageGB, setMaxStorageGB] = useState(50)

  // Storage info
  const [storageInfo, setStorageInfo] = useState<{
    total_gb: number
    used_gb: number
    free_gb: number
    used_percent: number
    path: string
  } | null>(null)

  // Load settings from localStorage and API keys from backend on mount
  useEffect(() => {
    // Load UI settings from localStorage
    try {
      const stored = localStorage.getItem('ktiseos-nyx-settings')
      if (stored) {
        const settings = JSON.parse(stored)
        setApiUrl(settings.apiUrl ?? 'http://localhost:8000')
        setApiTimeout(settings.apiTimeout ?? 30)
        setDefaultEpochs(settings.defaultEpochs ?? 10)
        setDefaultBatchSize(settings.defaultBatchSize ?? 1)
        setDefaultLearningRate(settings.defaultLearningRate ?? 0.0001)
        setAutoRefresh(settings.autoRefresh ?? true)
        setRefreshInterval(settings.refreshInterval ?? 5)
        setShowAdvancedOptions(settings.showAdvancedOptions ?? false)
        setShowAdvancedLycoris(settings.showAdvancedLycoris ?? false)
        setShowBlockwiseLR(settings.showBlockwiseLR ?? false)
        setShowSD2Params(settings.showSD2Params ?? false)
        setShowPerformanceTuning(settings.showPerformanceTuning ?? false)
        setShowExperimentalFeatures(settings.showExperimentalFeatures ?? false)
        setAutoCleanup(settings.autoCleanup ?? false)
        setMaxStorageGB(settings.maxStorageGB ?? 50)
      }
    } catch (error) {
      console.error('Failed to load settings:', error)
    }

    // Load API keys and storage info from backend
    loadApiKeys()
    loadStorageInfo()
  }, [])

  const loadStorageInfo = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/settings/storage`)
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setStorageInfo(data.storage)
        }
      }
    } catch (error) {
      console.error('Failed to load storage info:', error)
    }
  }

  const loadApiKeys = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/settings/user`)
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setHasHuggingfaceToken(data.settings.has_huggingface_token)
          setHasCivitaiApiKey(data.settings.has_civitai_api_key)
        }
      }
    } catch (error) {
      console.error('Failed to load API keys:', error)
    }
  }

  const saveApiKeys = async () => {
    try {
      const payload: any = {}

      // Only include keys if they've been entered (not empty)
      if (huggingfaceToken.trim()) {
        payload.huggingface_token = huggingfaceToken
      }
      if (civitaiApiKey.trim()) {
        payload.civitai_api_key = civitaiApiKey
      }

      if (Object.keys(payload).length === 0) {
        return true // Nothing to save
      }

      const response = await fetch(`${apiUrl}/api/settings/user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (response.ok) {
        // Clear input fields after successful save
        setHuggingfaceToken('')
        setCivitaiApiKey('')
        // Reload to get updated status
        await loadApiKeys()
        return true
      }
      return false
    } catch (error) {
      console.error('Failed to save API keys:', error)
      return false
    }
  }

  const handleSave = async () => {
    // Save API keys first
    const apiKeysSaved = await saveApiKeys()

    // Save UI settings to localStorage
    const settings = {
      apiUrl,
      apiTimeout,
      defaultEpochs,
      defaultBatchSize,
      defaultLearningRate,
      autoRefresh,
      refreshInterval,
      showAdvancedOptions,
      showAdvancedLycoris,
      showBlockwiseLR,
      showSD2Params,
      showPerformanceTuning,
      showExperimentalFeatures,
      autoCleanup,
      maxStorageGB,
    }
    localStorage.setItem('ktiseos-nyx-settings', JSON.stringify(settings))
    console.log('Settings saved:', settings)

    if (apiKeysSaved) {
      alert('Settings saved successfully!')
    } else {
      alert('UI settings saved, but there was an error saving API keys. Please try again.')
    }
  }

  const handleReset = () => {
    if (confirm('Reset all settings to defaults?')) {
      setApiUrl('http://localhost:8000')
      setApiTimeout(30)
      setDefaultEpochs(10)
      setDefaultBatchSize(1)
      setDefaultLearningRate(0.0001)
      setAutoRefresh(true)
      setRefreshInterval(5)
      setShowAdvancedOptions(false)
      setShowAdvancedLycoris(false)
      setShowBlockwiseLR(false)
      setShowSD2Params(false)
      setShowPerformanceTuning(false)
      setShowExperimentalFeatures(false)
      setAutoCleanup(false)
      setMaxStorageGB(50)
      localStorage.removeItem('ktiseos-nyx-settings')
    }
  }

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Settings', icon: <Settings className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            Settings
          </h1>
          <p className="text-xl text-muted-foreground">
            Configure your LoRA trainer preferences
          </p>
        </div>

        {/* API Configuration */}
        <GradientCard variant="dusk" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-4">API Configuration</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  API Base URL
                </label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="w-full px-4 py-2 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="http://localhost:8000"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  The backend API endpoint for training operations
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  API Timeout (seconds)
                </label>
                <input
                  type="number"
                  value={apiTimeout}
                  onChange={(e) => setApiTimeout(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-purple-500"
                  min="5"
                  max="300"
                />
              </div>
            </div>
          </div>
        </GradientCard>

        {/* API Keys */}
        <GradientCard variant="ocean" intensity="subtle" className="mb-6">
          <div className="p-6">
            <div className="flex items-center gap-2 mb-2">
              <Key className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
              <h2 className="text-2xl font-bold text-foreground">API Keys</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Store API keys securely on the backend for automatic use during model downloads
            </p>

            <div className="space-y-4">
              {/* HuggingFace Token */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  HuggingFace Token {hasHuggingfaceToken && <span className="text-green-600 dark:text-green-400 text-xs">(✓ Saved)</span>}
                </label>
                <div className="relative">
                  <input
                    type={showHfToken ? 'text' : 'password'}
                    value={huggingfaceToken}
                    onChange={(e) => setHuggingfaceToken(e.target.value)}
                    className="w-full px-4 py-2 pr-10 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder={hasHuggingfaceToken ? "Enter new token to update..." : "hf_..."}
                  />
                  <button
                    type="button"
                    onClick={() => setShowHfToken(!showHfToken)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showHfToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Used for downloading models from HuggingFace. Get your token from{' '}
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline"
                  >
                    huggingface.co/settings/tokens
                  </a>
                </p>
              </div>

              {/* Civitai API Key */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Civitai API Key {hasCivitaiApiKey && <span className="text-green-600 dark:text-green-400 text-xs">(✓ Saved)</span>}
                </label>
                <div className="relative">
                  <input
                    type={showCivitaiKey ? 'text' : 'password'}
                    value={civitaiApiKey}
                    onChange={(e) => setCivitaiApiKey(e.target.value)}
                    className="w-full px-4 py-2 pr-10 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder={hasCivitaiApiKey ? "Enter new key to update..." : "Your API key..."}
                  />
                  <button
                    type="button"
                    onClick={() => setShowCivitaiKey(!showCivitaiKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showCivitaiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Used for downloading models from Civitai. Get your key from{' '}
                  <a
                    href="https://civitai.com/user/account"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline"
                  >
                    civitai.com/user/account
                  </a>
                </p>
              </div>

              {/* Info Box */}
              <div className="mt-4 p-3 bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-500/30 rounded-lg">
                <p className="text-xs text-foreground">
                  <strong>Security:</strong> API keys are stored securely on the backend server and will be automatically used when downloading models from the respective platforms.
                </p>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* Training Defaults */}
        <GradientCard variant="watermelon" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-4">Training Defaults</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Default Epochs
                </label>
                <input
                  type="number"
                  value={defaultEpochs}
                  onChange={(e) => setDefaultEpochs(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  min="1"
                  max="100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Default Batch Size
                </label>
                <input
                  type="number"
                  value={defaultBatchSize}
                  onChange={(e) => setDefaultBatchSize(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  min="1"
                  max="32"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Default Learning Rate
                </label>
                <input
                  type="number"
                  value={defaultLearningRate}
                  onChange={(e) => setDefaultLearningRate(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-input border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  step="0.00001"
                  min="0.00001"
                  max="0.01"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Scientific notation: {defaultLearningRate.toExponential(2)}
                </p>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* UI Preferences */}
        <GradientCard variant="cotton-candy" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-4">UI Preferences</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Auto-refresh Training Monitor
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Automatically update training progress
                  </p>
                </div>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoRefresh ? 'bg-purple-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoRefresh ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {autoRefresh && (
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Refresh Interval (seconds)
                  </label>
                  <input
                    type="number"
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    className="w-full px-4 py-2 bg-amber-50 dark:bg-slate-800 border border-input rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="1"
                    max="60"
                  />
                </div>
              )}

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Show Advanced Options by Default
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Display advanced training parameters
                  </p>
                </div>
                <button
                  onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showAdvancedOptions ? 'bg-purple-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showAdvancedOptions ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* Training Parameter Visibility */}
        <GradientCard variant="dusk" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-2">Training Parameter Visibility</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Control which advanced training parameters are visible in the Training Config page
            </p>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Advanced LyCORIS Parameters
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Show rank_dropout, module_dropout (for LyCORIS training)
                  </p>
                </div>
                <button
                  onClick={() => setShowAdvancedLycoris(!showAdvancedLycoris)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showAdvancedLycoris ? 'bg-orange-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showAdvancedLycoris ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Block-wise Learning Rates
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Show per-block LR weights, dims, alphas (very advanced)
                  </p>
                </div>
                <button
                  onClick={() => setShowBlockwiseLR(!showBlockwiseLR)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showBlockwiseLR ? 'bg-orange-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showBlockwiseLR ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    SD 2.x Parameters
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Show v2, v_parameterization (for SD 2.0/2.1 models)
                  </p>
                </div>
                <button
                  onClick={() => setShowSD2Params(!showSD2Params)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showSD2Params ? 'bg-orange-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showSD2Params ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Performance Tuning
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Show persistent_data_loader_workers, no_token_padding
                  </p>
                </div>
                <button
                  onClick={() => setShowPerformanceTuning(!showPerformanceTuning)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showPerformanceTuning ? 'bg-orange-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showPerformanceTuning ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Experimental Features
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Show weighted_captions, save_last_n_steps_state
                  </p>
                </div>
                <button
                  onClick={() => setShowExperimentalFeatures(!showExperimentalFeatures)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    showExperimentalFeatures ? 'bg-orange-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showExperimentalFeatures ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* File Management */}
        <GradientCard variant="watermelon" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-4">File Management</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Auto-cleanup Old Files
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Automatically remove old training outputs
                  </p>
                </div>
                <button
                  onClick={() => setAutoCleanup(!autoCleanup)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoCleanup ? 'bg-pink-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoCleanup ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-3">
                  Disk Storage
                </label>

                {storageInfo ? (
                  <div className="space-y-3">
                    {/* Storage Usage Bar */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-foreground font-medium">
                          {storageInfo.used_gb} GB used of {storageInfo.total_gb} GB
                        </span>
                        <span className="text-muted-foreground">
                          {storageInfo.free_gb} GB free
                        </span>
                      </div>
                      <div className="w-full h-3 bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${
                            storageInfo.used_percent > 90
                              ? 'bg-red-500'
                              : storageInfo.used_percent > 75
                              ? 'bg-orange-500'
                              : 'bg-green-500'
                          }`}
                          style={{ width: `${storageInfo.used_percent}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {storageInfo.used_percent}% of disk capacity used
                      </p>
                    </div>

                    {/* Alert Threshold */}
                    <div className="pt-3 border-t border-border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-foreground">Storage Alert Threshold</span>
                        <span className="text-sm font-medium text-foreground">{maxStorageGB} GB</span>
                      </div>
                      <input
                        type="range"
                        value={maxStorageGB}
                        onChange={(e) => setMaxStorageGB(Number(e.target.value))}
                        className="w-full h-2 bg-amber-50 dark:bg-muted rounded-lg appearance-none cursor-pointer accent-purple-500"
                        min="10"
                        max={Math.ceil(storageInfo.total_gb)}
                        step="10"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Alert when training outputs exceed this size
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Loading storage information...
                  </div>
                )}

            </div>
          </div>
          </div>
        </GradientCard>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <button
            onClick={handleSave}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white font-semibold rounded-lg transition-all duration-300 hover:scale-105"
          >
            <Save className="w-5 h-5" />
            Save Settings
          </button>

          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-card hover:bg-accent text-foreground font-semibold rounded-lg transition-all duration-300 border border-border"
          >
            <RotateCcw className="w-5 h-5" />
            Reset to Defaults
          </button>
        </div>

        {/* Info Note */}
        <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-500/30 rounded-lg">
          <p className="text-sm text-foreground">
            <strong className="text-blue-600 dark:text-blue-400">Note:</strong> Settings are saved to your browser's local storage. They will persist across sessions but are device-specific.
          </p>
        </div>
      </div>
    </div>
  )
}
