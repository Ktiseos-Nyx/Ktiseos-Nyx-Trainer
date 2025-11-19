'use client'

import { useState, useEffect } from 'react'
import { Home, Save, RotateCcw, Settings } from 'lucide-react'
import { GradientCard } from '@/components/effects'
import Breadcrumbs from '@/components/Breadcrumbs'

export default function SettingsPage() {
  // API Configuration
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [apiTimeout, setApiTimeout] = useState(30)

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

  // Load settings from localStorage on mount
  useEffect(() => {
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
  }, [])

  const handleSave = () => {
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
    alert('Settings saved successfully!')
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-16">
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
          <p className="text-xl text-gray-300">
            Configure your LoRA trainer preferences
          </p>
        </div>

        {/* API Configuration */}
        <GradientCard variant="dusk" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-white mb-4">API Configuration</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  API Base URL
                </label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="http://localhost:8000"
                />
                <p className="text-xs text-gray-500 mt-1">
                  The backend API endpoint for training operations
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  API Timeout (seconds)
                </label>
                <input
                  type="number"
                  value={apiTimeout}
                  onChange={(e) => setApiTimeout(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  min="5"
                  max="300"
                />
              </div>
            </div>
          </div>
        </GradientCard>

        {/* Training Defaults */}
        <GradientCard variant="ocean" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-white mb-4">Training Defaults</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Default Epochs
                </label>
                <input
                  type="number"
                  value={defaultEpochs}
                  onChange={(e) => setDefaultEpochs(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  min="1"
                  max="100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Default Batch Size
                </label>
                <input
                  type="number"
                  value={defaultBatchSize}
                  onChange={(e) => setDefaultBatchSize(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  min="1"
                  max="32"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Default Learning Rate
                </label>
                <input
                  type="number"
                  value={defaultLearningRate}
                  onChange={(e) => setDefaultLearningRate(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  step="0.00001"
                  min="0.00001"
                  max="0.01"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Scientific notation: {defaultLearningRate.toExponential(2)}
                </p>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* UI Preferences */}
        <GradientCard variant="cotton-candy" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-white mb-4">UI Preferences</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-300">
                    Auto-refresh Training Monitor
                  </label>
                  <p className="text-xs text-gray-500">
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
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Refresh Interval (seconds)
                  </label>
                  <input
                    type="number"
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="1"
                    max="60"
                  />
                </div>
              )}

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-300">
                    Show Advanced Options by Default
                  </label>
                  <p className="text-xs text-gray-500">
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
        <GradientCard variant="sunset" intensity="subtle" className="mb-6">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-white mb-2">Training Parameter Visibility</h2>
            <p className="text-sm text-gray-400 mb-4">
              Control which advanced training parameters are visible in the Training Config page
            </p>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-300">
                    Advanced LyCORIS Parameters
                  </label>
                  <p className="text-xs text-gray-500">
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
                  <label className="text-sm font-medium text-gray-300">
                    Block-wise Learning Rates
                  </label>
                  <p className="text-xs text-gray-500">
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
                  <label className="text-sm font-medium text-gray-300">
                    SD 2.x Parameters
                  </label>
                  <p className="text-xs text-gray-500">
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
                  <label className="text-sm font-medium text-gray-300">
                    Performance Tuning
                  </label>
                  <p className="text-xs text-gray-500">
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
                  <label className="text-sm font-medium text-gray-300">
                    Experimental Features
                  </label>
                  <p className="text-xs text-gray-500">
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
            <h2 className="text-2xl font-bold text-white mb-4">File Management</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-300">
                    Auto-cleanup Old Files
                  </label>
                  <p className="text-xs text-gray-500">
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
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Maximum Storage (GB)
                </label>
                <input
                  type="number"
                  value={maxStorageGB}
                  onChange={(e) => setMaxStorageGB(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
                  min="10"
                  max="500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Alert when storage exceeds this limit
                </p>
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
            className="flex items-center justify-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white font-semibold rounded-lg transition-all duration-300 border border-slate-700"
          >
            <RotateCcw className="w-5 h-5" />
            Reset to Defaults
          </button>
        </div>

        {/* Info Note */}
        <div className="mt-8 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
          <p className="text-sm text-gray-300">
            <strong className="text-blue-400">Note:</strong> Settings are saved to your browser's local storage. They will persist across sessions but are device-specific.
          </p>
        </div>
      </div>
    </div>
  )
}
