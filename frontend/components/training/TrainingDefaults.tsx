'use client'

import { useState, useEffect } from 'react'
import { Save, RotateCcw } from 'lucide-react'
import { GradientCard } from '@/components/effects'

export function TrainingDefaults() {
  const [defaultEpochs, setDefaultEpochs] = useState(10)
  const [defaultBatchSize, setDefaultBatchSize] = useState(1)
  const [defaultLearningRate, setDefaultLearningRate] = useState(0.0001)
  
  // Load settings from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('ktiseos-nyx-settings')
    if (stored) {
      const settings = JSON.parse(stored)
      setDefaultEpochs(settings.defaultEpochs ?? 10)
      setDefaultBatchSize(settings.defaultBatchSize ?? 1)
      setDefaultLearningRate(settings.defaultLearningRate ?? 0.0001)
    }
  }, [])

  const handleSave = () => {
    const stored = localStorage.getItem('ktiseos-nyx-settings')
    const existingSettings = stored ? JSON.parse(stored) : {}
    const newSettings = {
      ...existingSettings,
      defaultEpochs,
      defaultBatchSize,
      defaultLearningRate,
    }
    localStorage.setItem('ktiseos-nyx-settings', JSON.stringify(newSettings))
    alert('Default training settings saved!')
  }

  const handleReset = () => {
    if (confirm('Reset training defaults?')) {
      setDefaultEpochs(10)
      setDefaultBatchSize(1)
      setDefaultLearningRate(0.0001)
      const stored = localStorage.getItem('ktiseos-nyx-settings')
      const existingSettings = stored ? JSON.parse(stored) : {}
      delete existingSettings.defaultEpochs
      delete existingSettings.defaultBatchSize
      delete existingSettings.defaultLearningRate
      localStorage.setItem('ktiseos-nyx-settings', JSON.stringify(existingSettings))
    }
  }

  return (
    <GradientCard variant="watermelon" intensity="subtle" className="mb-6">
      <div className="p-6">
        <h2 className="text-2xl font-bold text-foreground mb-4">Training Defaults</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Set the default values for the training configuration form below.
        </p>
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
        <div className="flex gap-4 mt-6">
          <button
            onClick={handleSave}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-semibold rounded-lg transition-all"
          >
            <Save className="w-4 h-4" />
            Save Defaults
          </button>

          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-card hover:bg-accent text-foreground font-semibold rounded-lg transition-all border border-border"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>
    </GradientCard>
  )
}
