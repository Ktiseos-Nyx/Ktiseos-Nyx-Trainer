'use client';

import { Home, FileText, Github, Calendar, GitCommit, Sparkles, Bug, Wrench, Zap } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ChangelogEntry {
  version: string;
  date: string;
  type: 'major' | 'minor' | 'patch';
  changes: {
    category: 'feature' | 'improvement' | 'bugfix' | 'breaking' | 'docs';
    items: string[];
  }[];
  commitHash?: string;
}

const changelog: ChangelogEntry[] = [
  {
    version: '0.1.0-dev',
    date: 'January 1, 2026',
    type: 'minor',
    commitHash: '64f4aaa',
    changes: [
      {
        category: 'feature',
        items: [
          'Turbopack enabled by default for Next.js dev server (50% RAM reduction on Windows)',
          'Complete migration to shadcn/ui components - removed all raw HTML form elements',
          'Auto-Tag page UX redesign with unified model selector (#93)',
          'Tag Editor redesigned with modern pill interface (#80)',
          'WebSocket proxy implementation for real-time log streaming',
          'Development environment documentation with platform-specific RAM usage details',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Windows-specific platform fixes (UTF encoding, installer improvements)',
          'Dataset uploader stability with queued one-by-one image uploads (d8a4740)',
          'VastAI deployment configuration improvements',
          'Cross-platform path validation and security hardening',
          'Comprehensive linting and code quality cleanup',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Training logs now properly streaming to WebSocket (#92)',
          'Light mode button visibility fixed across all pages (d277248)',
          'Fixed DatasetCard missing onSave prop destructuring',
          'API routing and backend integration fixes (be23a7b)',
          'Removed dangerouslySetInnerHTML and vanilla JS DOM manipulation',
          'Fixed TOML generation and absolute path handling (e4954f1)',
        ],
      },
      {
        category: 'breaking',
        items: [
          'Completely removed Jupyter notebook dependencies (ipython, ipywidgets)',
          'Migrated to full web UI architecture (FastAPI + Next.js)',
          'Windows now primary development platform (macOS training not supported)',
        ],
      },
      {
        category: 'docs',
        items: [
          'Added DEVELOPMENT_ENVIRONMENTS.md with RAM usage analysis',
          'Updated README with RAM requirements and platform support matrix',
          'Security notices for file manager path validation',
          'VastAI one-click deployment documentation',
        ],
      },
    ],
  },
  {
    version: '0.0.9-dev',
    date: 'December 17, 2025',
    type: 'minor',
    commitHash: '534b198',
    changes: [
      {
        category: 'feature',
        items: [
          'Complete TOML configuration generator for training parameters',
          'Reorganized training configuration tabs for better workflow',
          'Improved API error handling across all endpoints',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Updated CivitAI API key messaging for better user feedback (5ead7d8)',
        ],
      },
    ],
  },
  {
    version: '0.0.8-dev',
    date: 'December 15, 2025',
    type: 'patch',
    commitHash: '5ec8633',
    changes: [
      {
        category: 'improvement',
        items: [
          'Updated ESLint configuration and rules',
          'Aligned dependency versions across requirements.txt',
          'Pre-dependency cleanup pass (61104e3)',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Fixed ESLint errors throughout codebase',
          'Resolved dependency conflicts between vendored packages',
        ],
      },
    ],
  },
  {
    version: '0.0.7-dev',
    date: 'December 14, 2025',
    type: 'minor',
    commitHash: '94891de',
    changes: [
      {
        category: 'feature',
        items: [
          'Added frontend validation module (0149ad6)',
          'Improved job ID handling in training pipeline (6dac218)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Fixed NextJS API routing issues (11f84f2)',
          'Hardened NextJS implementation and cleaned up ignored files (bf7237c)',
          'Updated .gitignore to properly exclude lib/ directory (ae5b4e2)',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Fixed various brain dead mode issues (94891de)',
          'Resolved missing frontend validation module error',
        ],
      },
    ],
  },
  {
    version: '0.0.6-dev',
    date: 'December 13, 2025',
    type: 'minor',
    commitHash: '303e55e',
    changes: [
      {
        category: 'feature',
        items: [
          'Added VastAI one-click deployment section with working template (de5acd6)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Removed chunking implementation to prevent potential I/O issues (303e55e)',
          'Fixed binding configuration in NextJS package.json (8c73729)',
        ],
      },
      {
        category: 'docs',
        items: [
          'Added VastAI deployment documentation',
        ],
      },
    ],
  },
  {
    version: '0.0.5-dev',
    date: 'December 12, 2025',
    type: 'minor',
    commitHash: '9cbbacc',
    changes: [
      {
        category: 'feature',
        items: [
          'Massively improved upload performance with parallel processing (9cbbacc)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Removed Node.js installation code, use VastAI pre-installed version (0e6fe9c)',
          'Removed Jupyter dependencies (ipython, ipywidgets) (31ef740)',
          'Made VastAI provisioning script more resilient (519c08f)',
          'Updated Next.js from 15.4.8 to 15.4.10 (f87a1a3)',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Fixed path traversal vulnerability in tag editor (7d88424)',
          'Added path validation to training configuration (fd8d58a)',
          'Made dataset path validation more forgiving (1921743)',
          'Resolved Caddy port conflict by removing duplicate port mappings (d51440e)',
          'Fixed supervisor configuration - removed race conditions (533de41)',
          'Removed supervisor complexity that was breaking VastAI Portal (ba04871)',
          'Fixed supervisor script search term to match portal config (1eeefa2)',
          'Removed PORTAL_CONFIG override from provisioning script (ddbe898)',
        ],
      },
      {
        category: 'docs',
        items: [
          'Added security notice about file manager (6167ee2)',
        ],
      },
    ],
  },
  {
    version: '0.0.4-dev',
    date: 'December 9, 2025',
    type: 'minor',
    commitHash: 'e63931c',
    changes: [
      {
        category: 'feature',
        items: [
          'Added restart.sh script with --skip-install flag for quick restarts (ad540e4)',
          'UX improvements - navbar reorganization and placeholder cards (e63931c)',
          'Completed ZIP/URL upload and dataset navigation flow (02b3164)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Major repository cleanup - archived unused configs and organized docs (dbc2b62)',
          'Streamlined .gitignore and removed setup notes (3fb14e7)',
          'Fixed API routing, backend integration, and VastAI config (be23a7b)',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Added missing critical dependencies (torch, torchvision, numpy) (1629045)',
        ],
      },
      {
        category: 'docs',
        items: [
          'Added restart.sh to Quick Start guide (b32bfaf)',
        ],
      },
    ],
  },
  {
    version: '0.0.3-dev',
    date: 'December 8, 2025',
    type: 'minor',
    commitHash: 'a263963',
    changes: [
      {
        category: 'feature',
        items: [
          'Implemented VastAI best practices from Ostris\'s AI-Toolkit (02076be, a3ff9eb)',
          'Added template doc and config file for context preservation (52b0449)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Fixed starting scripts and readme, focused on NextJS structure (a263963)',
        ],
      },
    ],
  },
  {
    version: '0.0.2-dev',
    date: 'December 4, 2025',
    type: 'minor',
    commitHash: '8c98359',
    changes: [
      {
        category: 'feature',
        items: [
          'Added captioning service (BLIP, GIT) and checkpoint training support (8c98359)',
          'Added GitHub Actions workflow for Docker image builds (5e4d7f6)',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Modernized VastAI template scripts for vendored backend (f0cf25e)',
          'Updated Docker startup scripts to use correct VastAI paths (809c38c)',
          'Added ESLint config and type fixes for production build (5511554)',
        ],
      },
      {
        category: 'bugfix',
        items: [
          'Updated Dockerfile.vastai to match current codebase structure (6775041)',
          'Removed unused core/ directory from Dockerfile.vastai (2867089)',
          'Removed reference to non-existent shared_managers.py (6836576)',
          'Security patches and build fixes for frontend (3d199d9)',
        ],
      },
    ],
  },
  {
    version: '0.0.1-dev',
    date: 'December 1, 2025',
    type: 'major',
    commitHash: '3e6f1bb',
    changes: [
      {
        category: 'breaking',
        items: [
          'Flushed the Jupyter Toilet - complete removal of Jupyter notebook interface (3e6f1bb)',
        ],
      },
      {
        category: 'feature',
        items: [
          'Dataset editing gallery with tag management',
          'Initial web-based interface implementation',
        ],
      },
      {
        category: 'improvement',
        items: [
          'Fixed dataset editing gallery functionality (150dc35)',
          'Updated gitignore for new project structure (31b79b9)',
        ],
      },
      {
        category: 'docs',
        items: [
          'Documentation updates for web UI transition (150dc35)',
        ],
      },
    ],
  },
];

const categoryConfig = {
  feature: { icon: Sparkles, label: 'New Features', color: 'text-green-400', bg: 'bg-green-500/10' },
  improvement: { icon: Zap, label: 'Improvements', color: 'text-blue-400', bg: 'bg-blue-500/10' },
  bugfix: { icon: Bug, label: 'Bug Fixes', color: 'text-orange-400', bg: 'bg-orange-500/10' },
  breaking: { icon: Wrench, label: 'Breaking Changes', color: 'text-red-400', bg: 'bg-red-500/10' },
  docs: { icon: FileText, label: 'Documentation', color: 'text-purple-400', bg: 'bg-purple-500/10' },
};

const versionBadgeColor = {
  major: 'bg-gradient-to-r from-red-500 to-pink-500',
  minor: 'bg-gradient-to-r from-blue-500 to-cyan-500',
  patch: 'bg-gradient-to-r from-green-500 to-emerald-500',
};

export default function ChangelogPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Changelog', icon: <FileText className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-12 mt-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30">
              <GitCommit className="w-8 h-8 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Changelog
              </h1>
              <p className="text-xl text-muted-foreground mt-2">
                Track changes and updates to Ktiseos-Nyx-Trainer
              </p>
            </div>
          </div>
        </div>

        {/* Version Timeline */}
        <div className="space-y-8">
          {changelog.map((entry, idx) => (
            <Card key={entry.version} className="bg-card/50 backdrop-blur-sm border-border/50 hover:border-border transition-colors">
              <CardHeader>
                {/* Date-Focused Header */}
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <div className="flex items-center gap-3">
                    {/* Date - Primary */}
                    <Calendar className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-xl font-bold text-foreground">
                      {entry.date}
                    </h3>

                    {/* Version - Secondary */}
                    <span className="text-muted-foreground">•</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium text-white ${versionBadgeColor[entry.type]}`}>
                      {entry.version}
                    </span>
                  </div>

                  {entry.commitHash && (
                    <a
                      href={`https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/commit/${entry.commitHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
                    >
                      <Github className="w-4 h-4" />
                      <code className="font-mono">{entry.commitHash}</code>
                    </a>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-6">
                {entry.changes.map((changeGroup, groupIdx) => {
                  const config = categoryConfig[changeGroup.category];
                  const Icon = config.icon;

                  return (
                    <div key={groupIdx}>
                      {/* Category Header */}
                      <div className={`flex items-center gap-2 mb-3 px-3 py-2 rounded-lg ${config.bg}`}>
                        <Icon className={`w-4 h-4 ${config.color}`} />
                        <h3 className={`font-semibold ${config.color}`}>
                          {config.label}
                        </h3>
                      </div>

                      {/* Change Items */}
                      <ul className="space-y-2 ml-4">
                        {changeGroup.items.map((item, itemIdx) => (
                          <li key={itemIdx} className="flex items-start gap-3 text-muted-foreground">
                            <span className="text-cyan-400 mt-1.5 flex-shrink-0">•</span>
                            <span className="leading-relaxed">{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Footer Note */}
        <div className="mt-12 p-6 rounded-lg bg-card/30 border border-border/30">
          <div className="flex items-start gap-3">
            <Github className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-muted-foreground">
              <p className="mb-2">
                <strong className="text-foreground">Note:</strong> This changelog tracks major updates and features.
                For complete commit history, see the{' '}
                <a
                  href="https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/commits/main"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan-400 hover:text-cyan-300 transition-colors"
                >
                  GitHub repository
                </a>
                .
              </p>
              <p className="text-xs text-muted-foreground/70 mt-3">
                Version format: MAJOR.MINOR.PATCH-stage (following semantic versioning)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
