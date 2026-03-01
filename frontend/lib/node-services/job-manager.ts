/**
 * Job Manager Service - Node.js job execution and log streaming
 * Migrated from Python services/jobs/
 *
 * Handles:
 * - Spawning Python ML processes (tagging, captioning, training)
 * - Real-time log streaming via stdout/stderr
 * - Job state tracking (in-memory)
 * - Job control (start, stop, status)
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';

// ========== Types ==========

export type JobType = 'training' | 'tagging' | 'captioning_blip' | 'captioning_git' | 'lora_resize' | 'hf_upload';

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  command: string;
  args: string[];
  cwd?: string;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  exit_code?: number;
  error?: string;
  progress?: number; // 0-100
  logs: LogEntry[];
}

export interface LogEntry {
  timestamp: number;
  level: 'info' | 'error' | 'debug' | 'progress';
  message: string;
  raw?: string; // Original stdout/stderr line
}

export interface JobEventEmitter extends EventEmitter {
  on(event: 'log', listener: (jobId: string, log: LogEntry) => void): this;
  on(event: 'status', listener: (jobId: string, status: JobStatus) => void): this;
  on(event: 'progress', listener: (jobId: string, progress: number) => void): this;
  on(event: 'complete', listener: (jobId: string, exitCode: number) => void): this;
  on(event: 'error', listener: (jobId: string, error: string) => void): this;
  emit(event: 'log', jobId: string, log: LogEntry): boolean;
  emit(event: 'status', jobId: string, status: JobStatus): boolean;
  emit(event: 'progress', jobId: string, progress: number): boolean;
  emit(event: 'complete', jobId: string, exitCode: number): boolean;
  emit(event: 'error', jobId: string, error: string): boolean;
}

// ========== Configuration ==========

/** Max log entries per job. Oldest entries are trimmed when exceeded. */
const MAX_LOGS = parseInt(process.env.MAX_LOG_ENTRIES || '500', 10);

/** How often to run auto-cleanup of old completed/failed jobs (ms) */
const CLEANUP_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes

// ========== Shared State ==========
// Uses globalThis to ensure server.js (plain JS) and all Next.js API routes
// share the same job state regardless of how Next.js bundles modules.

if (!globalThis.__jobEvents) {
  const events = new EventEmitter();
  events.setMaxListeners(100);
  // IMPORTANT: 'error' events on EventEmitter throw if no listener is attached.
  // Add a default handler so job errors don't crash the process.
  events.on('error', (jobId: string, errorMsg: string) => {
    console.error(`[JobManager] Job ${jobId} error: ${errorMsg}`);
  });
  globalThis.__jobEvents = events;
}
if (!globalThis.__jobsMap) {
  globalThis.__jobsMap = new Map();
}

// ========== Job Manager ==========

class JobManager {
  private jobs: Map<string, Job> = globalThis.__jobsMap;
  private processes: Map<string, ChildProcess> = new Map();
  public events: JobEventEmitter = globalThis.__jobEvents as JobEventEmitter;
  private jobCounter = 0;

  /**
   * Generate unique job ID
   */
  private generateJobId(type: JobType): string {
    this.jobCounter++;
    const timestamp = Date.now();
    return `${type}_${timestamp}_${this.jobCounter}`;
  }

  /**
   * Create a new job
   */
  createJob(
    type: JobType,
    command: string,
    args: string[],
    cwd?: string
  ): string {
    const jobId = this.generateJobId(type);

    const job: Job = {
      id: jobId,
      type,
      status: 'pending',
      command,
      args,
      cwd,
      created_at: Date.now(),
      logs: [],
    };

    this.jobs.set(jobId, job);
    return jobId;
  }

  /**
   * Start a job (spawn child process)
   */
  startJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) {
      console.error(`Job ${jobId} not found`);
      return false;
    }

    if (job.status !== 'pending') {
      console.error(`Job ${jobId} is not pending (status: ${job.status})`);
      return false;
    }

    try {
      // Spawn child process
      const proc = spawn(job.command, job.args, {
        cwd: job.cwd,
        env: { ...process.env },
        shell: true,
      });

      this.processes.set(jobId, proc);

      // Update job status
      job.status = 'running';
      job.started_at = Date.now();
      this.events.emit('status', jobId, 'running');

      // Handle stdout
      proc.stdout?.on('data', (data) => {
        const lines = data.toString().split('\n').filter((l: string) => l.trim());
        lines.forEach((line: string) => {
          const logEntry = this.parseLogLine(line);
          job.logs.push(logEntry);
          // Cap log array to prevent unbounded memory growth
          if (job.logs.length > MAX_LOGS) {
            job.logs.splice(0, job.logs.length - MAX_LOGS);
          }
          this.events.emit('log', jobId, logEntry);

          // Check for progress updates
          const progressMatch = line.match(/(\d+)%/);
          if (progressMatch) {
            const progress = parseInt(progressMatch[1], 10);
            job.progress = progress;
            this.events.emit('progress', jobId, progress);
          }
        });
      });

      // Handle stderr
      proc.stderr?.on('data', (data) => {
        const lines = data.toString().split('\n').filter((l: string) => l.trim());
        lines.forEach((line: string) => {
          const logEntry = this.parseLogLine(line, 'error');
          job.logs.push(logEntry);
          if (job.logs.length > MAX_LOGS) {
            job.logs.splice(0, job.logs.length - MAX_LOGS);
          }
          this.events.emit('log', jobId, logEntry);
        });
      });

      // Handle process exit
      proc.on('exit', (code, signal) => {
        job.completed_at = Date.now();
        job.exit_code = code ?? undefined;

        if (code === 0) {
          job.status = 'completed';
          job.progress = 100;
          this.events.emit('status', jobId, 'completed');
          this.events.emit('complete', jobId, code);
        } else if (signal === 'SIGTERM' || signal === 'SIGKILL') {
          job.status = 'cancelled';
          this.events.emit('status', jobId, 'cancelled');
        } else {
          job.status = 'failed';
          job.error = `Process exited with code ${code}`;
          this.events.emit('status', jobId, 'failed');
          this.events.emit('error', jobId, job.error);
        }

        // Cleanup
        this.processes.delete(jobId);
      });

      // Handle process error
      proc.on('error', (err) => {
        job.status = 'failed';
        job.error = err.message;
        job.completed_at = Date.now();
        this.events.emit('status', jobId, 'failed');
        this.events.emit('error', jobId, err.message);
        this.processes.delete(jobId);
      });

      return true;
    } catch (error) {
      job.status = 'failed';
      job.error = error instanceof Error ? error.message : String(error);
      job.completed_at = Date.now();
      this.events.emit('status', jobId, 'failed');
      this.events.emit('error', jobId, job.error);
      return false;
    }
  }

  /**
   * Stop a running job
   */
  stopJob(jobId: string): boolean {
    const proc = this.processes.get(jobId);
    if (!proc) {
      return false;
    }

    try {
      proc.kill('SIGTERM');
      return true;
    } catch (error) {
      console.error(`Failed to stop job ${jobId}:`, error);
      return false;
    }
  }

  /**
   * Get job by ID
   */
  getJob(jobId: string): Job | undefined {
    return this.jobs.get(jobId);
  }

  /**
   * Get all jobs
   */
  getAllJobs(): Job[] {
    return Array.from(this.jobs.values());
  }

  /**
   * Get jobs by status
   */
  getJobsByStatus(status: JobStatus): Job[] {
    return Array.from(this.jobs.values()).filter((j) => j.status === status);
  }

  /**
   * Get jobs by type
   */
  getJobsByType(type: JobType): Job[] {
    return Array.from(this.jobs.values()).filter((j) => j.type === type);
  }

  /**
   * Delete a job (only if not running)
   */
  deleteJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) {
      return false;
    }

    if (job.status === 'running') {
      console.error(`Cannot delete running job ${jobId}`);
      return false;
    }

    this.jobs.delete(jobId);
    return true;
  }

  /**
   * Parse log line and extract level/message
   */
  private parseLogLine(line: string, defaultLevel: 'info' | 'error' = 'info'): LogEntry {
    const timestamp = Date.now();

    // Try to extract log level from common formats
    const levelMatch = line.match(/\[(ERROR|WARN|INFO|DEBUG)\]/i);
    let level: LogEntry['level'] = defaultLevel;
    let message = line;

    if (levelMatch) {
      const matchedLevel = levelMatch[1].toLowerCase();
      if (matchedLevel === 'error' || matchedLevel === 'warn') {
        level = 'error';
      } else if (matchedLevel === 'debug') {
        level = 'debug';
      } else {
        level = 'info';
      }
      // Remove level prefix from message
      message = line.replace(/\[(ERROR|WARN|INFO|DEBUG)\]\s*/i, '');
    }

    // Check if it's a progress line
    if (line.match(/\d+%/) || line.match(/epoch|step|batch/i)) {
      level = 'progress';
    }

    return {
      timestamp,
      level,
      message: message.trim(),
      raw: line,
    };
  }

  /**
   * Clean up old completed/failed jobs (keep last N)
   */
  cleanup(keepLast: number = 100): number {
    const allJobs = Array.from(this.jobs.values())
      .filter((j) => j.status === 'completed' || j.status === 'failed')
      .sort((a, b) => (b.completed_at || 0) - (a.completed_at || 0));

    const toDelete = allJobs.slice(keepLast);
    toDelete.forEach((job) => this.jobs.delete(job.id));

    return toDelete.length;
  }
}

// ========== Singleton Instance ==========

export const jobManager = new JobManager();

// ========== Auto-Cleanup ==========
// Periodically remove old completed/failed jobs to prevent unbounded memory growth.
// Uses globalThis guard so only one timer runs per process.

if (!globalThis.__jobCleanupTimer) {
  globalThis.__jobCleanupTimer = setInterval(() => {
    const removed = jobManager.cleanup(50);
    if (removed > 0) {
      console.log(`[JobManager] Auto-cleanup: removed ${removed} old jobs`);
    }
  }, CLEANUP_INTERVAL_MS);
  // Don't block process exit
  if (globalThis.__jobCleanupTimer.unref) {
    globalThis.__jobCleanupTimer.unref();
  }
}

// ========== Helper Functions ==========

/**
 * Get project root directory
 */
export function getProjectRoot(): string {
  // From frontend directory, go up one level
  return path.resolve(process.cwd(), '..');
}

/**
 * Get Python executable path
 * Tries to use venv python if available, falls back to system python
 */
export function getPythonPath(): string {
  const projectRoot = getProjectRoot();

  // Try venv paths (Windows and Linux)
  const venvPaths = [
    path.join(projectRoot, 'venv', 'Scripts', 'python.exe'), // Windows
    path.join(projectRoot, 'venv', 'bin', 'python'), // Linux
  ];

  // For now, just use 'python' and let the system resolve it
  // In production, you'd check if venv exists
  return 'python';
}

/**
 * Create and start a training job
 */
export async function createTrainingJob(
  configPath: string,
  datasetPath: string
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  // Path to training script
  const scriptPath = path.join(
    projectRoot,
    'trainer',
    'derrian_backend',
    'sd_scripts',
    'sdxl_train_network.py'
  );

  const args = [
    scriptPath,
    '--config_file',
    configPath,
    '--dataset_config',
    datasetPath,
  ];

  const jobId = jobManager.createJob('training', pythonPath, args, projectRoot);
  jobManager.startJob(jobId);

  return jobId;
}

/**
 * Configuration for WD14 tagger jobs.
 * Fields map to Python CLI flags in custom/tag_images_by_wd14_tagger.py
 */
export interface TaggingJobConfig {
  /** Directory containing images to tag (positional: train_data_dir) */
  inputDir: string;
  /** HuggingFace repo ID for the tagger model (--repo_id) */
  model?: string;
  /** Global confidence threshold for adding tags (--thresh) */
  threshold?: number;
  /** Threshold for general category tags (--general_threshold) */
  generalThreshold?: number;
  /** Threshold for character category tags (--character_threshold) */
  characterThreshold?: number;
  /** Extension for output caption files, e.g. ".txt" (--caption_extension) */
  captionExtension?: string;
  /** Batch size for inference (--batch_size) */
  batchSize?: number;
  /** Separator between tags in caption files (--caption_separator) */
  captionSeparator?: string;
  /** Comma-separated list of tags to exclude (--undesired_tags) */
  undesiredTags?: string;
  /** Comma-separated list of tags to always place first (--always_first_tags) */
  alwaysFirstTags?: string;
  /** Tag replacement rules: "source1,target1;source2,target2" (--tag_replacement) */
  tagReplacement?: string;
  /** Replace underscores with spaces in output tags (--remove_underscore) */
  removeUnderscore?: boolean;
  /** Place character tags before general tags (--character_tags_first) */
  characterTagsFirst?: boolean;
  /** Append to existing caption files instead of overwriting (--append_tags) */
  appendTags?: boolean;
  /** Search for images in subdirectories recursively (--recursive) */
  recursive?: boolean;
  /** Add rating tags as first tag (--use_rating_tags) */
  useRatingTags?: boolean;
  /** Add rating tags as last tag (--use_rating_tags_as_last_tag) */
  useRatingTagsAsLastTag?: boolean;
  /** Expand character tag parentheses to separate tags (--character_tag_expand) */
  characterTagExpand?: boolean;
}

/**
 * Create and start a tagging job using the Python WD14 tagger.
 * Always passes --onnx for GPU-accelerated inference.
 */
export async function createTaggingJob(
  config: TaggingJobConfig
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  const scriptPath = path.join(
    projectRoot,
    'custom',
    'tag_images_by_wd14_tagger.py'
  );

  // Build CLI args: script path, then positional train_data_dir
  const args: string[] = [scriptPath, config.inputDir];

  // Always enable ONNX for GPU support
  args.push('--onnx');

  // Model repo ID
  if (config.model) {
    args.push('--repo_id', config.model);
  }

  // Numeric / string options
  if (config.threshold !== undefined) {
    args.push('--thresh', String(config.threshold));
  }
  if (config.generalThreshold !== undefined) {
    args.push('--general_threshold', String(config.generalThreshold));
  }
  if (config.characterThreshold !== undefined) {
    args.push('--character_threshold', String(config.characterThreshold));
  }
  if (config.captionExtension) {
    args.push('--caption_extension', config.captionExtension);
  }
  if (config.batchSize !== undefined) {
    args.push('--batch_size', String(config.batchSize));
  }
  if (config.captionSeparator !== undefined) {
    args.push('--caption_separator', config.captionSeparator);
  }
  if (config.undesiredTags) {
    args.push('--undesired_tags', config.undesiredTags);
  }
  if (config.alwaysFirstTags) {
    args.push('--always_first_tags', config.alwaysFirstTags);
  }
  if (config.tagReplacement) {
    args.push('--tag_replacement', config.tagReplacement);
  }

  // Boolean flags (only pass when true)
  if (config.removeUnderscore) {
    args.push('--remove_underscore');
  }
  if (config.characterTagsFirst) {
    args.push('--character_tags_first');
  }
  if (config.appendTags) {
    args.push('--append_tags');
  }
  if (config.recursive) {
    args.push('--recursive');
  }
  if (config.useRatingTags) {
    args.push('--use_rating_tags');
  }
  if (config.useRatingTagsAsLastTag) {
    args.push('--use_rating_tags_as_last_tag');
  }
  if (config.characterTagExpand) {
    args.push('--character_tag_expand');
  }

  const jobId = jobManager.createJob('tagging', pythonPath, args, projectRoot);
  jobManager.startJob(jobId);

  return jobId;
}

/**
 * Create and start a BLIP captioning job
 */
export async function createBlipCaptioningJob(
  inputDir: string,
  captionExtension: string = '.txt'
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  const scriptPath = path.join(
    projectRoot,
    'trainer',
    'derrian_backend',
    'sd_scripts',
    'finetune',
    'make_captions.py'
  );

  const args = [
    scriptPath,
    inputDir,
    '--batch_size',
    '1',
    '--caption_extension',
    captionExtension,
  ];

  const jobId = jobManager.createJob('captioning_blip', pythonPath, args, projectRoot);
  jobManager.startJob(jobId);

  return jobId;
}

/**
 * Create and start a GIT captioning job
 */
export async function createGitCaptioningJob(
  inputDir: string,
  captionExtension: string = '.txt'
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  const scriptPath = path.join(
    projectRoot,
    'trainer',
    'derrian_backend',
    'sd_scripts',
    'finetune',
    'make_captions_by_git.py'
  );

  const args = [
    scriptPath,
    inputDir,
    '--batch_size',
    '1',
    '--caption_extension',
    captionExtension,
  ];

  const jobId = jobManager.createJob('captioning_git', pythonPath, args, projectRoot);
  jobManager.startJob(jobId);

  return jobId;
}
