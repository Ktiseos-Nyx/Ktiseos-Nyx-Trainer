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

// ========== Job Manager ==========

class JobManager {
  private jobs: Map<string, Job> = new Map();
  private processes: Map<string, ChildProcess> = new Map();
  public events: JobEventEmitter = new EventEmitter() as JobEventEmitter;
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
 * Create and start a tagging job
 */
export async function createTaggingJob(
  inputDir: string,
  model: string = 'wd-v1-4-moat-tagger-v2'
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  const scriptPath = path.join(
    projectRoot,
    'custom',
    'tag_images_by_wd14_tagger.py'
  );

  const args = [
    scriptPath,
    inputDir,
    '--model',
    model,
    '--general_threshold',
    '0.35',
    '--character_threshold',
    '0.85',
  ];

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
