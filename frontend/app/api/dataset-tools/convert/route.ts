import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'
import crypto from 'crypto'
import sharp from 'sharp'
import { assertWithinBase } from '@/lib/dataset-tools/base-path'

interface ConvertResult {
  success: boolean
  converted: number
  total: number
  errors: string[]
  output_dir: string
  target_format: string
  warning?: string
}

interface ConvertJob {
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  totalFiles: number
  convertedFiles: number
  currentFile: string | null
  logs: string[]
  errors: string[]
  result: ConvertResult | null
  abortController: AbortController
}

const ALLOWED_INPUT_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.jfif', '.tiff', '.tif', '.avif', '.gif',
])

const FORMAT_EXTENSIONS: Record<string, string> = {
  webp: '.webp',
  jpg: '.jpg',
  jpeg: '.jpg',
  png: '.png',
}

const SUPPORTED_OUTPUT_FORMATS = new Set(['webp', 'jpg', 'jpeg', 'png'])

const MAX_LOGS = 500

const jobs = new Map<string, ConvertJob>()

function scheduleCleanup(jobId: string) {
  setTimeout(() => {
    jobs.delete(jobId)
  }, 5 * 60 * 1000)
}

function getSharpFormat(format: string): string {
  switch (format) {
    case 'jpg':
    case 'jpeg':
      return 'jpeg'
    case 'webp':
      return 'webp'
    case 'png':
      return 'png'
    default:
      throw new Error(`Unsupported output format: ${format}`)
  }
}

function getSharpOptions(format: string, quality: number): Record<string, unknown> {
  switch (format) {
    case 'jpg':
    case 'jpeg':
      return { quality, mozjpeg: true }
    case 'webp':
      return { quality }
    case 'png':
      return {}
    default:
      return {}
  }
}

function resolveDatasetPath(datasetName: string): string {
  const cleanName = datasetName.replace(/^datasets[/\\]/, '').trim()
  const relativePath = path.join('datasets', cleanName)
  return assertWithinBase(relativePath)
}

async function convertFile(
  srcFile: string,
  dstFile: string,
  targetFormat: string,
  sharpFormat: string,
  sharpOpts: Record<string, unknown>,
  outputMode: string,
): Promise<void> {
  const pipeline = sharp(srcFile)

  if (targetFormat === 'jpg' || targetFormat === 'jpeg') {
    pipeline.flatten({ background: { r: 255, g: 255, b: 255 } })
  }

  pipeline.toFormat(sharpFormat as unknown as sharp.AvailableFormatInfo, sharpOpts)

  if (outputMode === 'in-place') {
    const tmpFile = dstFile + '.tmp_convert'
    await pipeline.toFile(tmpFile)
    await fs.rename(tmpFile, dstFile).catch(async () => {
      await fs.copyFile(tmpFile, dstFile)
      await fs.unlink(tmpFile)
    })
    const srcExt = path.extname(srcFile).toLowerCase()
    const srcStem = path.parse(srcFile).name
    const origFile = path.join(path.dirname(srcFile), srcStem + srcExt)
    if (origFile !== dstFile) {
      await fs.unlink(origFile).catch(() => {})
    }
  } else {
    await pipeline.toFile(dstFile)
  }
}

async function runConversion(
  job: ConvertJob,
  datasetPath: string,
  outputDir: string,
  imageFiles: string[],
  targetFormat: string,
  targetExt: string,
  quality: number,
  outputMode: string,
) {
  const total = imageFiles.length
  job.totalFiles = total
  const addLog = (msg: string) => {
    job.logs.push(msg)
    if (job.logs.length > MAX_LOGS) {
      job.logs = job.logs.slice(-MAX_LOGS)
    }
  }

  const sharpFormat = getSharpFormat(targetFormat)
  const sharpOpts = getSharpOptions(targetFormat, quality)

  for (let i = 0; i < total; i++) {
    if (job.abortController.signal.aborted) {
      job.status = 'cancelled'
      job.result = {
        success: false,
        converted: job.convertedFiles,
        total,
        errors: job.errors,
        output_dir: outputDir,
        target_format: targetFormat,
      }
      return
    }

    const srcFile = imageFiles[i]
    const dstStem = path.parse(srcFile).name
    const dstFile = path.join(outputDir, dstStem + targetExt)

    try {
      if (outputMode === 'in-place') {
        const srcExt = path.extname(srcFile).toLowerCase()
        if (srcExt === targetExt) {
          job.convertedFiles++
          job.progress = Math.round((job.convertedFiles / total) * 100)
          continue
        }
      }

      job.currentFile = path.basename(srcFile)

      await convertFile(srcFile, dstFile, targetFormat, sharpFormat, sharpOpts, outputMode)

      job.convertedFiles++
      job.progress = Math.round((job.convertedFiles / total) * 100)
      addLog(`Converted ${path.basename(srcFile)} -> ${path.basename(dstFile)}`)
    } catch (err) {
      const errorMsg = `${path.basename(srcFile)}: ${err instanceof Error ? err.message : String(err)}`
      job.errors.push(errorMsg)
      addLog(`ERROR: ${errorMsg}`)
    }
  }

  if (outputMode === 'new_dataset' && outputDir !== datasetPath) {
    const entries = await fs.readdir(datasetPath, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase()
        if (!ALLOWED_INPUT_EXTENSIONS.has(ext)) {
          try {
            await fs.copyFile(path.join(datasetPath, entry.name), path.join(outputDir, entry.name))
          } catch (err) {
            addLog(`WARN: Failed to copy ${entry.name}: ${err instanceof Error ? err.message : err}`)
          }
        }
      }
    }
  }

  job.status = 'completed'
  job.currentFile = null
  job.result = {
    success: job.errors.length === 0 || job.convertedFiles > 0,
    converted: job.convertedFiles,
    total,
    errors: job.errors,
    output_dir: outputDir,
    target_format: targetFormat,
  }
  if (job.errors.length > 0) {
    job.result.warning = `${job.errors.length} files failed to convert`
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { dataset_dir, target_format, quality, output_mode } = body

    if (!dataset_dir || !target_format) {
      return NextResponse.json(
        { success: false, message: 'dataset_dir and target_format are required', total_files: 0 },
        { status: 400 },
      )
    }

    if (!SUPPORTED_OUTPUT_FORMATS.has(target_format)) {
      return NextResponse.json(
        {
          success: false,
          message: `Unsupported target format: ${target_format}. Sharp supports: webp, jpg, png (BMP output was removed — it's uncompressed and rarely used for training data)`,
          total_files: 0,
        },
        { status: 400 },
      )
    }

    const q = typeof quality === 'number' ? Math.max(1, Math.min(100, Math.round(quality))) : 90
    const mode = output_mode === 'in-place' ? 'in-place' : 'new_dataset'

    let datasetPath: string
    try {
      datasetPath = resolveDatasetPath(dataset_dir)
    } catch (err) {
      return NextResponse.json(
        { success: false, message: `Invalid dataset path: ${err instanceof Error ? err.message : err}`, total_files: 0 },
        { status: 400 },
      )
    }

    let entries: string[]
    try {
      entries = await fs.readdir(datasetPath)
    } catch {
      return NextResponse.json(
        { success: false, message: `Dataset not found: ${dataset_dir}`, total_files: 0 },
        { status: 404 },
      )
    }

    const imageFiles = entries
      .filter(f => ALLOWED_INPUT_EXTENSIONS.has(path.extname(f).toLowerCase()))
      .map(f => path.join(datasetPath, f))
      .sort()

    if (imageFiles.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No image files found in dataset', total_files: 0 },
        { status: 400 },
      )
    }

    const targetExt = FORMAT_EXTENSIONS[target_format]
    const outputDir = mode === 'new_dataset'
      ? path.join(path.dirname(datasetPath), `${path.basename(datasetPath)}_${target_format}`)
      : datasetPath

    if (mode === 'new_dataset') {
      await fs.mkdir(outputDir, { recursive: true })
    }

    const jobId = crypto.randomUUID()
    const abortController = new AbortController()
    const job: ConvertJob = {
      status: 'running',
      progress: 0,
      totalFiles: imageFiles.length,
      convertedFiles: 0,
      currentFile: null,
      logs: [],
      errors: [],
      result: null,
      abortController,
    }
    jobs.set(jobId, job)

    runConversion(job, datasetPath, outputDir, imageFiles, target_format, targetExt, q, mode)
      .catch(err => {
        job.status = 'failed'
        job.errors.push(`Fatal: ${err instanceof Error ? err.message : String(err)}`)
      })
      .finally(() => {
        scheduleCleanup(jobId)
      })

    return NextResponse.json({
      success: true,
      job_id: jobId,
      message: `Converting ${imageFiles.length} images to .${target_format}`,
      total_files: imageFiles.length,
    })
  } catch (err) {
    return NextResponse.json(
      { success: false, message: `Invalid request: ${err instanceof Error ? err.message : String(err)}`, total_files: 0 },
      { status: 400 },
    )
  }
}

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId')
  if (!jobId) {
    return NextResponse.json({ error: 'jobId query parameter is required' }, { status: 400 })
  }

  const job = jobs.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  return NextResponse.json({
    job_id: jobId,
    status: job.status,
    progress: job.progress,
    total_files: job.totalFiles,
    converted_files: job.convertedFiles,
    current_file: job.currentFile,
    logs: job.logs,
    errors: job.errors,
    result: job.result,
  })
}

export async function DELETE(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId')
  if (!jobId) {
    return NextResponse.json({ error: 'jobId query parameter is required' }, { status: 400 })
  }

  const job = jobs.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  if (job.status === 'running') {
    job.abortController.abort()
    job.status = 'cancelled'
  }

  return NextResponse.json({ success: true })
}
