import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'
import crypto from 'crypto'
import sharp from 'sharp'
import { assertWithinBase } from '@/lib/dataset-tools/base-path'

interface CropRegion {
  filename: string
  source_x: number
  source_y: number
  source_width: number
  source_height: number
}

interface CropJob {
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  totalFiles: number
  croppedFiles: number
  currentFile: string | null
  errors: string[]
  result: CropResult | null
  abortController: AbortController
}

interface CropResult {
  success: boolean
  cropped: number
  total: number
  errors: string[]
  output_dir: string
  target_size: string
  warning?: string
}

const ALLOWED_INPUT_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.jfif',
])

const FORMAT_EXTENSIONS: Record<string, string> = {
  webp: '.webp',
  jpg: '.jpg',
  png: '.png',
}

const SUPPORTED_OUTPUT_FORMATS = new Set(['webp', 'jpg', 'png'])

const MAX_LOGS = 500

const jobs = new Map<string, CropJob>()

function scheduleCleanup(jobId: string) {
  setTimeout(() => jobs.delete(jobId), 5 * 60 * 1000)
}

function getSharpFormat(format: string): string {
  switch (format) {
    case 'jpg': return 'jpeg'
    case 'webp': return 'webp'
    case 'png': return 'png'
    default: throw new Error(`Unsupported output format: ${format}`)
  }
}

function getSharpOptions(format: string, quality: number): Record<string, unknown> {
  switch (format) {
    case 'jpg': return { quality, mozjpeg: true }
    case 'webp': return { quality }
    case 'png': return {}
    default: return {}
  }
}

function resolveDatasetPath(datasetName: string): string {
  const cleanName = datasetName.replace(/^datasets[/\\]/, '').trim()
  return assertWithinBase(path.join('datasets', cleanName))
}

async function cropSingleFile(
  srcFile: string,
  dstFile: string,
  region: { x: number; y: number; width: number; height: number },
  targetWidth: number,
  targetHeight: number,
  format: string,
  sharpFormat: string,
  sharpOpts: Record<string, unknown>,
  outputMode: string,
): Promise<void> {
  const metadata = await sharp(srcFile).metadata()
  const srcW = metadata.width ?? 0
  const srcH = metadata.height ?? 0

  const sx = Math.max(0, Math.min(Math.round(region.x), srcW))
  const sy = Math.max(0, Math.min(Math.round(region.y), srcH))
  const sw = Math.max(1, Math.min(Math.round(region.width), srcW - sx))
  const sh = Math.max(1, Math.min(Math.round(region.height), srcH - sy))

  const pipeline = sharp(srcFile)
    .extract({ left: sx, top: sy, width: sw, height: sh })
    .resize(targetWidth, targetHeight, { kernel: 'cubic' })

  if (format === 'jpg' || format === 'jpeg') {
    pipeline.flatten({ background: { r: 255, g: 255, b: 255 } })
  }

  pipeline.toFormat(sharpFormat as unknown as sharp.AvailableFormatInfo, sharpOpts)

  if (outputMode === 'in-place') {
    const tmpFile = dstFile + '.tmp_crop'
    await pipeline.toFile(tmpFile)
    await fs.rename(tmpFile, dstFile).catch(async () => {
      await fs.copyFile(tmpFile, dstFile)
      await fs.unlink(tmpFile)
    })
    if (srcFile !== dstFile) {
      await fs.unlink(srcFile).catch(() => {})
    }
  } else {
    await pipeline.toFile(dstFile)
  }
}

async function runCrop(
  job: CropJob,
  datasetPath: string,
  outputDir: string,
  crops: CropRegion[],
  targetWidth: number,
  targetHeight: number,
  targetFormat: string,
  targetExt: string,
  quality: number,
  outputMode: string,
) {
  const total = crops.length
  job.totalFiles = total
  const addLog = (msg: string) => {
    job.errors.push(msg)
    if (job.errors.length > MAX_LOGS) {
      job.errors = job.errors.slice(-MAX_LOGS)
    }
  }

  const sharpFormat = getSharpFormat(targetFormat)
  const sharpOpts = getSharpOptions(targetFormat, quality)

  for (let i = 0; i < total; i++) {
    if (job.abortController.signal.aborted) {
      job.status = 'cancelled'
      job.result = {
        success: false,
        cropped: job.croppedFiles,
        total,
        errors: job.errors,
        output_dir: outputDir,
        target_size: `${targetWidth}x${targetHeight}`,
      }
      return
    }

    const crop = crops[i]
    const safeName = path.basename(crop.filename)
    const srcFile = path.resolve(datasetPath, safeName)

    if (!srcFile.startsWith(datasetPath + path.sep) && srcFile !== datasetPath) {
      job.errors.push(`${crop.filename}: access denied`)
      addLog(`SKIP: ${crop.filename}: access denied`)
      continue
    }

    if (!ALLOWED_INPUT_EXTENSIONS.has(path.extname(safeName).toLowerCase())) {
      job.errors.push(`${crop.filename}: file type not allowed`)
      addLog(`SKIP: ${crop.filename}: file type not allowed`)
      continue
    }

    const dstStem = path.parse(safeName).name
    const dstFile = path.join(outputDir, dstStem + targetExt)

    try {
      await fs.access(srcFile)
    } catch {
      job.errors.push(`${crop.filename}: file not found`)
      addLog(`SKIP: ${crop.filename}: file not found`)
      continue
    }

    try {
      job.currentFile = safeName

      await cropSingleFile(
        srcFile, dstFile,
        { x: crop.source_x, y: crop.source_y, width: crop.source_width, height: crop.source_height },
        targetWidth, targetHeight,
        targetFormat, sharpFormat, sharpOpts,
        outputMode,
      )

      job.croppedFiles++
      job.progress = Math.round((job.croppedFiles / total) * 100)

      if (outputMode === 'in-place' && srcFile !== dstFile) {
        addLog(`Cropped ${safeName} (-> ${dstStem}${targetExt}, original deleted)`)
      } else {
        addLog(`Cropped ${safeName} -> ${dstStem}${targetExt}`)
      }
    } catch (err) {
      const errorMsg = `${crop.filename}: ${err instanceof Error ? err.message : String(err)}`
      job.errors.push(errorMsg)
      addLog(`ERROR: ${errorMsg}`)
    }
  }

  job.status = 'completed'
  job.currentFile = null
  job.result = {
    success: job.errors.length === 0 || job.croppedFiles > 0,
    cropped: job.croppedFiles,
    total,
    errors: job.errors,
    output_dir: outputDir,
    target_size: `${targetWidth}x${targetHeight}`,
  }
  if (job.errors.length > 0) {
    job.result.warning = `${job.errors.length} files failed to crop`
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const {
      dataset_dir,
      target_width,
      target_height,
      output_format,
      quality,
      output_mode,
      crops,
    } = body

    if (!dataset_dir || !crops || !Array.isArray(crops) || crops.length === 0) {
      return NextResponse.json(
        { success: false, message: 'dataset_dir and crops array are required', total_files: 0 },
        { status: 400 },
      )
    }

    const format = SUPPORTED_OUTPUT_FORMATS.has(output_format) ? output_format : 'webp'
    const q = typeof quality === 'number' ? Math.max(1, Math.min(100, Math.round(quality))) : 90
    const mode = output_mode === 'in-place' ? 'in-place' : 'new_dataset'
    const tw = Math.max(1, Math.round(target_width) || 512)
    const th = Math.max(1, Math.round(target_height) || 512)

    let datasetPath: string
    try {
      datasetPath = resolveDatasetPath(dataset_dir)
    } catch (err) {
      return NextResponse.json(
        { success: false, message: `Invalid dataset path: ${err instanceof Error ? err.message : err}`, total_files: 0 },
        { status: 400 },
      )
    }

    try {
      await fs.access(datasetPath)
    } catch {
      return NextResponse.json(
        { success: false, message: `Dataset not found: ${dataset_dir}`, total_files: 0 },
        { status: 404 },
      )
    }

    const targetExt = FORMAT_EXTENSIONS[format]
    const outputDir = mode === 'new_dataset'
      ? path.join(path.dirname(datasetPath), `${path.basename(datasetPath)}_${tw}x${th}`)
      : datasetPath

    if (mode === 'new_dataset') {
      await fs.mkdir(outputDir, { recursive: true })
    }

    const jobId = crypto.randomUUID()
    const abortController = new AbortController()
    const job: CropJob = {
      status: 'running',
      progress: 0,
      totalFiles: crops.length,
      croppedFiles: 0,
      currentFile: null,
      errors: [],
      result: null,
      abortController,
    }
    jobs.set(jobId, job)

    runCrop(job, datasetPath, outputDir, crops, tw, th, format, targetExt, q, mode)
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
      message: `Cropping ${crops.length} images to ${tw}x${th}`,
      total_files: crops.length,
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
    cropped_files: job.croppedFiles,
    current_file: job.currentFile,
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
