"use client";

/**
 * @author: @dorian_baffier
 * @description: File Upload
 * @version: 1.0.0
 * @date: 2025-06-26
 * @license: MIT
 * @website: https://kokonutui.com
 * @github: https://github.com/kokonut-labs/kokonutui
 */

import {
    useState,
    useRef,
    useCallback,
    type DragEvent,
    useEffect,
} from "react";
import { motion, AnimatePresence } from "motion/react";
import { UploadCloud, File as FileIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type FileStatus = "idle" | "dragging" | "uploading" | "error";

interface FileError {
    message: string;
    code: string;
}

interface FileUploadProps {
    onUploadSuccess?: (file: File) => void;
    onUploadError?: (error: FileError) => void;
    acceptedFileTypes?: string[];
    maxFileSize?: number;
    currentFile?: File | null;
    onFileRemove?: () => void;
    /** Duration in milliseconds for the upload simulation. Defaults to 2000ms (2s), 0 for no simulation */
    uploadDelay?: number;
    validateFile?: (file: File) => FileError | null;
    className?: string;
}

const DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const UPLOAD_STEP_SIZE = 5;
const FILE_SIZES = [
    "Bytes",
    "KB",
    "MB",
    "GB",
    "TB",
    "PB",
    "EB",
    "ZB",
    "YB",
] as const;

const formatBytes = (bytes: number, decimals = 2): string => {
    if (!+bytes) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const unit = FILE_SIZES[i] || FILE_SIZES[FILE_SIZES.length - 1];
    return `${Number.parseFloat((bytes / k ** i).toFixed(dm))} ${unit}`;
};

const UploadIllustration = () => (
    <div className="relative w-16 h-16">
        <svg
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="w-full h-full"
            aria-label="Upload illustration"
        >
            <title>Upload File Illustration</title>
            <circle
                cx="50"
                cy="50"
                r="45"
                className="stroke-gray-200 dark:stroke-gray-700"
                strokeWidth="2"
                strokeDasharray="4 4"
            >
                <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from="0 50 50"
                    to="360 50 50"
                    dur="60s"
                    repeatCount="indefinite"
                />
            </circle>

            <path
                d="M30 35H70C75 35 75 40 75 40V65C75 70 70 70 70 70H30C25 70 25 65 25 65V40C25 35 30 35 30 35Z"
                className="fill-blue-100 dark:fill-blue-900/30 stroke-blue-500 dark:stroke-blue-400"
                strokeWidth="2"
            >
                <animate
                    attributeName="d"
                    dur="2s"
                    repeatCount="indefinite"
                    values="
                        M30 35H70C75 35 75 40 75 40V65C75 70 70 70 70 70H30C25 70 25 65 25 65V40C25 35 30 35 30 35Z;
                        M30 38H70C75 38 75 43 75 43V68C75 73 70 73 70 73H30C25 73 25 68 25 68V43C25 38 30 38 30 38Z;
                        M30 35H70C75 35 75 40 75 40V65C75 70 70 70 70 70H30C25 70 25 65 25 65V40C25 35 30 35 30 35Z"
                />
            </path>

            <path
                d="M30 35C30 35 35 35 40 35C45 35 45 30 50 30C55 30 55 35 60 35C65 35 70 35 70 35"
                className="stroke-blue-500 dark:stroke-blue-400"
                strokeWidth="2"
                fill="none"
            />

            <g className="transform translate-y-2">
                <line
                    x1="50"
                    y1="45"
                    x2="50"
                    y2="60"
                    className="stroke-blue-500 dark:stroke-blue-400"
                    strokeWidth="2"
                    strokeLinecap="round"
                >
                    <animate
                        attributeName="y2"
                        values="60;55;60"
                        dur="2s"
                        repeatCount="indefinite"
                    />
                </line>
                <polyline
                    points="42,52 50,45 58,52"
                    className="stroke-blue-500 dark:stroke-blue-400"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                >
                    <animate
                        attributeName="points"
                        values="42,52 50,45 58,52;42,47 50,40 58,47;42,52 50,45 58,52"
                        dur="2s"
                        repeatCount="indefinite"
                    />
                </polyline>
            </g>
        </svg>
    </div>
);

const UploadingAnimation = ({ progress }: { progress: number }) => (
    <div className="relative w-16 h-16">
        <svg
            viewBox="0 0 240 240"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="w-full h-full"
            aria-label={`Upload progress: ${Math.round(progress)}%`}
        >
            <title>Upload Progress Indicator</title>

            <defs>
                <mask id="progress-mask">
                    <rect width="240" height="240" fill="black" />
                    <circle
                        r="120"
                        cx="120"
                        cy="120"
                        fill="white"
                        strokeDasharray={`${(progress / 100) * 754}, 754`}
                        transform="rotate(-90 120 120)"
                    />
                </mask>
            </defs>

            <style>
                {`
                    @keyframes rotate-cw {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                    @keyframes rotate-ccw {
                        from { transform: rotate(360deg); }
                        to { transform: rotate(0deg); }
                    }
                    .g-spin circle {
                        transform-origin: 120px 120px;
                    }
                    .g-spin circle:nth-child(1) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(2) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(3) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(4) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(5) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(6) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(7) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(8) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(9) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(10) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(11) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(12) { animation: rotate-ccw 8s linear infinite; }
                    .g-spin circle:nth-child(13) { animation: rotate-cw 8s linear infinite; }
                    .g-spin circle:nth-child(14) { animation: rotate-ccw 8s linear infinite; }

                    .g-spin circle:nth-child(2n) { animation-delay: 0.2s; }
                    .g-spin circle:nth-child(3n) { animation-delay: 0.3s; }
                    .g-spin circle:nth-child(5n) { animation-delay: 0.5s; }
                    .g-spin circle:nth-child(7n) { animation-delay: 0.7s; }
                `}
            </style>

            <g
                className="g-spin"
                strokeWidth="10"
                strokeDasharray="18% 40%"
                mask="url(#progress-mask)"
            >
                <circle
                    r="150"
                    cx="120"
                    cy="120"
                    stroke="#FF2E7E"
                    opacity="0.95"
                />
                <circle
                    r="140"
                    cx="120"
                    cy="120"
                    stroke="#FFD600"
                    opacity="0.95"
                />
                <circle
                    r="130"
                    cx="120"
                    cy="120"
                    stroke="#00E5FF"
                    opacity="0.95"
                />
                <circle
                    r="120"
                    cx="120"
                    cy="120"
                    stroke="#FF3D71"
                    opacity="0.95"
                />
                <circle
                    r="110"
                    cx="120"
                    cy="120"
                    stroke="#4ADE80"
                    opacity="0.95"
                />
                <circle
                    r="100"
                    cx="120"
                    cy="120"
                    stroke="#2196F3"
                    opacity="0.95"
                />
                <circle
                    r="90"
                    cx="120"
                    cy="120"
                    stroke="#FFA726"
                    opacity="0.95"
                />
                <circle
                    r="80"
                    cx="120"
                    cy="120"
                    stroke="#FF1493"
                    opacity="0.95"
                />
                <circle
                    r="70"
                    cx="120"
                    cy="120"
                    stroke="#FFEB3B"
                    opacity="0.95"
                />
                <circle
                    r="60"
                    cx="120"
                    cy="120"
                    stroke="#00BCD4"
                    opacity="0.95"
                />
                <circle
                    r="50"
                    cx="120"
                    cy="120"
                    stroke="#FF4081"
                    opacity="0.95"
                />
                <circle
                    r="40"
                    cx="120"
                    cy="120"
                    stroke="#76FF03"
                    opacity="0.95"
                />
                <circle
                    r="30"
                    cx="120"
                    cy="120"
                    stroke="#448AFF"
                    opacity="0.95"
                />
                <circle
                    r="20"
                    cx="120"
                    cy="120"
                    stroke="#FF3D00"
                    opacity="0.95"
                />
            </g>
        </svg>
    </div>
);

export default function FileUpload({
    onUploadSuccess = () => {},
    onUploadError = () => {},
    acceptedFileTypes = [],
    maxFileSize = DEFAULT_MAX_FILE_SIZE,
    currentFile: initialFile = null,
    onFileRemove = () => {},
    uploadDelay = 2000,
    validateFile = () => null,
    className,
}: FileUploadProps) {
    const [file, setFile] = useState<File | null>(initialFile);
    const [status, setStatus] = useState<FileStatus>("idle");
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<FileError | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const uploadIntervalRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        return () => {
            if (uploadIntervalRef.current) {
                clearInterval(uploadIntervalRef.current);
            }
        };
    }, []);

    const validateFileSize = useCallback(
        (file: File): FileError | null => {
            if (file.size > maxFileSize) {
                return {
                    message: `File size exceeds ${formatBytes(maxFileSize)}`,
                    code: "FILE_TOO_LARGE",
                };
            }
            return null;
        },
        [maxFileSize]
    );

    const validateFileType = useCallback(
        (file: File): FileError | null => {
            if (!acceptedFileTypes?.length) return null;

            const fileType = file.type.toLowerCase();
            if (
                !acceptedFileTypes.some((type) =>
                    fileType.match(type.toLowerCase())
                )
            ) {
                return {
                    message: `File type must be ${acceptedFileTypes.join(
                        ", "
                    )}`,
                    code: "INVALID_FILE_TYPE",
                };
            }
            return null;
        },
        [acceptedFileTypes]
    );

    const handleError = useCallback(
        (error: FileError) => {
            setError(error);
            setStatus("error");
            onUploadError?.(error);

            setTimeout(() => {
                setError(null);
                setStatus("idle");
            }, 3000);
        },
        [onUploadError]
    );

    const simulateUpload = useCallback(
        (uploadingFile: File) => {
            let currentProgress = 0;

            if (uploadIntervalRef.current) {
                clearInterval(uploadIntervalRef.current);
            }

            uploadIntervalRef.current = setInterval(() => {
                currentProgress += UPLOAD_STEP_SIZE;
                if (currentProgress >= 100) {
                    if (uploadIntervalRef.current) {
                        clearInterval(uploadIntervalRef.current);
                    }
                    setProgress(0);
                    setStatus("idle");
                    setFile(null);
                    onUploadSuccess?.(uploadingFile);
                } else {
                    setStatus((prevStatus) => {
                        if (prevStatus === "uploading") {
                            setProgress(currentProgress);
                            return "uploading";
                        }
                        if (uploadIntervalRef.current) {
                            clearInterval(uploadIntervalRef.current);
                        }
                        return prevStatus;
                    });
                }
            }, uploadDelay / (100 / UPLOAD_STEP_SIZE));
        },
        [onUploadSuccess, uploadDelay]
    );

    const handleFileSelect = useCallback(
        (selectedFile: File | null) => {
            if (!selectedFile) return;

            // Reset error state
            setError(null);

            // Validate file
            const sizeError = validateFileSize(selectedFile);
            if (sizeError) {
                handleError(sizeError);
                return;
            }

            const typeError = validateFileType(selectedFile);
            if (typeError) {
                handleError(typeError);
                return;
            }

            const customError = validateFile?.(selectedFile);
            if (customError) {
                handleError(customError);
                return;
            }

            setFile(selectedFile);
            setStatus("uploading");
            setProgress(0);
            simulateUpload(selectedFile);
        },
        [
            simulateUpload,
            validateFileSize,
            validateFileType,
            validateFile,
            handleError,
        ]
    );

    const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setStatus((prev) => (prev !== "uploading" ? "dragging" : prev));
    }, []);

    const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setStatus((prev) => (prev === "dragging" ? "idle" : prev));
    }, []);

    const handleDrop = useCallback(
        (e: DragEvent<HTMLDivElement>) => {
            e.preventDefault();
            e.stopPropagation();
            if (status === "uploading") return;
            setStatus("idle");
            const droppedFile = e.dataTransfer.files?.[0];
            if (droppedFile) handleFileSelect(droppedFile);
        },
        [status, handleFileSelect]
    );

    const handleFileInputChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const selectedFile = e.target.files?.[0];
            handleFileSelect(selectedFile || null);
            if (e.target) e.target.value = "";
        },
        [handleFileSelect]
    );

    const triggerFileInput = useCallback(() => {
        if (status === "uploading") return;
        fileInputRef.current?.click();
    }, [status]);

    const resetState = useCallback(() => {
        setFile(null);
        setStatus("idle");
        setProgress(0);
        if (onFileRemove) onFileRemove();
    }, [onFileRemove]);

    return (
        <div
            className={cn("relative w-full max-w-sm mx-auto", className || "")}
            role="complementary"
            aria-label="File upload"
        >
            <div className="group relative w-full rounded-xl bg-white dark:bg-black ring-1 ring-gray-200 dark:ring-white/10 p-0.5">
                <div className="absolute inset-x-0 -top-px h-px w-full bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />

                <div className="relative w-full rounded-[10px] bg-gray-50/50 dark:bg-white/[0.02] p-1.5">
                    <div
                        className={cn(
                            "relative mx-auto w-full overflow-hidden rounded-lg border border-gray-100 dark:border-white/[0.08] bg-white dark:bg-black/50",
                            error ? "border-red-500/50" : ""
                        )}
                    >
                        <div
                            className={cn(
                                "absolute inset-0 transition-opacity duration-300",
                                status === "dragging"
                                    ? "opacity-100"
                                    : "opacity-0"
                            )}
                        >
                            <div className="absolute inset-x-0 top-0 h-[20%] bg-gradient-to-b from-blue-500/10 to-transparent" />
                            <div className="absolute inset-x-0 bottom-0 h-[20%] bg-gradient-to-t from-blue-500/10 to-transparent" />
                            <div className="absolute inset-y-0 left-0 w-[20%] bg-gradient-to-r from-blue-500/10 to-transparent" />
                            <div className="absolute inset-y-0 right-0 w-[20%] bg-gradient-to-l from-blue-500/10 to-transparent" />
                            <div className="absolute inset-[20%] bg-blue-500/5 rounded-lg transition-all duration-300 animate-pulse" />
                        </div>

                        <div className="absolute -right-4 -top-4 h-8 w-8 bg-gradient-to-br from-blue-500/20 to-transparent blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                        <div className="relative h-[240px]">
                            <AnimatePresence mode="wait">
                                {status === "idle" || status === "dragging" ? (
                                    <motion.div
                                        key="dropzone"
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{
                                            opacity:
                                                status === "dragging" ? 0.8 : 1,
                                            y: 0,
                                            scale:
                                                status === "dragging"
                                                    ? 0.98
                                                    : 1,
                                        }}
                                        exit={{ opacity: 0, y: -10 }}
                                        transition={{ duration: 0.2 }}
                                        className="absolute inset-0 flex flex-col items-center justify-center p-6"
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                        onDrop={handleDrop}
                                    >
                                        <div className="mb-4">
                                            <UploadIllustration />
                                        </div>

                                        <div className="text-center space-y-1.5 mb-4">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white tracking-tight">
                                                Drag and drop or
                                            </h3>
                                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                                {acceptedFileTypes?.length
                                                    ? `${acceptedFileTypes
                                                          .map(
                                                              (t) =>
                                                                  t.split(
                                                                      "/"
                                                                  )[1]
                                                          )
                                                          .join(", ")
                                                          .toUpperCase()}`
                                                    : "SVG, PNG, JPG or GIF"}{" "}
                                                {maxFileSize &&
                                                    `up to ${formatBytes(
                                                        maxFileSize
                                                    )}`}
                                            </p>
                                        </div>

                                        <button
                                            type="button"
                                            onClick={triggerFileInput}
                                            className="w-4/5 flex items-center justify-center gap-2 rounded-lg bg-gray-100 dark:bg-white/10 px-4 py-2.5 text-sm font-semibold text-gray-900 dark:text-white transition-all duration-200 hover:bg-gray-200 dark:hover:bg-white/20 group"
                                        >
                                            <span>Upload File</span>
                                            <UploadCloud className="w-4 h-4 group-hover:scale-110 transition-transform duration-200" />
                                        </button>

                                        <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                                            or drag and drop your file here
                                        </p>

                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            className="sr-only"
                                            onChange={handleFileInputChange}
                                            accept={acceptedFileTypes?.join(
                                                ","
                                            )}
                                            aria-label="File input"
                                        />
                                    </motion.div>
                                ) : status === "uploading" ? (
                                    <motion.div
                                        key="uploading"
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.95 }}
                                        className="absolute inset-0 flex flex-col items-center justify-center p-6"
                                    >
                                        <div className="mb-4">
                                            <UploadingAnimation
                                                progress={progress}
                                            />
                                        </div>

                                        <div className="text-center space-y-1.5 mb-4">
                                            <h3 className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                                                {file?.name}
                                            </h3>
                                            <div className="flex items-center justify-center gap-2 text-xs">
                                                <span className="text-gray-500 dark:text-gray-400">
                                                    {formatBytes(
                                                        file?.size || 0
                                                    )}
                                                </span>
                                                <span className="font-medium text-blue-500">
                                                    {Math.round(progress)}%
                                                </span>
                                            </div>
                                        </div>

                                        <button
                                            onClick={resetState}
                                            type="button"
                                            className="w-4/5 flex items-center justify-center gap-2 rounded-lg bg-gray-100 dark:bg-white/10 px-4 py-2.5 text-sm font-semibold text-gray-900 dark:text-white transition-all duration-200 hover:bg-gray-200 dark:hover:bg-white/20"
                                        >
                                            Cancel
                                        </button>
                                    </motion.div>
                                ) : null}
                            </AnimatePresence>
                        </div>

                        <AnimatePresence>
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    className="absolute bottom-4 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-red-500/10 border border-red-500/20 rounded-lg"
                                >
                                    <p className="text-sm text-red-500 dark:text-red-400">
                                        {error.message}
                                    </p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
}

FileUpload.displayName = "FileUpload";
