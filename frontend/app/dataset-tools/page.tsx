"use client"

import { useState, useTransition, useRef } from "react"
import { PanelGroup, Panel, PanelResizeHandle } from "react-resizable-panels"
import { Grid3x3, List, SidebarClose, SidebarOpen, ImageIcon, Brain } from "lucide-react"
import { FileTree } from "@/components/dataset-tools/file-tree"
import { ImagePreview } from "@/components/dataset-tools/image-preview"
import { MetadataPanel } from "@/components/dataset-tools/metadata-panel"
import { SafetensorsPanel } from "@/components/dataset-tools/safetensors-panel"
import { DropZone } from "@/components/dataset-tools/drop-zone"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Empty, EmptyHeader, EmptyMedia, EmptyTitle, EmptyDescription } from "@/components/ui/empty"
import type { FsItem } from "@/types/dataset-tools/fs"
import type { ImageMetadata, ViewMode } from "@/types/dataset-tools/metadata"
import { datasetToolsAPI } from "@/lib/api"
import type { SafetensorsMetadata } from "@/types/dataset-tools/safetensors"
import { useDtSettings } from "@/hooks/use-dt-settings"

function isSafetensorsFile(file: FsItem): boolean {
  return file.name.toLowerCase().endsWith('.safetensors')
}

export default function DatasetToolsPage() {
  const { settings, updateSettings } = useDtSettings()
  const [selectedFile, setSelectedFile] = useState<FsItem | null>(null)
  const [metadata, setMetadata] = useState<{ data: ImageMetadata | null; loading: boolean }>({ data: null, loading: false })
  const [viewMode, setViewMode] = useState<ViewMode>("list")
  const [showMetadata, setShowMetadata] = useState(true)
  const [safetensors, setSafetensors] = useState<{ data: SafetensorsMetadata | null; loading: boolean }>({ data: null, loading: false })
  const [refreshKey, setRefreshKey] = useState(0)
  const [, startTransition] = useTransition()
  const fetchAbortRef = useRef<AbortController | null>(null)

  const fetchSafetensors = async (file: FsItem) => {
    fetchAbortRef.current?.abort()
    const controller = new AbortController()
    fetchAbortRef.current = controller

    setSafetensors({ data: null, loading: true })
    try {
      const data = await datasetToolsAPI.fetchSafetensors(file.path, settings.currentFolder, controller.signal)
      setSafetensors({ data, loading: false })
    } catch (error) {
      if ((error as Error).name === 'AbortError') return
      console.error(error)
      setSafetensors({ data: null, loading: false })
    }
  }

  const fetchMetadata = async (file: FsItem) => {
    // Cancel any in-flight metadata request
    fetchAbortRef.current?.abort()
    const controller = new AbortController()
    fetchAbortRef.current = controller

    setMetadata({ data: null, loading: true });
    try {
      const data = await datasetToolsAPI.fetchMetadata(file.path, settings.currentFolder, controller.signal)
      setMetadata({ data, loading: false })
    } catch (error) {
      if ((error as Error).name === 'AbortError') return;
      console.error(error);
      setMetadata({ data: null, loading: false });
    }
  };

  const handleFileSelect = (file: FsItem) => {
    if (file.isDirectory) {
      setSelectedFile(null)
      setMetadata({ data: null, loading: false })
      setSafetensors({ data: null, loading: false })
      return
    }

    startTransition(() => {
      setSelectedFile(file)
    })

    if (isSafetensorsFile(file)) {
      setMetadata({ data: null, loading: false })
      fetchSafetensors(file)
    } else {
      setSafetensors({ data: null, loading: false })
      fetchMetadata(file)
    }
  }

  // Reload the file list (thumbnails + tree root) and re-read the current file's
  // metadata — used after editing creates/updates files on disk.
  const handleRefresh = () => {
    setRefreshKey((k) => k + 1)
    if (selectedFile && !selectedFile.path.startsWith('blob:')) {
      if (isSafetensorsFile(selectedFile)) fetchSafetensors(selectedFile)
      else fetchMetadata(selectedFile)
    }
  }

  const handleFileDrop = async (file: File, folderPath?: string) => {
    // Show the image immediately
    const objectUrl = URL.createObjectURL(file)
    setSelectedFile({
      name: file.name,
      path: objectUrl,
      isDirectory: false,
    })

    // If we got the folder path from the drop (Electron), switch to it
    if (folderPath) {
      updateSettings({ currentFolder: folderPath })
    }

    // Extract metadata (file is in memory, no disk I/O needed). The whole-machine
    // find-file fallback that DT used is intentionally gone (removed for security);
    // dropped files preview via blob URL without resolving their on-disk folder.
    setMetadata({ data: null, loading: true })
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch('/api/dataset-tools/metadata-from-file', {
        method: 'POST',
        body: formData,
      })
      if (response.ok) {
        setMetadata({ data: await response.json(), loading: false })
      } else {
        setMetadata({ data: null, loading: false })
      }
    } catch {
      setMetadata({ data: null, loading: false })
    }
  }

  return (
    <>
      <DropZone onFileDrop={(file, folder) => handleFileDrop(file, folder)} />
      <div className="flex flex-col h-[calc(100vh-4rem)]">
        {/* Page-specific toolbar */}
        <div className="h-10 border-b border-border bg-muted/20 flex items-center justify-between px-4">
          <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setViewMode("list")}
              aria-label="List view"
              className={cn(
                "h-7 w-7",
                viewMode === "list" && "bg-accent text-accent-foreground shadow-sm"
              )}
            >
              <List className="w-3.5 h-3.5" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setViewMode("thumbnail")}
              aria-label="Thumbnail view"
              className={cn(
                "h-7 w-7",
                viewMode === "thumbnail" && "bg-accent text-accent-foreground shadow-sm"
              )}
            >
              <Grid3x3 className="w-3.5 h-3.5" />
            </Button>
          </div>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => setShowMetadata(!showMetadata)}
            aria-label={showMetadata ? "Hide metadata" : "Show metadata"}
            className="h-7 w-7 text-muted-foreground hover:text-accent-foreground"
          >
            {showMetadata ? <SidebarClose className="w-3.5 h-3.5" /> : <SidebarOpen className="w-3.5 h-3.5" />}
          </Button>
        </div>

        {/* Resizable panel layout */}
        <PanelGroup id="main-layout" direction="horizontal" className="flex-1">
          {/* File Tree (list OR thumbnail per toolbar toggle) */}
          <Panel id="file-tree" defaultSize={15} minSize={10} maxSize={30}>
            <FileTree
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile ?? undefined}
              viewMode={viewMode}
              refreshKey={refreshKey}
            />
          </Panel>

          <PanelResizeHandle className="w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize" />

          {/* Center: Image Preview (full height — bottom thumbnail drawer removed) */}
          <Panel id="center" defaultSize={showMetadata ? 60 : 85} minSize={30}>
            <div className="h-full flex flex-col">
              {!selectedFile ? (
                <Empty className="flex-1 border-0">
                  <EmptyHeader>
                    <EmptyMedia variant="icon"><ImageIcon /></EmptyMedia>
                    <EmptyTitle>No image selected</EmptyTitle>
                    <EmptyDescription>Browse the file tree and select an image to preview</EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : isSafetensorsFile(selectedFile) ? (
                <Empty className="flex-1 border-0">
                  <EmptyHeader>
                    <EmptyMedia variant="icon"><Brain /></EmptyMedia>
                    <EmptyTitle>{selectedFile.name}</EmptyTitle>
                    <EmptyDescription>Model file — see metadata panel</EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : (
                <ImagePreview
                  src={selectedFile.path.startsWith('blob:') ? selectedFile.path : `/api/dataset-tools/image?path=${encodeURIComponent(selectedFile.path)}&baseFolder=${encodeURIComponent(settings.currentFolder)}`}
                  fileName={selectedFile.name}
                  onRefresh={handleRefresh}
                />
              )}
            </div>
          </Panel>

          {/* Metadata / Safetensors Panel */}
          {showMetadata && (
            <>
              <PanelResizeHandle className="w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize" />
              <Panel id="metadata" defaultSize={25} minSize={15} maxSize={50}>
                {selectedFile && isSafetensorsFile(selectedFile) ? (
                  <SafetensorsPanel data={safetensors.data} isLoading={safetensors.loading} fileName={selectedFile.name} />
                ) : (
                  <MetadataPanel
                    metadata={metadata.data}
                    isLoading={metadata.loading}
                    filePath={selectedFile?.path}
                    baseFolder={settings.currentFolder}
                    onRefresh={handleRefresh}
                  />
                )}
              </Panel>
            </>
          )}
        </PanelGroup>
      </div>
    </>
  )
}
