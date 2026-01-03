"use client"

import { cn } from "@/lib/utils"
import { useEffect, useState } from "react"
import {
	HoverCard,
	HoverCardContent,
	HoverCardTrigger,
} from "@/components/ui/hover-card"
import { IconExternalLink } from "@tabler/icons-react"

type PreviewMetadata = {
	title: string | null
	description: string | null
	image: string | null
	url: string | null
}

export interface PreviewProps {
	url: string
	children?: React.ReactNode
	showImage?: boolean
	showTitle?: boolean
	showDescription?: boolean
	className?: string
	contentClassName?: string
	onError?: (error: Error) => void
}

const fetchMetadata = async (url: string): Promise<PreviewMetadata> => {
	try {
		// Use a CORS proxy for client-side requests
		const proxyUrl = `/api/preview?url=${encodeURIComponent(url)}`
		const response = await fetch(proxyUrl)
		
		if (!response.ok) {
			throw new Error(`Failed to fetch metadata: ${response.statusText}`)
		}
		
		return await response.json()
	} catch (error) {
		// Fallback to basic metadata from URL
		const domain = new URL(url).hostname
		return {
			title: domain,
			description: `Visit ${domain}`,
			image: null,
			url,
		}
	}
}

export function Preview({
	url,
	children,
	showImage = true,
	showTitle = true,
	showDescription = true,
	className,
	contentClassName,
	onError,
}: PreviewProps) {
	const [metadata, setMetadata] = useState<PreviewMetadata | null>(null)
	const [loading, setLoading] = useState(false)

	useEffect(() => {
		async function fetchData() {
			setLoading(true)
			try {
				const data = await fetchMetadata(url)
				setMetadata(data)
			} catch (error) {
				const err = error instanceof Error ? error : new Error('Unknown error')
				onError?.(err)
				
				// Fallback metadata
				const domain = new URL(url).hostname
				setMetadata({
					title: domain,
					description: `Visit ${domain}`,
					image: null,
					url,
				})
			} finally {
				setLoading(false)
			}
		}

		fetchData()
	}, [url, onError])

	const defaultTrigger = (
		<a
			href={url}
			className="text-primary hover:underline inline-flex items-center gap-1"
		>
			{new URL(url).hostname}
			<IconExternalLink className="size-3" />
		</a>
	)

	return (
		<HoverCard openDelay={200}>
			<HoverCardTrigger asChild className={className}>
				{children || defaultTrigger}
			</HoverCardTrigger>
			<HoverCardContent className={cn("md:w-80 p-0 overflow-hidden border", contentClassName)}>
                <a href={url}>
				{loading ? (
					<div className="flex items-center justify-center py-4">
						<div className="text-sm text-muted-foreground">Loading...</div>
					</div>
				) : (
					<div>
						{showImage && metadata?.image && (
							<img
								src={metadata.image}
								alt={metadata.title || ""}
								className="aspect-video w-full border object-cover bg-muted"
							/>
						)}
						<div className="p-3">
							{showTitle && metadata?.title && (
								<h4 className="font-semibold text-sm leading-tight">
									{metadata.title}
								</h4>
							)}
							{showDescription && metadata?.description && (
								<p className="text-muted-foreground text-xs leading-relaxed line-clamp-2">
									{metadata.description}
								</p>
							)}
							
						</div>
					</div>
				)}</a>
			</HoverCardContent>
		</HoverCard>
	)
}
