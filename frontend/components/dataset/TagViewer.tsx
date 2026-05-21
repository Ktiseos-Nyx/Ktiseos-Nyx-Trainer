'use client';

import { useEffect, useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Search,
  Tag,
  Trash2,
  ArrowLeftRight,
  Image as ImageIcon,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { ReplaceTagsDialog } from '@/components/dataset/ReplaceTagsDialog';
import type { ImageWithTags } from '@/lib/api';

/** Sentinel value representing images with no tags — mirrors Civitai's blankTagStr. */
export const BLANK_TAG = '@@none@@';

interface TagViewerProps {
  images: ImageWithTags[];
  selectedTags: string[];
  onSelectedTagsChange: (tags: string[]) => void;
  onRemoveTags: (tags: string[]) => void;
  onReplaceTags: (replacements: Record<string, string>) => void;
}

/**
 * Collapsible tag frequency viewer with multi-select chips, search, and bulk actions.
 *
 * Mirrors Civitai's TrainingImagesTagViewer: chips select tags that filter the
 * image grid in the parent, and the Actions menu drives remove/replace bulk ops.
 * Untagged images are represented by the BLANK_TAG sentinel shown as "None".
 */
export function TagViewer({
  images,
  selectedTags,
  onSelectedTagsChange,
  onRemoveTags,
  onReplaceTags,
}: TagViewerProps) {
  const [open, setOpen] = useState(true);
  const [search, setSearch] = useState('');
  const [tagList, setTagList] = useState<[string, number][]>([]);
  const [replaceOpen, setReplaceOpen] = useState(false);

  useEffect(() => {
    const searchLower = search.toLowerCase();
    const allTags = images.flatMap(img => img.tags);
    const filtered = searchLower
      ? allTags.filter(t => t.toLowerCase().includes(searchLower))
      : allTags;

    const counts = filtered.reduce<Record<string, number>>((acc, tag) => {
      acc[tag] = (acc[tag] ?? 0) + 1;
      return acc;
    }, {});

    const sorted: [string, number][] = Object.entries(counts).sort(([, a], [, b]) => b - a);

    const untaggedCount = images.filter(img => img.tags.length === 0).length;
    if (untaggedCount > 0 && !search) {
      sorted.unshift([BLANK_TAG, untaggedCount]);
    }

    setTagList(sorted);

    // Prune selections that no longer exist in the current image list
    const valid = new Set(allTags);
    onSelectedTagsChange(
      selectedTags.filter(t => (t === BLANK_TAG ? untaggedCount > 0 : valid.has(t)))
    );
  // Intentionally omit selectedTags from deps — pruning must not re-trigger itself
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, search]);

  const selectedNonBlank = selectedTags.filter(t => t !== BLANK_TAG);

  return (
    <>
      <Collapsible open={open} onOpenChange={setOpen} className="mb-6 bg-card border border-border rounded-lg shadow-sm">
        <CollapsibleTrigger asChild>
          <button type="button" className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/30 rounded-t-lg select-none text-left">
            <div className="flex items-center gap-3">
              <Tag className="w-5 h-5 text-cyan-500" />
              <span className="text-lg font-semibold">Tag Viewer</span>
              <Badge variant="outline" className="gap-1 text-xs">
                <ImageIcon className="w-3 h-3" />
                {images.length}
              </Badge>
              {selectedTags.length > 0 && (
                <Badge variant="destructive" className="gap-1 text-xs">
                  <Tag className="w-3 h-3" />
                  {selectedTags.length} selected
                </Badge>
              )}
            </div>
            {open
              ? <ChevronUp className="w-4 h-4 text-muted-foreground" />
              : <ChevronDown className="w-4 h-4 text-muted-foreground" />
            }
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-5 pb-5 space-y-3">
            <p className="text-sm text-muted-foreground">
              Click tags to filter images. Use Actions to bulk-edit selected tags.
            </p>

            {/* Search + actions row */}
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 w-4 h-4 text-muted-foreground pointer-events-none" />
                <Input
                  placeholder="Search tags..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  className="pl-8 pr-8"
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch('')}
                    className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                    aria-label="Clear search"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              <Button
                variant="ghost"
                size="sm"
                disabled={selectedTags.length === 0}
                onClick={() => onSelectedTagsChange([])}
              >
                Deselect All
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="sm" disabled={selectedNonBlank.length === 0}>
                    Actions <ChevronDown className="ml-1 w-3 h-3" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    className="gap-2 text-destructive focus:text-destructive"
                    onClick={() => {
                      if (confirm(`Remove ${selectedNonBlank.length} tag(s) from all images?`)) {
                        onRemoveTags(selectedNonBlank);
                        onSelectedTagsChange([]);
                      }
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                    Remove {selectedNonBlank.length} tag{selectedNonBlank.length !== 1 ? 's' : ''}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="gap-2"
                    onClick={() => setReplaceOpen(true)}
                  >
                    <ArrowLeftRight className="w-4 h-4" />
                    Replace {selectedNonBlank.length} tag{selectedNonBlank.length !== 1 ? 's' : ''}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Tag chips */}
            {tagList.length === 0 ? (
              <p className="text-sm text-muted-foreground italic text-center py-6">
                No tags to display.
              </p>
            ) : (
              <ToggleGroup
                type="multiple"
                value={selectedTags}
                onValueChange={onSelectedTagsChange}
                className="flex flex-wrap justify-start gap-1.5 max-h-64 overflow-y-auto p-2 bg-background/50 rounded-md"
              >
                {tagList.map(([tag, count]) => (
                  <ToggleGroupItem
                    key={tag}
                    value={tag}
                    variant="outline"
                    size="sm"
                    className="h-auto py-1 px-2.5 rounded-full text-xs font-normal data-[state=on]:bg-cyan-500/20 data-[state=on]:border-cyan-500 data-[state=on]:text-cyan-400"
                  >
                    {tag === BLANK_TAG
                      ? <span className="text-red-400 font-medium">None</span>
                      : tag
                    }
                    <span className="ml-1.5 opacity-60">({count})</span>
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>

      <ReplaceTagsDialog
        open={replaceOpen}
        onOpenChange={setReplaceOpen}
        selectedTags={selectedNonBlank}
        onConfirm={replacements => {
          onReplaceTags(replacements);
          onSelectedTagsChange([]);
        }}
      />
    </>
  );
}
