'use client';

import { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';

interface ReplaceTagsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedTags: string[];
  onConfirm: (replacements: Record<string, string>) => void;
}

/**
 * Modal for replacing selected tags across all images in a dataset.
 *
 * Shows each selected tag alongside a text input for its replacement value.
 * Tags with an empty replacement input are left unchanged.
 */
export function ReplaceTagsDialog({
  open,
  onOpenChange,
  selectedTags,
  onConfirm,
}: ReplaceTagsDialogProps) {
  const [replacements, setReplacements] = useState<Record<string, string>>({});

  const handleConfirm = () => {
    onConfirm(replacements);
    setReplacements({});
    onOpenChange(false);
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) setReplacements({});
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Replace tags</DialogTitle>
          <DialogDescription className="sr-only">
            Enter a replacement value for each selected tag. Leave blank to keep the tag unchanged.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-2">
          {selectedTags.map(tag => {
            const inputId = `replace-${tag.replace(/\s+/g, '-')}`;
            return (
              <div key={tag} className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                <Badge variant="outline" className="justify-center truncate py-2">
                  {tag}
                </Badge>
                <ArrowRight className="w-4 h-4 text-muted-foreground shrink-0" />
                <div>
                  <label htmlFor={inputId} className="sr-only">
                    Replace "{tag}" with
                  </label>
                  <Input
                    id={inputId}
                    placeholder={tag}
                    value={replacements[tag] ?? ''}
                    onChange={e =>
                      setReplacements(prev => ({ ...prev, [tag]: e.target.value }))
                    }
                  />
                </div>
              </div>
            );
          })}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirm}>Confirm</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
