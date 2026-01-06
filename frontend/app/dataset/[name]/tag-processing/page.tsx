'use client';

import { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Home, Database, Sparkles, Plus, Minus, Replace, ArrowLeft, ArrowRight } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { captionAPI } from '@/lib/api';

export default function TagProcessingPage() {
  const params = useParams();
  const router = useRouter();
  const datasetName = params.name as string;

  // Add Trigger Word
  const [triggerWord, setTriggerWord] = useState('');
  const [triggerPosition, setTriggerPosition] = useState<'first' | 'last'>('first');
  const [addingTrigger, setAddingTrigger] = useState(false);

  // Remove Tags
  const [tagsToRemove, setTagsToRemove] = useState('');
  const [removingTags, setRemovingTags] = useState(false);

  // Find & Replace
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [useRegex, setUseRegex] = useState(false);
  const [replacingText, setReplacingText] = useState(false);

  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleAddTrigger = async () => {
    if (!triggerWord.trim()) {
      setMessage({ type: 'error', text: 'Please enter a trigger word' });
      return;
    }

    try {
      setAddingTrigger(true);
      setMessage(null);

      const response = await captionAPI.addTrigger({
        dataset_path: datasetName,
        trigger_word: triggerWord.trim(),
        position: triggerPosition,
      });

      setMessage({
        type: 'success',
        text: `✅ Added "${triggerWord}" to ${response.files_modified} files`,
      });
      setTriggerWord('');
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setAddingTrigger(false);
    }
  };

  const handleRemoveTags = async () => {
    if (!tagsToRemove.trim()) {
      setMessage({ type: 'error', text: 'Please enter tags to remove' });
      return;
    }

    try {
      setRemovingTags(true);
      setMessage(null);

      const tags = tagsToRemove.split(',').map(t => t.trim()).filter(t => t);
      const response = await captionAPI.removeTags({
        dataset_path: datasetName,
        tags_to_remove: tags,
      });

      setMessage({
        type: 'success',
        text: `✅ Removed tags from ${response.files_modified} files`,
      });
      setTagsToRemove('');
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setRemovingTags(false);
    }
  };

  const handleReplace = async () => {
    if (!findText.trim()) {
      setMessage({ type: 'error', text: 'Please enter text to find' });
      return;
    }

    try {
      setReplacingText(true);
      setMessage(null);

      const response = await captionAPI.replace({
        dataset_path: datasetName,
        find: findText,
        replace: replaceText,
        use_regex: useRegex,
      });

      setMessage({
        type: 'success',
        text: `✅ Replaced text in ${response.files_modified} files`,
      });
      setFindText('');
      setReplaceText('');
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setReplacingText(false);
    }
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-5xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: datasetName, href: `/dataset/${datasetName}/tags` },
            { label: 'Tag Processing', icon: <Sparkles className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
            Tag Processing
          </h1>
          <p className="text-xl text-muted-foreground">
            Bulk tag operations for {datasetName}
          </p>
        </div>

        {/* Status Message */}
        {message && (
          <div className={`mb-6 p-4 rounded-lg border ${
            message.type === 'success'
              ? 'bg-green-500/10 border-green-500/50 text-green-400'
              : 'bg-red-500/10 border-red-500/50 text-red-400'
          }`}>
            {message.text}
          </div>
        )}

        {/* Add Trigger Word */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="w-5 h-5 text-green-500" />
              Add Trigger/Activation Tag
            </CardTitle>
            <CardDescription>
              Add a tag to ALL captions (e.g., character name, style trigger)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="trigger-word">Trigger Word</Label>
              <Input
                id="trigger-word"
                value={triggerWord}
                onChange={(e) => setTriggerWord(e.target.value)}
                placeholder="e.g., mycharacter, mystyle"
                disabled={addingTrigger}
              />
            </div>

            <div>
              <Label>Position</Label>
              <div className="flex gap-4 mt-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={triggerPosition === 'first'}
                    onChange={() => setTriggerPosition('first')}
                    disabled={addingTrigger}
                  />
                  <span>First (recommended)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={triggerPosition === 'last'}
                    onChange={() => setTriggerPosition('last')}
                    disabled={addingTrigger}
                  />
                  <span>Last</span>
                </label>
              </div>
            </div>

            <Button
              onClick={handleAddTrigger}
              disabled={addingTrigger || !triggerWord.trim()}
            >
              <Plus className="w-4 h-4 mr-2" />
              {addingTrigger ? 'Adding...' : 'Add Trigger Word'}
            </Button>
          </CardContent>
        </Card>

        {/* Remove Tags */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Minus className="w-5 h-5 text-red-500" />
              Remove Unwanted Tags
            </CardTitle>
            <CardDescription>
              Remove specific tags from ALL captions (e.g., watermark, logo)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="remove-tags">Tags to Remove (comma-separated)</Label>
              <Input
                id="remove-tags"
                value={tagsToRemove}
                onChange={(e) => setTagsToRemove(e.target.value)}
                placeholder="e.g., watermark, logo, text"
                disabled={removingTags}
              />
            </div>

            <Button
              onClick={handleRemoveTags}
              disabled={removingTags || !tagsToRemove.trim()}
              variant="destructive"
            >
              <Minus className="w-4 h-4 mr-2" />
              {removingTags ? 'Removing...' : 'Remove Tags'}
            </Button>
          </CardContent>
        </Card>

        {/* Find & Replace */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Replace className="w-5 h-5 text-blue-500" />
              Find & Replace
            </CardTitle>
            <CardDescription>
              Replace text across ALL captions (supports regex)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="find-text">Find</Label>
              <Input
                id="find-text"
                value={findText}
                onChange={(e) => setFindText(e.target.value)}
                placeholder="Text to find"
                disabled={replacingText}
              />
            </div>

            <div>
              <Label htmlFor="replace-text">Replace With</Label>
              <Input
                id="replace-text"
                value={replaceText}
                onChange={(e) => setReplaceText(e.target.value)}
                placeholder="Replacement text (leave empty to remove)"
                disabled={replacingText}
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="use-regex"
                checked={useRegex}
                onChange={(e) => setUseRegex(e.target.checked)}
                disabled={replacingText}
              />
              <Label htmlFor="use-regex" className="cursor-pointer">
                Use Regular Expression (regex)
              </Label>
            </div>

            <Button
              onClick={handleReplace}
              disabled={replacingText || !findText.trim()}
            >
              <Replace className="w-4 h-4 mr-2" />
              {replacingText ? 'Replacing...' : 'Find & Replace'}
            </Button>
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-between items-center mt-8">
          <Button
            variant="outline"
            onClick={() => router.push(`/dataset/${datasetName}/auto-tag`)}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Auto-Tag
          </Button>

          <Button
            onClick={() => router.push(`/dataset/${datasetName}/tags`)}
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
          >
            Gallery Editor
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
}
