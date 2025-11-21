'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, Download, RotateCw, Save, ChevronLeft, ChevronRight, ImageIcon, Loader2, Home, Database, Edit } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Dynamic import to avoid SSR issues with canvas/Fabric.js
const ImageEditor = dynamic(
  () => import('@ozdemircibaris/react-image-editor').then((mod) => mod.ImageEditor),
  { ssr: false }
);

export default function ImageEditorPage() {
  // Dataset management
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [loading, setLoading] = useState(false);

  // Editor state
  const [selectedImage, setSelectedImage] = useState<ImageWithTags | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [saving, setSaving] = useState(false);

  // Load datasets
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const data = await datasetAPI.list();
        setDatasets(data.datasets || []);
        if (data.datasets && data.datasets.length > 0) {
          setSelectedDataset(data.datasets[0].path);
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    loadDatasets();
  }, []);

  // Load images when dataset changes
  useEffect(() => {
    if (!selectedDataset) return;

    const loadImages = async () => {
      try {
        setLoading(true);
        const data = await datasetAPI.getImagesWithTags(selectedDataset);
        setImages(data.images || []);
      } catch (err) {
        console.error('Failed to load images:', err);
      } finally {
        setLoading(false);
      }
    };

    loadImages();
  }, [selectedDataset]);

  const handleSelectImage = (img: ImageWithTags, index: number) => {
    setSelectedImage(img);
    setSelectedIndex(index);
  };

  const handleSave = async (imageBlob: Blob) => {
    if (!selectedImage) return;

    try {
      setSaving(true);

      // Convert blob to base64
      const reader = new FileReader();
      reader.readAsDataURL(imageBlob);
      reader.onloadend = async () => {
        const base64data = reader.result as string;

        // Save via API
        const response = await fetch(`${API_BASE}/files/write`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: selectedImage.image_path,
            content: base64data.split(',')[1], // Remove data:image/png;base64, prefix
          }),
        });

        if (response.ok) {
          alert('Image saved successfully!');
        } else {
          alert('Failed to save image');
        }
      };
    } catch (err) {
      console.error('Failed to save:', err);
      alert('Error saving image');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setSelectedImage(null);
    setSelectedIndex(-1);
  };

  const handlePrevious = () => {
    if (selectedIndex > 0) {
      const prevIndex = selectedIndex - 1;
      handleSelectImage(images[prevIndex], prevIndex);
    }
  };

  const handleNext = () => {
    if (selectedIndex < images.length - 1) {
      const nextIndex = selectedIndex + 1;
      handleSelectImage(images[nextIndex], nextIndex);
    }
  };

  const getImageUrl = (imagePath: string) => {
    // Remove leading slash if present
    const cleanPath = imagePath.startsWith('/') ? imagePath.substring(1) : imagePath;
    return `${API_BASE}/files/read/${encodeURIComponent(cleanPath)}`;
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Image Editor', icon: <Edit className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
            Image Editor
          </h1>
          <p className="text-xl text-muted-foreground">
            Crop, edit, and prepare images for your training dataset
          </p>
        </div>

        {/* Dataset Selector */}
        <div className="mb-6 bg-card backdrop-blur-sm rounded-lg border border-border p-6">
          <label className="block text-sm font-medium text-foreground mb-2">Select Dataset</label>
          <select
            value={selectedDataset}
            onChange={(e) => {
              setSelectedDataset(e.target.value);
              setSelectedImage(null);
              setSelectedIndex(-1);
            }}
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {datasets.map((dataset) => (
              <option key={dataset.path} value={dataset.path}>
                {dataset.name} ({dataset.image_count} images)
              </option>
            ))}
          </select>
        </div>

        {/* Gallery View */}
        {!selectedImage && (
          <Card>
            <CardHeader>
              <CardTitle>Image Gallery</CardTitle>
              <CardDescription>
                Select an image to edit
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-64">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
              ) : images.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
                  <ImageIcon className="w-16 h-16 mb-4" />
                  <p>No images found in this dataset</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                  {images.map((img, index) => (
                    <button
                      key={img.image_path}
                      onClick={() => handleSelectImage(img, index)}
                      className="group relative aspect-square rounded-lg overflow-hidden border-2 border-border hover:border-primary transition-all bg-card hover:shadow-lg"
                    >
                      <img
                        src={getImageUrl(img.image_path)}
                        alt={img.image_name}
                        className="w-full h-full object-cover"
                      />

                      {/* Overlay */}
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="absolute bottom-0 left-0 right-0 p-3">
                          <div className="text-white text-xs font-medium truncate">
                            {img.image_name}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Editor View */}
        {selectedImage && (
          <div className="space-y-4">
            {/* Navigation Bar */}
            <Card>
              <CardContent className="flex items-center justify-between p-4">
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handlePrevious}
                    disabled={selectedIndex === 0}
                    variant="outline"
                    size="sm"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    {selectedIndex + 1} / {images.length}
                  </span>
                  <Button
                    onClick={handleNext}
                    disabled={selectedIndex === images.length - 1}
                    variant="outline"
                    size="sm"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>

                <div className="text-sm font-medium truncate max-w-md">
                  {selectedImage.image_name}
                </div>

                <Button onClick={handleCancel} variant="ghost" size="sm">
                  Back to Gallery
                </Button>
              </CardContent>
            </Card>

            {/* Image Editor */}
            <Card>
              <CardContent className="p-6">
                <div className="w-full min-h-[700px] border rounded-lg overflow-hidden bg-muted/20">
                  <ImageEditor
                    imageUrl={getImageUrl(selectedImage.image_path)}
                    onSave={handleSave}
                    onCancel={handleCancel}
                    showCancelButton={true}
                    saveButtonTitle={saving ? "Saving..." : "Save Changes"}
                    cancelButtonTitle="Back to Gallery"
                    className="w-full h-full"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Info Card */}
            <Card>
              <CardHeader>
                <CardTitle>Image Editor Tools</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <h3 className="font-semibold mb-1">Crop</h3>
                    <p className="text-muted-foreground">
                      Resize and crop images to standard training resolutions (512x512, 1024x1024)
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Blur</h3>
                    <p className="text-muted-foreground">
                      Blur sensitive areas, watermarks, or unwanted elements
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Draw & Annotate</h3>
                    <p className="text-muted-foreground">
                      Add shapes and annotations to mark regions of interest
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
