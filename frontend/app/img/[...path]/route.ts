import { NextRequest, NextResponse } from 'next/server';
import { API_BASE } from '@/lib/api';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    // Await params in Next.js 15
    const resolvedParams = await params;

    // Reconstruct the full path
    const imagePath = resolvedParams.path.join('/');

    // Fetch from backend
    const backendUrl = `${API_BASE}/files/image/${imagePath}`;
    const response = await fetch(backendUrl);

    if (!response.ok) {
      return new NextResponse('Image not found', { status: response.status });
    }

    // Get the image data
    const imageData = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';

    // Return the image with proper headers
    return new NextResponse(imageData, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (error) {
    console.error('Error proxying image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
