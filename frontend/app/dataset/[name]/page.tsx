import { redirect } from 'next/navigation';
import { use } from 'react';

export default function DatasetDashboardPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  // Redirect to tags view by default
  redirect(`/dataset/${name}/tags`);
}
