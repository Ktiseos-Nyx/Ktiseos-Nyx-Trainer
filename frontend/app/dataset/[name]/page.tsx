import { redirect } from 'next/navigation';

export default function DatasetDashboardPage({ params }: { params: { name: string } }) {
  // Redirect to tags view by default
  redirect(`/dataset/${params.name}/tags`);
}
