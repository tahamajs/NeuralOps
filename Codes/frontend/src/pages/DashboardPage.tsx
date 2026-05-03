import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import MetricSummaryCards from '../components/MetricSummaryCards'
import RecentExperimentsTable from '../components/RecentExperimentsTable'
import ActiveJobsWidget from '../components/ActiveJobsWidget'
import SystemResourcesWidget from '../components/SystemResourcesWidget'

export default function DashboardPage() {
  const { data: experiments = [] } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get('/experiments/')).data,
    refetchInterval: 5000,
  })

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-800">Dashboard</h1>
      <MetricSummaryCards experiments={experiments} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <RecentExperimentsTable experiments={experiments} />
        </div>
        <div>
          <ActiveJobsWidget experiments={experiments} />
        </div>
      </div>
      <SystemResourcesWidget />
    </div>
  )
}