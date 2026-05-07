import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { MetricSummaryCards } from '../components/MetricSummaryCards'
import { ExperimentTable } from '../components/ExperimentTable'
import { ActiveJobsWidget } from '../components/ActiveJobsWidget'
import { SystemResourcesWidget } from '../../hardware/components/SystemResourcesWidget'
import { Experiment } from '../../../types/experiment'

export default function DashboardPage() {
  const { data: experiments = [], isLoading } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data,
    refetchInterval: 5000,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white tracking-tighter">
            SYSTEM <span className="text-neon-blue">CORE</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">Real‑time attention compression monitoring</p>
        </div>
        <Card className="py-1.5 px-4 inline-flex items-center gap-2 border-none bg-neon-blue/5" hover={false}>
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs font-bold text-green-400 uppercase tracking-widest">Cluster Active</span>
        </Card>
      </div>

      <MetricSummaryCards experiments={experiments} />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-8">
          <Card>
            <h2 className="text-sm font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
              <span className="w-1 h-5 bg-neon-blue rounded-full" />
              Recent Activity
            </h2>
            <ExperimentTable experiments={experiments.slice(0, 8)} />
          </Card>
        </div>
        <div className="lg:col-span-4 space-y-8">
          <Card className="border-none bg-gradient-to-br from-blue-500/10 to-purple-500/10" hover={false}>
            <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-4">Quick Stats</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-[10px] text-slate-500">COMPRESSION</div>
                <div className="text-2xl font-black text-neon-blue">84.2%</div>
              </div>
              <div>
                <div className="text-[10px] text-slate-500">LATENCY Δ</div>
                <div className="text-2xl font-black text-purple-400">-42ms</div>
              </div>
            </div>
          </Card>
          <ActiveJobsWidget experiments={experiments} />
          <SystemResourcesWidget />
        </div>
      </div>
    </div>
  )
}