import { useQuery } from '@tanstack/react-query'
import React, { Suspense, lazy } from 'react'
import api from '../api/client'
import MetricSummaryCards from '../components/MetricSummaryCards'

const ExperimentTable = lazy(() => import('../components/ExperimentTable'))
const ActiveJobsWidget = lazy(() => import('../components/ActiveJobsWidget'))
const SystemResourcesWidget = lazy(() => import('../components/SystemResourcesWidget'))

const LoadingCard = () => (
  <div className="glass-card animate-pulse h-48 flex items-center justify-center">
    <div className="text-slate-600 font-black tracking-widest text-[10px]">INITIALIZING_WIDGET...</div>
  </div>
)

export default function DashboardPage() {
  const { data: experiments = [] } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get('/experiments/')).data,
    refetchInterval: 5000,
  })

  return (
    <div className="space-y-8 animate-fade-in relative">
      <div className="absolute top-[-100px] right-[-100px] w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-100px] left-[-100px] w-96 h-96 bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none" />

      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-black text-white tracking-tighter">
            SYSTEM <span className="text-blue-500">CORE</span>
          </h1>
          <p className="text-slate-400 font-medium tracking-tight">Real-time attention compression monitoring engine</p>
        </div>
        <div className="flex gap-2">
           <div className="glass-card py-2 px-4 flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.5)]" />
              <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">Cluster Node Active</span>
           </div>
        </div>
      </header>

      <section>
        <MetricSummaryCards experiments={experiments} />
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="glass-card overflow-hidden">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span className="w-1 h-6 bg-blue-500 rounded-full" />
                RECENT ACTIVITY
              </h2>
              <button className="text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors">VIEW ALL →</button>
            </div>
            <div className="overflow-x-auto">
              <Suspense fallback={<LoadingCard />}>
                <ExperimentTable experiments={experiments.slice(0, 8)} />
              </Suspense>
            </div>
          </div>
        </div>

        <div className="lg:col-span-4 space-y-8">
          <div className="glass-card border-none bg-gradient-to-br from-blue-600/20 to-indigo-600/20">
            <h2 className="text-sm font-black text-white uppercase tracking-[0.2em] mb-4">Quick Stats</h2>
            <div className="grid grid-cols-2 gap-4">
               <div>
                  <div className="text-xs font-bold text-slate-500 mb-1">COMPRESSION RATE</div>
                  <div className="text-2xl font-black text-blue-400">84.2%</div>
               </div>
               <div>
                  <div className="text-xs font-bold text-slate-500 mb-1">LATENCY REDUX</div>
                  <div className="text-2xl font-black text-indigo-400">-42ms</div>
               </div>
            </div>
          </div>
          
          <Suspense fallback={<LoadingCard />}>
            <ActiveJobsWidget experiments={experiments} />
          </Suspense>

          <Suspense fallback={<LoadingCard />}>
            <SystemResourcesWidget />
          </Suspense>
        </div>
      </div>
    </div>
  )
}