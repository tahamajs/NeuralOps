import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { acceleratorAPI } from '../services/api'
import { KpiGrid } from '../components/KpiGrid'
import { InsightCardList } from '../components/InsightCardList'
import { ExperimentComparisonChart } from '../components/ExperimentComparisonChart'
import { InterventionSuggestion } from '../components/InterventionSuggestion'
import { OpportunityScoreWidget } from '../components/OpportunityScoreWidget'
import { AutoRefreshToggle } from '../components/AutoRefreshToggle'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Tabs, TabList, Tab, TabPanel } from '../../../components/ui/Tabs'
import { Brain, Sparkles, BarChart3, Lightbulb, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function AcceleratorDashboard({ onClose }: { onClose?: () => void }) {
  const [selectedExperimentIds, setSelectedExperimentIds] = useState<string[]>([])
  const [autoRefresh, setAutoRefresh] = useState({ enabled: true, interval_seconds: 30 })
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: kpis, isLoading: kpisLoading, refetch: refetchKpis } = useQuery({
    queryKey: ['accelerator-kpis'],
    queryFn: () => acceleratorAPI.getKpis(),
    refetchInterval: autoRefresh.enabled ? autoRefresh.interval_seconds * 1000 : false,
  })

  const { data: insights, isLoading: insightsLoading, refetch: refetchInsights } = useQuery({
    queryKey: ['accelerator-insights'],
    queryFn: () => acceleratorAPI.getInsights({ status: 'new', limit: 20 }),
    refetchInterval: autoRefresh.enabled ? autoRefresh.interval_seconds * 1000 : false,
  })

  const { data: opportunityScore, refetch: refetchOpportunity } = useQuery({
    queryKey: ['opportunity-score'],
    queryFn: () => acceleratorAPI.getOpportunityScore(),
    refetchInterval: autoRefresh.enabled ? autoRefresh.interval_seconds * 1000 : false,
  })

  const { data: interventions } = useQuery({
    queryKey: ['interventions'],
    queryFn: () => acceleratorAPI.getInterventions(),
  })

  const generateInsightsMutation = useMutation({
    mutationFn: () => acceleratorAPI.generateInsights(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['accelerator-insights'] })
      success('New insights generated', 'AI analysis completed')
    },
    onError: () => error('Failed to generate insights'),
  })

  const autoRefreshMutation = useMutation({
    mutationFn: (config: { enabled: boolean; interval_seconds: number }) =>
      acceleratorAPI.updateAutoRefreshConfig(config.enabled, config.interval_seconds),
    onSuccess: (_, vars) => setAutoRefresh(vars),
  })

  const handleRefresh = () => {
    refetchKpis()
    refetchInsights()
    refetchOpportunity()
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <Sparkles size={20} className="text-neon-mint" /> Experiment Accelerator
        </h1>
        <div className="flex gap-2">
          <AutoRefreshToggle
            enabled={autoRefresh.enabled}
            interval={autoRefresh.interval_seconds}
            onSave={(enabled, interval) => autoRefreshMutation.mutate({ enabled, interval })}
          />
          <Button size="sm" variant="ghost" onClick={handleRefresh}>
            <RefreshCw size={14} className="mr-1" /> Refresh
          </Button>
          <Button
            size="sm"
            variant="primary"
            onClick={() => generateInsightsMutation.mutate()}
            disabled={generateInsightsMutation.isPending}
          >
            <Brain size={14} className="mr-1" /> Generate AI Insights
          </Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        <KpiGrid kpis={kpis?.data || []} isLoading={kpisLoading} />

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <TerminalCard className="p-4">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-sm font-bold text-white flex items-center gap-2">
                  <BarChart3 size={16} className="text-neon-blue" /> Experiment Comparison
                </h2>
              </div>
              <ExperimentComparisonChart />
            </TerminalCard>
            <InsightCardList insights={insights?.data?.results || []} isLoading={insightsLoading} />
          </div>

          <div className="space-y-6">
            <OpportunityScoreWidget score={opportunityScore?.data} />
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Lightbulb size={16} className="text-yellow-400" /> Suggested Interventions
              </h3>
              {(interventions?.data || []).map((intervention) => (
                <InterventionSuggestion key={intervention.id} suggestion={intervention} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}