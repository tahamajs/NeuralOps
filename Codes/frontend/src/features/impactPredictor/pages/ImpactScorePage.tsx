import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { impactAPI } from '../services/api'
import { ImpactScoreCard } from '../components/ImpactScoreCard'
import { ForecastingChart } from '../components/ForecastingChart'
import { MetricsRadar } from '../components/MetricsRadar'
import { CitationPredictionWidget } from '../components/CitationPredictionWidget'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Tabs, TabList, Tab, TabPanel } from '../../../components/ui/Tabs'
import { TrendingUp, AlertCircle, Lightbulb, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ImpactScorePage({ onClose }: { onClose?: () => void }) {
  const { projectId } = useParams()
  const [activeTab, setActiveTab] = useState('overview')
  const { success, error } = useToast()

  const {
    data: score,
    isLoading: scoreLoading,
    refetch: refetchScore,
  } = useQuery({
    queryKey: ['impact-score', projectId],
    queryFn: () => impactAPI.getImpactScore(projectId!),
    enabled: !!projectId,
  })

  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['impact-metrics', projectId],
    queryFn: () => impactAPI.getMetrics(projectId!),
    enabled: !!projectId,
  })

  const { data: history } = useQuery({
    queryKey: ['impact-history', projectId],
    queryFn: () => impactAPI.getHistory(projectId!),
    enabled: !!projectId,
  })

  const { data: featureImportance } = useQuery({
    queryKey: ['feature-importance'],
    queryFn: () => impactAPI.getFeatureImportance(),
  })

  const refreshMutation = useMutation({
    mutationFn: () => impactAPI.getImpactScore(projectId!),
    onSuccess: (data) => {
      success('Impact score refreshed', `New score: ${data.overall_score}`)
      refetchScore()
    },
    onError: () => error('Failed to refresh impact score'),
  })

  if (scoreLoading || metricsLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white">Impact Score Dashboard</h1>
          <p className="text-xs text-slate-400">Project ID: {projectId?.slice(0, 8)}</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refreshMutation.mutate()}>
            <RefreshCw size={14} className="mr-1" /> Refresh
          </Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <ImpactScoreCard score={score!} />
          </div>
          <div className="lg:col-span-2">
            <MetricsRadar metrics={metrics!} />
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabList>
            <Tab value="overview">Forecasting & Trends</Tab>
            <Tab value="insights">AI Insights</Tab>
            <Tab value="comparison">Comparison</Tab>
          </TabList>

          <TabPanel value="overview">
            <div className="grid lg:grid-cols-2 gap-6">
              <ForecastingChart history={history || []} predictedScore={score?.predicted_h_index} />
              <CitationPredictionWidget
                predictedCitations={score?.predicted_citations_5y}
                confidenceInterval={score?.confidence_interval}
              />
            </div>
          </TabPanel>

          <TabPanel value="insights">
            <div className="space-y-4">
              <TerminalCard className="p-4">
                <div className="flex items-center gap-2 text-yellow-400 mb-2">
                  <AlertCircle size={16} /> Risk Factors
                </div>
                <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
                  {score?.risk_factors.map((factor, idx) => (
                    <li key={idx}>{factor}</li>
                  ))}
                </ul>
              </TerminalCard>
              <TerminalCard className="p-4">
                <div className="flex items-center gap-2 text-neon-mint mb-2">
                  <Lightbulb size={16} /> Opportunities
                </div>
                <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
                  {score?.opportunities.map((opportunity, idx) => (
                    <li key={idx}>{opportunity}</li>
                  ))}
                </ul>
              </TerminalCard>
              {featureImportance && featureImportance.length > 0 && (
                <TerminalCard className="p-4">
                  <div className="text-sm font-bold text-white mb-2">What Drives Impact?</div>
                  <div className="space-y-3">
                    {featureImportance.slice(0, 5).map((fi) => (
                      <div key={fi.feature}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-300">{fi.feature}</span>
                          <span className="text-neon-mint">{fi.importance.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-1.5">
                          <div className="bg-neon-blue h-1.5 rounded-full" style={{ width: `${fi.importance}%` }} />
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">{fi.description}</p>
                      </div>
                    ))}
                  </div>
                </TerminalCard>
              )}
            </div>
          </TabPanel>

          <TabPanel value="comparison">
            <div className="text-center text-slate-400 py-8">Project comparison view is available in the <strong>Project Leaderboard</strong></div>
          </TabPanel>
        </Tabs>
      </div>
    </div>
  )
}