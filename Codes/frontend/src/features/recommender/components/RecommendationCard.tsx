import { useMutation, useQueryClient } from '@tanstack/react-query'
import { recommenderAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Rocket, X } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export function RecommendationCard({ recommendation }: { recommendation: any }) {
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const accept = useMutation({
    mutationFn: () => recommenderAPI.acceptRecommendation(recommendation.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['recommendations'] })
      success('Recommendation accepted', 'Launching experiment...')
    },
  })

  const dismiss = useMutation({
    mutationFn: () => recommenderAPI.dismissRecommendation(recommendation.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['recommendations'] })
      success('Recommendation dismissed')
    },
  })

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2"><Badge variant="info">{recommendation.type}</Badge><span className="text-xs text-slate-500">Confidence: {Math.round(recommendation.confidence * 100)}%</span></div>
          <h3 className="text-white font-bold mt-1">{recommendation.title}</h3>
          <p className="text-sm text-slate-300 mt-1">{recommendation.description}</p>
          <div className="mt-2 text-xs text-neon-mint">Expected improvement: +{recommendation.expected_improvement}%</div>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="primary" onClick={() => accept.mutate()} disabled={accept.isPending}><Rocket size={14} className="mr-1" /> Launch</Button>
          <Button size="sm" variant="ghost" onClick={() => dismiss.mutate()} disabled={dismiss.isPending}><X size={14} /></Button>
        </div>
      </div>
    </TerminalCard>
  )
}