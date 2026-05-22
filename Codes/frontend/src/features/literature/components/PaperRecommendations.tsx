import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { literatureAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Sparkles, X, Bookmark } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export function PaperRecommendations({ paperId }: { paperId: string }) {
  const queryClient = useQueryClient()
  const { success } = useToast()

  const { data: recommendations, isLoading, refetch } = useQuery({
    queryKey: ['paper-recommendations', paperId],
    queryFn: () => literatureAPI.getRecommendations(paperId),
    enabled: !!paperId,
  })

  const generate = useMutation({
    mutationFn: () => literatureAPI.generateRecommendations(paperId),
    onSuccess: () => refetch(),
  })

  const dismiss = useMutation({
    mutationFn: (recId: string) => literatureAPI.dismissRecommendation(recId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['paper-recommendations', paperId] }),
  })

  const save = useMutation({
    mutationFn: (recId: string) => literatureAPI.saveRecommendation(recId),
    onSuccess: () => success('Recommendation saved'),
  })

  if (isLoading) return <Spinner size="sm" />
  if (!recommendations?.length) return <div className="text-center text-slate-500 text-sm p-4">No recommendations yet. <Button size="sm" variant="ghost" onClick={() => generate.mutate()}>Generate</Button></div>

  return (
    <div className="space-y-3">
      {recommendations.map(rec => (
        <TerminalCard key={rec.paper_id} className="p-3">
          <div className="flex justify-between items-start gap-2">
            <div><div className="font-medium text-white text-sm">{rec.title}</div><div className="text-xs text-slate-400 mt-0.5">Relevance: {Math.round(rec.relevance_score * 100)}% · {rec.reason}</div></div>
            <div className="flex gap-1"><button onClick={() => save.mutate(rec.paper_id)}><Bookmark size={14} className="text-slate-400 hover:text-yellow-400" /></button><button onClick={() => dismiss.mutate(rec.paper_id)}><X size={14} className="text-slate-400 hover:text-red-400" /></button></div>
          </div>
        </TerminalCard>
      ))}
    </div>
  )
}