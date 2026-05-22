import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { recommenderAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Rocket } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export function VariantSelector({ runId }: { runId: string }) {
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: variants, isLoading } = useQuery({
    queryKey: ['variants', runId],
    queryFn: () => recommenderAPI.getVariants(runId),
    enabled: !!runId,
  })

  const acceptVariant = useMutation({
    mutationFn: (variantId: string) => recommenderAPI.acceptVariant(variantId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['variants', runId] })
      success('Variant accepted', 'New experiment will be created')
    },
  })

  if (isLoading) return <Spinner size="sm" />
  if (!variants?.length) return <div className="text-slate-500 text-sm">No creative variants available for this run.</div>

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-white">Creative Variants</h3>
      {variants.map(variant => (
        <TerminalCard key={variant.id} className="p-3">
          <div className="flex justify-between items-start">
            <div><div className="font-medium text-white">{variant.name}</div><div className="text-xs text-slate-400 mt-0.5">{variant.description}</div><div className="flex flex-wrap gap-1 mt-1">{variant.tags.map(tag => <Badge key={tag}>{tag}</Badge>)}</div><div className="text-xs text-neon-mint mt-1">Expected lift: +{variant.expected_performance_lift}%</div></div>
            <Button size="sm" onClick={() => acceptVariant.mutate(variant.id)}><Rocket size={14} className="mr-1" /> Try</Button>
          </div>
        </TerminalCard>
      ))}
    </div>
  )
}