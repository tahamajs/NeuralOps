import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { acceleratorAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Rocket, AlertCircle, Zap, ChevronDown, ChevronUp } from 'lucide-react'
import type { InterventionSuggestion } from '../types'
import { useToast } from '../../../components/ui/Toast'

export function InterventionSuggestion({ suggestion }: { suggestion: InterventionSuggestion }) {
  const [expanded, setExpanded] = useState(false)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const apply = useMutation({
    mutationFn: () => acceleratorAPI.applyIntervention(suggestion.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['interventions'] })
      success('Intervention applied', 'New experiment will be created with suggested parameters.')
    },
    onError: () => error('Failed to apply intervention'),
  })

  const priorityColor = {
    high: 'bg-red-500/20 text-red-400 border-red-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-green-500/20 text-green-400 border-green-500/30',
  }[suggestion.priority]

  const effortIcons = { low: '🟢', medium: '🟡', high: '🔴' }

  return (
    <TerminalCard className="p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1"><div className="flex items-center gap-2 flex-wrap"><span className={`px-2 py-0.5 rounded text-xs font-medium ${priorityColor}`}>{suggestion.priority}</span><span className="text-xs text-slate-500">Effort: {effortIcons[suggestion.effort_estimate]}</span><span className="text-xs text-neon-blue">+{suggestion.expected_improvement}% expected</span></div>
          <h3 className="text-white font-medium mt-1">{suggestion.title}</h3>
          <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">{expanded ? suggestion.description : suggestion.description.slice(0, 100)}</p>
          {!expanded && suggestion.description.length > 100 && <button onClick={() => setExpanded(true)} className="text-xs text-neon-mint mt-0.5">Read more</button>}
          {expanded && suggestion.parameters && <pre className="mt-2 text-xs bg-black/30 p-2 rounded overflow-auto">{JSON.stringify(suggestion.parameters, null, 2)}</pre>}
          {expanded && <button onClick={() => setExpanded(false)} className="text-xs text-neon-mint mt-1">Show less</button>}
        </div>
        <Button size="sm" variant="primary" onClick={() => apply.mutate()} disabled={apply.isPending}><Rocket size={12} className="mr-1" /> Try</Button>
      </div>
    </TerminalCard>
  )
}