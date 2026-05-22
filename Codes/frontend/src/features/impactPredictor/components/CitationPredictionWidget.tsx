import { TerminalCard } from '../../../components/ui/TerminalCard'
import { ArrowUpRight } from 'lucide-react'

export function CitationPredictionWidget({
  predictedCitations,
  confidenceInterval,
}: {
  predictedCitations?: number
  confidenceInterval?: [number, number]
}) {
  if (!predictedCitations) {
    return (
      <TerminalCard className="p-4 text-center text-slate-500">
        <ArrowUpRight size={24} className="mx-auto mb-2 opacity-30" />
        <p>Not enough data for citation prediction</p>
      </TerminalCard>
    )
  }

  return (
    <TerminalCard className="p-4">
      <h3 className="text-sm font-bold text-white mb-2">Citation Prediction (5 years)</h3>
      <div className="text-4xl font-bold text-neon-mint">{predictedCitations.toLocaleString()}</div>
      {confidenceInterval && (
        <div className="text-xs text-slate-400 mt-1">
          95% CI: {confidenceInterval[0].toLocaleString()} – {confidenceInterval[1].toLocaleString()}
        </div>
      )}
      <div className="mt-3 pt-3 border-t border-white/10">
        <div className="text-xs text-slate-400">Based on:</div>
        <ul className="text-xs text-slate-300 list-disc list-inside mt-1">
          <li>Field‑normalized citation rates</li>
          <li>Venue impact factors</li>
          <li>Collaboration network density</li>
          <li>Historical citation velocity</li>
        </ul>
      </div>
    </TerminalCard>
  )
}