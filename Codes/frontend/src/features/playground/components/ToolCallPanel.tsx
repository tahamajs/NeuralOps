import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Code } from 'lucide-react'

export function ToolCallPanel({ toolCall, onExecute }: any) {
  return (
    <TerminalCard className="p-3">
      <div className="flex justify-between items-center"><div><span className="font-mono text-sm">{toolCall.function.name}</span><pre className="text-xs text-slate-400 mt-1">{toolCall.function.arguments}</pre></div><button onClick={() => onExecute(toolCall)} className="text-neon-mint text-sm">Execute</button></div>
    </TerminalCard>
  )
}