// src/features/playground/pages/ModelComparisonView.tsx
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { playgroundAPI } from '../services/api'
import { ComparisonTable } from '../components/ComparisonTable'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Textarea } from '../../../components/ui/Input'
import { ModelSelector } from '../components/ModelSelector'
import { ParameterControls } from '../components/ParameterControls'
import { Spinner } from '../../../components/ui/Spinner'
import { GitBranch, Play } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ModelComparisonView({ onClose }: { onClose?: () => void }) {
  const [prompt, setPrompt] = useState('')
  const [configA, setConfigA] = useState({ provider: 'openai', model: 'gpt-4o-mini', temperature: 0.7, max_tokens: 500, top_p: 0.9, frequency_penalty: 0, presence_penalty: 0 })
  const [configB, setConfigB] = useState({ provider: 'openai', model: 'gpt-3.5-turbo', temperature: 0.7, max_tokens: 500, top_p: 0.9, frequency_penalty: 0, presence_penalty: 0 })
  const [result, setResult] = useState<any>(null)
  const { success, error: toastError } = useToast()

  const compareMutation = useMutation({
    mutationFn: () => playgroundAPI.compareModels(configA, configB, prompt),
    onSuccess: (res) => setResult(res.data),
    onError: () => toastError('Comparison failed'),
  })

  const handleCompare = () => { if (prompt.trim()) compareMutation.mutate() }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><GitBranch size={20} className="text-neon-mint" /> Model Comparison</h1><p className="text-xs text-slate-400">Side‑by‑side LLM output comparison</p></div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4"><Textarea rows={4} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Enter prompt to test both models..." /></TerminalCard>
        <div className="grid lg:grid-cols-2 gap-4"><TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Model A</h3><ModelSelector models={[]} value={configA} onChange={setConfigA} /><ParameterControls config={configA} onChange={setConfigA} /></TerminalCard>
          <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Model B</h3><ModelSelector models={[]} value={configB} onChange={setConfigB} /><ParameterControls config={configB} onChange={setConfigB} /></TerminalCard>
        </div>
        <Button onClick={handleCompare} disabled={!prompt.trim() || compareMutation.isPending}><Play size={14} className="mr-1" /> Compare Models</Button>
        {compareMutation.isPending && <Spinner />}
        {result && <ComparisonTable result={result} />}
      </div>
    </div>
  )
}