// src/features/llmHub/pages/GuardrailEditor.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { llmHubAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Textarea } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Switch } from '../../../components/ui/Switch'
import { Shield, Plus, Trash2, TestTube } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function GuardrailEditor({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [testText, setTestText] = useState('')
  const [testResult, setTestResult] = useState<any>(null)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: guardrails, isLoading } = useQuery({ queryKey: ['guardrails'], queryFn: () => llmHubAPI.getGuardrails() })
  const createGuardrail = useMutation({ mutationFn: (data: any) => llmHubAPI.createGuardrail(data), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['guardrails'] }); setShowCreateModal(false); success('Guardrail created') } })
  const deleteGuardrail = useMutation({ mutationFn: (id: string) => llmHubAPI.deleteGuardrail(id), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['guardrails'] }) })
  const testGuardrail = useMutation({ mutationFn: ({ id, text }: { id: string; text: string }) => llmHubAPI.testGuardrail(id, text), onSuccess: (res) => setTestResult(res.data) })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Shield size={20} className="text-red-400" /> Guardrail Editor</h1><p className="text-xs text-slate-400">Content safety rules for LLM inputs/outputs</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> Add Rule</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {guardrails?.map(g => (<TerminalCard key={g.id} className="p-3"><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><h3 className="font-bold text-white">{g.name}</h3><Badge variant={g.type}>{g.type}</Badge><Badge variant={g.action === 'block' ? 'danger' : 'warning'}>{g.action}</Badge><Badge variant={g.severity === 'high' ? 'danger' : 'secondary'}>{g.severity}</Badge></div><div className="text-xs text-slate-400 mt-1">Pattern: {g.pattern}</div></div><div className="flex gap-2"><Button size="sm" variant="ghost" onClick={() => testGuardrail.mutate({ id: g.id, text: testText })}><TestTube size={14} /></Button><button onClick={() => deleteGuardrail.mutate(g.id)}><Trash2 size={14} className="text-red-400" /></button></div></div></TerminalCard>))}
        <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Test Guardrail</h3><Textarea rows={3} value={testText} onChange={e => setTestText(e.target.value)} placeholder="Enter text to test..." /><div className="flex gap-2 mt-2"><Button size="sm" variant="ghost" onClick={() => testGuardrail.mutate({ id: guardrails[0]?.id, text: testText })}>Test</Button></div>{testResult && <div className="mt-3 p-2 rounded bg-white/5 text-sm"><div className="text-neon-mint">Result: {testResult.flagged ? '⚠️ Flagged' : '✅ Safe'}</div>{testResult.matches && <div className="text-xs text-slate-400">Matches: {testResult.matches.join(', ')}</div>}</div>}</TerminalCard>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Guardrail Rule"><div className="space-y-4"><Input label="Name" value={newName} onChange={e => setNewName(e.target.value)} /><Select label="Type" options={[{ value: 'input', label: 'Input' }, { value: 'output', label: 'Output' }, { value: 'both', label: 'Both' }]} value={type} onChange={e => setType(e.target.value)} /><Select label="Action" options={[{ value: 'block', label: 'Block' }, { value: 'warn', label: 'Warn' }, { value: 'replace', label: 'Replace' }]} value={action} onChange={e => setAction(e.target.value)} /><Input label="Pattern (regex)" value={pattern} onChange={e => setPattern(e.target.value)} placeholder="violence|hate|pii" /><Select label="Severity" options={[{ value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }]} value={severity} onChange={e => setSeverity(e.target.value)} /><Button onClick={() => createGuardrail.mutate({ name: newName, type, action, pattern, severity })}>Create</Button></div></Modal>
    </div>
  )
}