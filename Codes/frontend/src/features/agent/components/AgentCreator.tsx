import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Textarea } from '../../../components/ui/Textarea'
import { Bot } from 'lucide-react'

export default function AgentCreator() {
  const navigate = useNavigate()
  const [form, setForm] = useState({
    name: '',
    objective: 'Improve MRPC accuracy above 0.89',
    agent_type: 'deep_researcher',
    max_cycles: 50,
    max_llm_cost_cents: 500,
    metric_name: 'eval_accuracy',
    metric_direction: 'maximize',
  })

  const createMut = useMutation({
    mutationFn: () => api.post('/api/agents/sessions/', form),
    onSuccess: (res) => navigate(`/agents/${res.data.id}`),
  })

  return (
    <Card className="max-w-2xl mx-auto space-y-5">
      <div className="flex items-center gap-3">
        <Bot size={28} className="text-neon-blue" />
        <h1 className="text-2xl font-display font-bold text-white">Launch AI Agent</h1>
      </div>
      <p className="text-slate-400 text-sm">
        Give the agent a research objective. It will autonomously propose hypotheses,
        run experiments, analyze results, and iterate – all within your budget.
      </p>

      <Input label="Agent Name" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} placeholder="My Research Agent" />

      <Select
        label="Agent Type"
        options={[
          { value: 'deep_researcher', label: 'Deep Researcher – full experiment loop' },
          { value: 'hpo_agent', label: 'HPO Agent – hyperparameter optimization only' },
          { value: 'code_agent', label: 'Code Optimizer – edit code to improve metric' },
        ]}
        value={form.agent_type}
        onChange={e => setForm({ ...form, agent_type: e.target.value })}
      />

      <Textarea
        label="Research Objective"
        value={form.objective}
        onChange={e => setForm({ ...form, objective: e.target.value })}
        rows={3}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input label="Max Cycles" type="number" value={form.max_cycles} onChange={e => setForm({ ...form, max_cycles: parseInt(e.target.value) || 1 })} />
        <Input label="Budget (cents)" type="number" value={form.max_llm_cost_cents} onChange={e => setForm({ ...form, max_llm_cost_cents: parseInt(e.target.value) || 0 })} />
        <Input label="Target Metric" value={form.metric_name} onChange={e => setForm({ ...form, metric_name: e.target.value })} />
        <Select
          label="Direction"
          options={[{ value: 'maximize', label: 'Maximize' }, { value: 'minimize', label: 'Minimize' }]}
          value={form.metric_direction}
          onChange={e => setForm({ ...form, metric_direction: e.target.value })}
        />
      </div>

      <Button onClick={() => createMut.mutate()} disabled={createMut.isPending} loading={createMut.isPending} className="w-full">
        🚀 {createMut.isPending ? 'Launching...' : 'Launch Agent'}
      </Button>
    </Card>
  )
}