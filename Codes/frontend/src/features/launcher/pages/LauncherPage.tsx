import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { ModelPicker } from '../components/ModelPicker'
import { TaskPicker } from '../components/TaskPicker'
import { AttentionPicker } from '../components/AttentionPicker'
import { MethodPicker } from '../components/MethodPicker'
import { HyperparameterFields } from '../components/HyperparameterFields'
import { PresetSelector } from '../components/PresetSelector'
import { LaunchButton } from '../components/LaunchButton'

type JobType = 'training' | 'evaluation' | 'compression'

const initialForm = {
  name: '',
  model_name: 'distilbert-base-uncased',
  task_name: 'mrpc',
  attention_type: 'full',
  method: 'baseline',
  compression_rank: 64,
  epochs: 3,
  batch_size: 16,
  learning_rate: 2e-5,
  max_seq_length: 128,
  fp16: false,
  seed: 42,
}

export default function LauncherPage() {
  const [tab, setTab] = useState<JobType>('training')
  const [form, setForm] = useState(initialForm)
  const navigate = useNavigate()

  const mutation = useMutation({
    mutationFn: async () => {
      const endpoint =
        tab === 'training' ? ENDPOINTS.EXPERIMENTS.TRAIN :
        tab === 'evaluation' ? ENDPOINTS.EXPERIMENTS.EVALUATE :
        ENDPOINTS.EXPERIMENTS.COMPRESS
      const res = await api.post(endpoint, form)
      return res.data
    },
    onSuccess: (data) => navigate(`/experiments/${data.id}`),
  })

  const updateField = (key: string, value: any) => setForm((prev) => ({ ...prev, [key]: value }))

  return (
    <div className="space-y-8 max-w-4xl mx-auto animate-fade-in">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tighter">
          LAUNCH <span className="text-neon-blue">SEQUENCE</span>
        </h1>
        <p className="text-slate-400 text-sm mt-1">Configure and deploy new attention compression jobs.</p>
      </div>

      <div className="flex p-1 bg-white/5 rounded-xl w-fit">
        {(['training', 'evaluation', 'compression'] as JobType[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2 rounded-lg text-xs font-black uppercase tracking-widest transition-all ${
              tab === t ? 'bg-neon-blue text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <Card>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-6">Configuration</h2>
            <div className="space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ModelPicker value={form.model_name} onChange={(v) => updateField('model_name', v)} />
                <TaskPicker value={form.task_name} onChange={(v) => updateField('task_name', v)} />
              </div>
              <AttentionPicker value={form.attention_type} onChange={(v) => updateField('attention_type', v)} />
              <MethodPicker value={form.method} onChange={(v) => updateField('method', v)} />
              <HyperparameterFields values={form} onChange={updateField} />
              <LaunchButton loading={mutation.isPending} onClick={() => mutation.mutate()} />
            </div>
          </Card>
        </div>
        <div className="space-y-6">
          <Card>
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">Config Preview</h3>
            <pre className="text-xs text-slate-400 overflow-auto max-h-64 bg-white/5 p-3 rounded-lg scrollbar-thin">
              {JSON.stringify(form, null, 2)}
            </pre>
          </Card>
          <PresetSelector onApply={(preset) => setForm({ ...form, ...preset })} />
        </div>
      </div>
    </div>
  )
}