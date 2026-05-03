import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import ModelPicker from '../components/ModelPicker'
import TaskPicker from '../components/TaskPicker'
import AttentionPicker from '../components/AttentionPicker'
import MethodPicker from '../components/MethodPicker'
import HyperparameterFields from '../components/HyperparameterFields'
import PresetSelector from '../components/PresetSelector'
import ConfigPreviewPanel from '../components/ConfigPreviewPanel'
import LaunchButton from '../components/LaunchButton'

type Tab = 'training' | 'evaluation' | 'compression'

export default function LauncherPage() {
  const [tab, setTab] = useState<Tab>('training')
  const [form, setForm] = useState({
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
    target_rank: 64,
    compression_method: 'tt',
  })
  const navigate = useNavigate()

  const mutation = useMutation({
    mutationFn: async () => {
      const endpoint = tab === 'training' ? '/train/' : tab === 'evaluation' ? '/evaluate/' : '/compress/'
      const res = await api.post(endpoint, form)
      return res.data
    },
    onSuccess: (data) => {
      navigate(`/experiments/${data.id}`)
    },
  })

  const updateField = (key: string, value: any) => setForm((prev) => ({ ...prev, [key]: value }))

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Launcher</h1>
      <div className="flex gap-2 border-b pb-2">
        {(['training', 'evaluation', 'compression'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded ${tab === t ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tab === 'training' && (
          <>
            <ModelPicker value={form.model_name} onChange={(v) => updateField('model_name', v)} />
            <TaskPicker value={form.task_name} onChange={(v) => updateField('task_name', v)} />
            <AttentionPicker value={form.attention_type} onChange={(v) => updateField('attention_type', v)} />
            <MethodPicker value={form.method} onChange={(v) => updateField('method', v)} />
            <HyperparameterFields values={form} onChange={updateField} />
            <PresetSelector onApply={(preset) => setForm({ ...form, ...preset })} />
          </>
        )}
        {tab === 'evaluation' && (
          <>
            <div>
              <label className="block text-sm font-medium">Model Path</label>
              <input
                className="w-full border rounded px-3 py-2"
                value={form.model_name}
                onChange={(e) => updateField('model_name', e.target.value)}
              />
            </div>
            <TaskPicker value={form.task_name} onChange={(v) => updateField('task_name', v)} />
          </>
        )}
        {tab === 'compression' && (
          <>
            <div>
              <label className="block text-sm font-medium">Model Name</label>
              <input
                className="w-full border rounded px-3 py-2"
                value={form.model_name}
                onChange={(e) => updateField('model_name', e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium">Compression Method</label>
              <select
                className="w-full border rounded px-3 py-2"
                value={form.compression_method}
                onChange={(e) => updateField('compression_method', e.target.value)}
              >
                <option value="tt">Tensor Train (tt)</option>
                <option value="svd">SVD</option>
                <option value="soka">SoKA</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium">Target Rank</label>
              <input
                type="number"
                className="w-full border rounded px-3 py-2"
                value={form.target_rank}
                onChange={(e) => updateField('target_rank', parseInt(e.target.value))}
              />
            </div>
          </>
        )}
      </div>

      <ConfigPreviewPanel config={form} />
      <LaunchButton loading={mutation.isPending} onClick={() => mutation.mutate()} />
    </div>
  )
}