import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import ModelPicker from '../components/ModelPicker';
import TaskPicker from '../components/TaskPicker';
import AttentionPicker from '../components/AttentionPicker';
import MethodPicker from '../components/MethodPicker';
import HyperparameterFields from '../components/HyperparameterFields';
import PresetSelector from '../components/PresetSelector';
import ConfigPreviewPanel from '../components/ConfigPreviewPanel';
import LaunchButton from '../components/LaunchButton';

export default function LauncherPage() {
  const [tab, setTab] = useState('training');
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
  });

  const navigate = useNavigate();
  const mutation = useMutation({
    mutationFn: async () => {
      const endpoint = tab === 'training' ? '/train/' : tab === 'evaluation' ? '/evaluate/' : '/compress/';
      const res = await api.post(endpoint, form);
      return res.data;
    },
    onSuccess: (data) => {
      navigate(`/experiments/${data.id}`);
    },
  });

  const updateField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Experiment Launcher</h1>
        <p className="text-slate-500">Configure and deploy new attention compression jobs to the cluster.</p>
      </div>

      <div className="flex p-1 bg-slate-200/50 backdrop-blur rounded-xl w-fit">
        {['training', 'evaluation', 'compression'].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all ${
              tab === t ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="card space-y-6">
            <div className="flex items-center gap-2 pb-4 border-b border-slate-100">
              <div className="w-2 h-2 rounded-full bg-blue-600" />
              <h2 className="font-bold text-slate-800 uppercase tracking-wider text-xs">
                {tab === 'training'
                  ? 'Training Configuration'
                  : tab === 'evaluation'
                  ? 'Evaluation Configuration'
                  : 'Compression Configuration'}
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {tab === 'training' && (
                <>
                  <ModelPicker value={form.model_name} onChange={(v) => updateField('model_name', v)} />
                  <TaskPicker value={form.task_name} onChange={(v) => updateField('task_name', v)} />
                  <div className="md:col-span-2">
                    <AttentionPicker value={form.attention_type} onChange={(v) => updateField('attention_type', v)} />
                  </div>
                  <MethodPicker value={form.method} onChange={(v) => updateField('method', v)} />
                  <HyperparameterFields values={form} onChange={updateField} />
                </>
              )}

              {tab === 'compression' && (
                <>
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-1">Compression Method</label>
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
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-1">Target Rank</label>
                    <input
                      type="number"
                      className="w-full border rounded px-3 py-2"
                      value={form.target_rank}
                      onChange={(e) => updateField('target_rank', parseInt(e.target.value, 10))}
                    />
                  </div>
                </>
              )}

              {tab === 'evaluation' && (
                <div className="md:col-span-2 text-slate-500 text-sm">
                  Evaluation uses the same model and task settings. Add specific fields later.
                </div>
              )}
            </div>

            <div className="pt-6">
              <LaunchButton
                onClick={() => mutation.mutate()}
                loading={mutation.isPending}
                disabled={!form.name}
              />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <ConfigPreviewPanel config={form} />
          <PresetSelector onApply={(preset) => setForm({ ...form, ...preset })} />
        </div>
      </div>
    </div>
  );
}