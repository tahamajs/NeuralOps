// src/features/inference/pages/ModelDistillationWizard.tsx
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { inferenceAPI } from '../services/api'
import { DistillationStep } from '../components/DistillationStep'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Progress } from '../../../components/ui/Progress'
import { Spinner } from '../../../components/ui/Spinner'
import { FlaskConical, ArrowRight, ArrowLeft, Rocket } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ModelDistillationWizard({ onClose }: { onClose?: () => void }) {
  const [step, setStep] = useState(1)
  const [config, setConfig] = useState({
    teacher_model_id: '',
    student_model_id: '',
    temperature: 4.0,
    alpha: 0.5,
    epochs: 10,
    batch_size: 32,
    learning_rate: 1e-4,
    dataset_path: '',
  })
  const [runId, setRunId] = useState<string | null>(null)
  const { success, error } = useToast()

  const startDistillation = useMutation({
    mutationFn: () => inferenceAPI.startDistillation(config),
    onSuccess: (res) => { setRunId(res.data.id); success('Distillation started') },
  })

  const nextStep = () => setStep(s => s + 1)
  const prevStep = () => setStep(s => s - 1)

  if (runId) return <DistillationProgress runId={runId} onClose={onClose} />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><FlaskConical size={20} className="text-neon-mint" /> Model Distillation Wizard</h1><p className="text-xs text-slate-400">Compress large models into efficient small ones</p></div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-6">
          <div className="flex gap-2"><div className={`flex-1 h-1 rounded-full ${step >= 1 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 2 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 3 ? 'bg-neon-blue' : 'bg-white/10'}`} /></div>
          <TerminalCard className="p-6">
            {step === 1 && (<div className="space-y-4"><Input label="Teacher Model ID" value={config.teacher_model_id} onChange={e => setConfig({ ...config, teacher_model_id: e.target.value })} /><Input label="Student Model ID" value={config.student_model_id} onChange={e => setConfig({ ...config, student_model_id: e.target.value })} /><Input label="Dataset Path" value={config.dataset_path} onChange={e => setConfig({ ...config, dataset_path: e.target.value })} /><div className="flex justify-end"><Button onClick={nextStep} disabled={!config.teacher_model_id || !config.student_model_id}><ArrowRight size={14} className="ml-1" /> Next</Button></div></div>)}
            {step === 2 && (<div className="space-y-4"><Input label="Temperature" type="number" step="0.1" value={config.temperature} onChange={e => setConfig({ ...config, temperature: parseFloat(e.target.value) })} /><Input label="Alpha (hard/soft loss)" type="number" step="0.05" value={config.alpha} onChange={e => setConfig({ ...config, alpha: parseFloat(e.target.value) })} /><Input label="Epochs" type="number" value={config.epochs} onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) })} /><Input label="Batch Size" type="number" value={config.batch_size} onChange={e => setConfig({ ...config, batch_size: parseInt(e.target.value) })} /><Input label="Learning Rate" type="number" step="1e-5" value={config.learning_rate} onChange={e => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })} /><div className="flex justify-between"><Button variant="ghost" onClick={prevStep}><ArrowLeft size={14} className="mr-1" /> Back</Button><Button onClick={nextStep}>Next <ArrowRight size={14} className="ml-1" /></Button></div></div>)}
            {step === 3 && (<div className="space-y-4"><div className="p-3 bg-white/5 rounded"><div className="text-sm text-slate-400">Summary</div><div className="mt-2 space-y-1"><div>Teacher: {config.teacher_model_id}</div><div>Student: {config.student_model_id}</div><div>Temperature: {config.temperature}</div><div>Epochs: {config.epochs}</div></div></div><div className="flex justify-between"><Button variant="ghost" onClick={prevStep}><ArrowLeft size={14} className="mr-1" /> Back</Button><Button variant="primary" onClick={() => startDistillation.mutate()} disabled={startDistillation.isPending}><Rocket size={14} className="mr-1" /> Start Distillation</Button></div></div>)}
          </TerminalCard>
        </div>
      </div>
    </div>
  )
}

// Internal component for monitoring progress
function DistillationProgress({ runId, onClose }: { runId: string; onClose?: () => void }) {
  const { data: status, isLoading } = useQuery({
    queryKey: ['distillation-status', runId],
    queryFn: () => inferenceAPI.getDistillationStatus(runId),
    refetchInterval: 2000,
  })

  if (isLoading) return <Spinner />
  return <TerminalCard className="p-6 text-center"><Progress value={status?.progress || 0} className="h-2 mb-2" /><p>{status?.message || 'Distillation in progress...'}</p><Button className="mt-4" onClick={() => window.location.reload()}>Refresh</Button></TerminalCard>
}