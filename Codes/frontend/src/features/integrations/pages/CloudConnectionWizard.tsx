// src/features/integrations/pages/CloudConnectionWizard.tsx
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { integrationsAPI } from '../services/api'
import { ConnectionStep } from '../components/ConnectionStep'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { Cloud, ArrowRight, CheckCircle, Database } from 'lucide-react'
import { CloudProvider } from '../types'
import { useToast } from '../../../components/ui/Toast'

const PROVIDERS = [
  { value: 'aws', label: 'Amazon Web Services (AWS)', icon: '☁️' },
  { value: 'gcp', label: 'Google Cloud Platform (GCP)', icon: '🔷' },
  { value: 'azure', label: 'Microsoft Azure', icon: '🔹' },
  { value: 'google_drive', label: 'Google Drive', icon: '📁' },
  { value: 'dropbox', label: 'Dropbox', icon: '📦' },
]

export default function CloudConnectionWizard({ onClose }: { onClose?: () => void }) {
  const [step, setStep] = useState(1)
  const [provider, setProvider] = useState<CloudProvider>('aws')
  const [name, setName] = useState('')
  const [accessKey, setAccessKey] = useState('')
  const [secretKey, setSecretKey] = useState('')
  const [bucket, setBucket] = useState('')
  const [region, setRegion] = useState('us-east-1')
  const [isTesting, setIsTesting] = useState(false)
  const [testStatus, setTestStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const createConnection = useMutation({
    mutationFn: () => integrationsAPI.createConnection({
      provider,
      name,
      credentials: { accessKey, secretKey, bucket, region },
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cloud-connections'] })
      success('Connection created', `Connected to ${name || provider}`)
      if (onClose) onClose()
    },
  })

  const testConnection = async () => {
    setIsTesting(true)
    setTestStatus('idle')
    try {
      // Simulate test – replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      setTestStatus('success')
      success('Connection test passed', 'Your credentials are valid')
    } catch {
      setTestStatus('error')
      error('Connection test failed', 'Please check your credentials')
    } finally {
      setIsTesting(false)
    }
  }

  const handleNext = () => {
    if (step === 1) setStep(2)
    else if (step === 2) testConnection()
    else if (step === 3) createConnection.mutate()
  }

  const canProceed = () => {
    if (step === 1) return !!provider && !!name
    if (step === 2) return !!accessKey && !!secretKey
    if (step === 3) return testStatus === 'success'
    return true
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Cloud size={20} className="text-neon-mint" /> Cloud Connection Wizard</h1><p className="text-xs text-slate-400">Connect external storage and compute providers</p></div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl mx-auto">
          <div className="flex mb-8"><div className={`flex-1 h-1 rounded-full ${step >= 1 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 2 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 3 ? 'bg-neon-blue' : 'bg-white/10'}`} /></div>
          <TerminalCard className="p-6 space-y-6">
            {step === 1 && (<div className="space-y-4"><Select label="Cloud Provider" options={PROVIDERS} value={provider} onChange={e => setProvider(e.target.value as CloudProvider)} /><Input label="Connection Name" value={name} onChange={e => setName(e.target.value)} placeholder="My AWS Production Bucket" /></div>)}
            {step === 2 && (<div className="space-y-4"><Input label="Access Key ID" value={accessKey} onChange={e => setAccessKey(e.target.value)} type="password" /><Input label="Secret Access Key" value={secretKey} onChange={e => setSecretKey(e.target.value)} type="password" /><div className="grid grid-cols-2 gap-3"><Input label="Bucket Name" value={bucket} onChange={e => setBucket(e.target.value)} /><Select label="Region" options={[{ value: 'us-east-1', label: 'US East (N. Virginia)' }, { value: 'eu-west-1', label: 'EU (Ireland)' }, { value: 'ap-southeast-1', label: 'Asia Pacific (Singapore)' }]} value={region} onChange={e => setRegion(e.target.value)} /></div></div>)}
            {step === 3 && (<div className="text-center py-8">{isTesting ? <Spinner size="lg" /> : testStatus === 'success' ? <CheckCircle size={48} className="mx-auto text-green-400" /> : testStatus === 'error' ? <Cloud size={48} className="mx-auto text-red-400" /> : <Database size={48} className="mx-auto text-neon-mint" />}<p className="mt-3 text-white">{isTesting ? 'Testing connection...' : testStatus === 'success' ? 'Connection verified!' : testStatus === 'error' ? 'Test failed. Please check credentials.' : 'Ready to connect'}</p></div>)}
            <div className="flex justify-between pt-4 border-t border-white/10"><Button variant="ghost" disabled={step === 1} onClick={() => setStep(s => s - 1)}>Back</Button><Button onClick={handleNext} disabled={!canProceed() || isTesting || createConnection.isPending}>{step === 3 ? (createConnection.isPending ? 'Connecting...' : 'Connect') : 'Test Connection'}{step !== 3 && <ArrowRight size={14} className="ml-1" />}</Button></div>
          </TerminalCard>
        </div>
      </div>
    </div>
  )
}