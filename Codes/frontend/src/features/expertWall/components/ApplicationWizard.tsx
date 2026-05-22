import { useState } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Textarea } from '../../../components/ui/Input'
import { useToast } from '../../../components/ui/Toast'

interface ApplicationWizardProps {
  onSubmit: (data: any) => void
  isSubmitting: boolean
}

export function ApplicationWizard({ onSubmit, isSubmitting }: ApplicationWizardProps) {
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState({ cover_letter: '', proposed_approach: '', expected_timeline: '' })

  const nextStep = () => setStep(s => s + 1)
  const prevStep = () => setStep(s => s - 1)

  const handleSubmit = () => onSubmit(formData)

  return (
    <TerminalCard className="p-6 max-w-2xl mx-auto">
      <div className="flex gap-2 mb-6"><div className={`flex-1 h-1 rounded-full ${step >= 1 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 2 ? 'bg-neon-blue' : 'bg-white/10'}`} /><div className={`flex-1 h-1 rounded-full ${step >= 3 ? 'bg-neon-blue' : 'bg-white/10'}`} /></div>
      {step === 1 && (<div className="space-y-4"><h2 className="text-lg font-bold text-white">Cover Letter</h2><Textarea rows={6} value={formData.cover_letter} onChange={e => setFormData({ ...formData, cover_letter: e.target.value })} placeholder="Explain why you're the right fit..." /><div className="flex justify-end"><Button onClick={nextStep} disabled={!formData.cover_letter.trim()}>Next →</Button></div></div>)}
      {step === 2 && (<div className="space-y-4"><h2 className="text-lg font-bold text-white">Proposed Approach</h2><Textarea rows={6} value={formData.proposed_approach} onChange={e => setFormData({ ...formData, proposed_approach: e.target.value })} placeholder="Describe your methodology..." /><div className="flex justify-between"><Button variant="ghost" onClick={prevStep}>← Back</Button><Button onClick={nextStep} disabled={!formData.proposed_approach.trim()}>Next →</Button></div></div>)}
      {step === 3 && (<div className="space-y-4"><h2 className="text-lg font-bold text-white">Expected Timeline</h2><Input value={formData.expected_timeline} onChange={e => setFormData({ ...formData, expected_timeline: e.target.value })} placeholder="e.g., 3 months" /><div className="flex justify-between"><Button variant="ghost" onClick={prevStep}>← Back</Button><Button onClick={handleSubmit} disabled={!formData.expected_timeline.trim() || isSubmitting}>{isSubmitting ? 'Submitting...' : 'Submit Application →'}</Button></div></div>)}
    </TerminalCard>
  )
}