import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Input } from '../../../components/ui/Input'
import { Textarea } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import { GripVertical, Trash2, ChevronUp, ChevronDown } from 'lucide-react'

export function StepEditor({ step, index, onUpdate, onRemove, onMoveUp, onMoveDown }: any) {
  return (
    <TerminalCard className="p-3">
      <div className="flex gap-3">
        <div className="cursor-grab text-slate-400 hover:text-white"><GripVertical size={16} /></div>
        <div className="flex-1 space-y-2"><Input value={step.title} onChange={e => onUpdate(step.id, { title: e.target.value })} placeholder="Step title" /><Textarea rows={2} value={step.description} onChange={e => onUpdate(step.id, { description: e.target.value })} placeholder="Detailed instructions (Markdown supported)" /><Select options={[{ value: 'text', label: 'Text' }, { value: 'image', label: 'Image' }, { value: 'video', label: 'Video' }, { value: 'resource', label: 'Resource' }, { value: 'decision', label: 'Decision Point' }]} value={step.type} onChange={e => onUpdate(step.id, { type: e.target.value })} className="w-32" /></div>
        <div className="flex flex-col gap-1"><button onClick={onMoveUp} className="p-1 rounded hover:bg-white/10"><ChevronUp size={14} /></button><button onClick={onMoveDown} className="p-1 rounded hover:bg-white/10"><ChevronDown size={14} /></button><button onClick={onRemove} className="p-1 rounded hover:bg-red-500/20"><Trash2 size={14} className="text-red-400" /></button></div>
      </div>
    </TerminalCard>
  )
}