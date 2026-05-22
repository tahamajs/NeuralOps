// src/features/annotation/pages/LabelingInterface.tsx
import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { annotationAPI } from '../services/api'
import { BoundingBoxCanvas } from '../components/BoundingBoxCanvas'
import { LabelSelector } from '../components/LabelSelector'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { ChevronLeft, ChevronRight, Save, CheckCircle, Layers, MousePointer } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function LabelingInterface({ onClose }: { onClose?: () => void }) {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const [currentIndex, setCurrentIndex] = useState(0)
  const [selectedLabelId, setSelectedLabelId] = useState('')
  const [annotations, setAnnotations] = useState<any[]>([])
  const { success, error } = useToast()
  const queryClient = useQueryClient()

  const { data: project, isLoading: projLoading } = useQuery({
    queryKey: ['annotation-project', projectId],
    queryFn: () => annotationAPI.getProject(projectId!),
    enabled: !!projectId,
  })

  const { data: itemsData, isLoading: itemsLoading } = useQuery({
    queryKey: ['annotation-items', projectId],
    queryFn: () => annotationAPI.getItems(projectId!, { status: 'pending' }),
    enabled: !!projectId,
  })

  const saveAnnotation = useMutation({
    mutationFn: (annotation: any) => annotationAPI.saveAnnotation(itemsData?.results[currentIndex]?.id, annotation),
    onSuccess: () => success('Annotation saved'),
  })

  const submitItem = useMutation({
    mutationFn: () => annotationAPI.submitItem(itemsData?.results[currentIndex]?.id),
    onSuccess: () => {
      success('Item submitted')
      if (currentIndex + 1 < (itemsData?.results.length || 0)) setCurrentIndex(prev => prev + 1)
      else success('All items completed!')
    },
  })

  const currentItem = itemsData?.results[currentIndex]

  const handleSave = () => {
    if (annotations.length === 0) return error('No annotations to save')
    saveAnnotation.mutate({ annotations })
  }

  const handleSubmit = () => {
    if (annotations.length === 0) return error('Please add at least one annotation')
    submitItem.mutate()
  }

  if (projLoading || itemsLoading) return <Spinner />
  if (!project) return <div className="p-4 text-red-400">Project not found</div>

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white">{project.name}</h1><p className="text-xs text-slate-400">Item {currentIndex + 1} of {itemsData?.count || 0} · {project.type}</p></div>
        <div className="flex gap-2">
          <Badge variant="info" className="capitalize">{project.type}</Badge>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-4 gap-4 h-full">
          <div className="lg:col-span-3 bg-black/20 rounded-lg flex items-center justify-center relative">
            {currentItem?.data.image_url ? (
              <BoundingBoxCanvas imageUrl={currentItem.data.image_url} onAnnotationsChange={setAnnotations} labels={project.labels} />
            ) : currentItem?.data.text ? (
              <TerminalCard className="p-4 w-full"><h3 className="text-sm font-bold text-white mb-2">Text Annotation</h3><p className="text-slate-300">{currentItem.data.text}</p><div className="mt-4"><LabelSelector labels={project.labels} selected={selectedLabelId} onSelect={setSelectedLabelId} /></div></TerminalCard>
            ) : <div className="text-slate-500">Unsupported data type</div>}
          </div>
          <div className="space-y-4">
            <TerminalCard className="p-4"><LabelSelector labels={project.labels} selected={selectedLabelId} onSelect={setSelectedLabelId} /></TerminalCard>
            <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Annotations ({annotations.length})</h3><div className="space-y-1 max-h-48 overflow-y-auto">{annotations.map((a, i) => (<div key={i} className="text-xs text-slate-300 flex justify-between"><span>{a.label_name}</span><button onClick={() => setAnnotations(prev => prev.filter((_, idx) => idx !== i))}>✕</button></div>))}</div></TerminalCard>
            <TerminalCard className="p-4"><div className="flex gap-2"><Button onClick={handleSave} className="flex-1"><Save size={14} className="mr-1" /> Save</Button><Button variant="primary" onClick={handleSubmit} className="flex-1"><CheckCircle size={14} className="mr-1" /> Submit</Button></div></TerminalCard>
            <div className="flex justify-between"><Button variant="ghost" disabled={currentIndex === 0} onClick={() => setCurrentIndex(prev => prev - 1)}><ChevronLeft size={16} /> Previous</Button><Button variant="ghost" disabled={currentIndex + 1 >= (itemsData?.results.length || 0)} onClick={() => setCurrentIndex(prev => prev + 1)}>Next <ChevronRight size={16} /></Button></div>
          </div>
        </div>
      </div>
    </div>
  )
}