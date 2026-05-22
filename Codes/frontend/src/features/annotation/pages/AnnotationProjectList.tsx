// src/features/annotation/pages/AnnotationProjectList.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { annotationAPI } from '../services/api'
import { ProjectCard } from '../components/ProjectCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { FolderKanban, Plus, Search, Filter } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

const ANNOTATION_TYPES = [
  { value: 'bounding_box', label: 'Bounding Box (Object Detection)' },
  { value: 'polygon', label: 'Polygon (Segmentation)' },
  { value: 'classification', label: 'Image Classification' },
  { value: 'ner', label: 'Named Entity Recognition' },
]

export default function AnnotationProjectList({ onClose }: { onClose?: () => void }) {
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newProject, setNewProject] = useState({ name: '', description: '', type: 'bounding_box' })
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: projects, isLoading } = useQuery({
    queryKey: ['annotation-projects'],
    queryFn: () => annotationAPI.getProjects(),
  })

  const createProject = useMutation({
    mutationFn: () => annotationAPI.createProject(newProject),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotation-projects'] })
      setShowCreateModal(false)
      setNewProject({ name: '', description: '', type: 'bounding_box' })
      success('Project created')
    },
  })

  const filtered = projects?.filter(p => p.name.toLowerCase().includes(search.toLowerCase())) || []

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2"><FolderKanban size={20} className="text-neon-mint" /> Annotation Projects</h1>
          <p className="text-xs text-slate-400">Manage data labeling workflows and quality reviews</p>
        </div>
        <div className="flex gap-2">
          <div className="relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search projects..." className="pl-9 w-64" /></div>
          <Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Project</Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">{filtered.map(p => <ProjectCard key={p.id} project={p} />)}</div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Annotation Project">
        <div className="space-y-4"><Input label="Project Name" value={newProject.name} onChange={e => setNewProject({ ...newProject, name: e.target.value })} /><Input label="Description" value={newProject.description} onChange={e => setNewProject({ ...newProject, description: e.target.value })} /><Select label="Annotation Type" options={ANNOTATION_TYPES} value={newProject.type} onChange={e => setNewProject({ ...newProject, type: e.target.value })} /><Button onClick={() => createProject.mutate()} disabled={!newProject.name}>Create</Button></div>
      </Modal>
    </div>
  )
}