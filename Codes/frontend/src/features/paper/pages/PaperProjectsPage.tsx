import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { paperAPI } from '../services/api'
import { Card, Button, Spinner, Input, Modal } from '../../../components/ui'
import { Plus, GitBranch, Edit, Trash2, Download } from 'lucide-react'
import { useState } from 'react'

export default function PaperProjectsPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const queryClient = useQueryClient()

  const { data: projects, isLoading } = useQuery({
    queryKey: ['paper-projects'],
    queryFn: () => paperAPI.getProjects(),
  })

  const createProject = useMutation({
    mutationFn: () => paperAPI.createProject({ name: newProjectName, description: newProjectDesc }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper-projects'] })
      setShowCreateModal(false)
      setNewProjectName('')
      setNewProjectDesc('')
    },
  })

  const deleteProject = useMutation({
    mutationFn: (id: string) => paperAPI.deleteProject(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['paper-projects'] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Paper Projects</h1>
        <Button onClick={() => setShowCreateModal(true)}><Plus size={16} className="mr-1" /> New Project</Button>
      </div>
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects?.map((project: any) => (
          <Card key={project.id} hover>
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-bold text-white">{project.name}</h3>
                <p className="text-sm text-slate-400">{project.description || 'No description'}</p>
              </div>
              <div className="flex gap-1">
                <button className="p-1 rounded hover:bg-white/10" title="Git sync"><GitBranch size={14} /></button>
                <button className="p-1 rounded hover:bg-white/10" title="Edit"><Edit size={14} /></button>
                <button className="p-1 rounded hover:bg-red-500/20" title="Delete" onClick={() => deleteProject.mutate(project.id)}><Trash2 size={14} className="text-red-400" /></button>
              </div>
            </div>
            <div className="flex justify-between mt-3 text-xs text-slate-500">
              <span>{project.collaborators?.length || 0} collaborators</span>
              <span>Updated {new Date(project.updated_at).toLocaleDateString()}</span>
            </div>
            <Link to={`/paper/editor/${project.id}`} className="mt-3 block"><Button size="sm" variant="ghost" className="w-full">Open Project →</Button></Link>
          </Card>
        ))}
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create New Project">
        <div className="space-y-4">
          <Input label="Project Name" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} />
          <Input label="Description" value={newProjectDesc} onChange={(e) => setNewProjectDesc(e.target.value)} />
          <Button onClick={() => createProject.mutate()} disabled={!newProjectName || createProject.isPending}>Create Project</Button>
        </div>
      </Modal>
    </div>
  )
}