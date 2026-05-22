// src/features/liveDashboards/pages/LiveDashboardList.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { liveDashboardsAPI } from '../services/api'
import { DashboardCard } from '../components/DashboardCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { LayoutDashboard, Plus, Search } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function LiveDashboardList({ onClose }: { onClose?: () => void }) {
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: dashboards, isLoading } = useQuery({
    queryKey: ['dashboards'],
    queryFn: () => liveDashboardsAPI.getDashboards(),
  })

  const createDashboard = useMutation({
    mutationFn: () => liveDashboardsAPI.createDashboard({ name: newName, description: newDesc, is_public: false, refresh_interval: 5, widgets: [] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboards'] })
      setShowCreateModal(false)
      setNewName('')
      setNewDesc('')
      success('Dashboard created')
    },
  })

  const filtered = dashboards?.filter(d => d.name.toLowerCase().includes(search.toLowerCase())) || []

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <LayoutDashboard size={20} className="text-neon-mint" /> Live Dashboards
          </h1>
          <p className="text-xs text-slate-400">Real‑time experiment monitoring dashboards</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Dashboard</Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <Input placeholder="Search dashboards..." value={search} onChange={e => setSearch(e.target.value)} className="max-w-sm" />
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(d => <DashboardCard key={d.id} dashboard={d} />)}
        </div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create New Dashboard">
        <div className="space-y-4">
          <Input label="Dashboard Name" value={newName} onChange={e => setNewName(e.target.value)} />
          <Input label="Description (optional)" value={newDesc} onChange={e => setNewDesc(e.target.value)} />
          <Button onClick={() => createDashboard.mutate()} disabled={!newName.trim()}>Create</Button>
        </div>
      </Modal>
    </div>
  )
}