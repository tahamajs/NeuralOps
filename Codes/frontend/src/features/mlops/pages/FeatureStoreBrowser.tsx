// src/features/mlops/pages/FeatureStoreBrowser.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mlopsAPI } from '../services/api'
import { FeatureGroupCard } from '../components/FeatureGroupCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Database, Plus, Search } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function FeatureStoreBrowser({ onClose }: { onClose?: () => void }) {
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [newEntityKey, setNewEntityKey] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: groups, isLoading } = useQuery({
    queryKey: ['feature-groups'],
    queryFn: () => mlopsAPI.getFeatureGroups(),
  })

  const createGroup = useMutation({
    mutationFn: () => mlopsAPI.createFeatureGroup({ name: newGroupName, entity_key: newEntityKey, features: [] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feature-groups'] })
      setShowCreateModal(false)
      setNewGroupName('')
      setNewEntityKey('')
      success('Feature group created')
    },
  })

  const filtered = groups?.filter(g => g.name.toLowerCase().includes(search.toLowerCase())) || []

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Database size={20} className="text-neon-mint" /> Feature Store</h1><p className="text-xs text-slate-400">Manage feature groups for training and inference</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Group</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <Input placeholder="Search feature groups..." value={search} onChange={e => setSearch(e.target.value)} className="max-w-sm" />
        <div className="grid md:grid-cols-2 gap-4">
          {filtered.map(group => <FeatureGroupCard key={group.id} group={group} onRefresh={() => queryClient.invalidateQueries({ queryKey: ['feature-groups'] })} />)}
        </div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Feature Group">
        <div className="space-y-4">
          <Input label="Group Name" value={newGroupName} onChange={e => setNewGroupName(e.target.value)} placeholder="user_features" />
          <Input label="Entity Key" value={newEntityKey} onChange={e => setNewEntityKey(e.target.value)} placeholder="user_id" />
          <Button onClick={() => createGroup.mutate()} disabled={!newGroupName || !newEntityKey}>Create</Button>
        </div>
      </Modal>
    </div>
  )
}