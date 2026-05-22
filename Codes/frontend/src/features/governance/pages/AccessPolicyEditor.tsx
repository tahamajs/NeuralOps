// src/features/governance/pages/AccessPolicyEditor.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { governanceAPI } from '../services/api'
import { PolicyMatrix } from '../components/PolicyMatrix'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Switch } from '../../../components/ui/Switch'
import { Shield, Plus, Trash2, Edit, Copy, Play } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function AccessPolicyEditor({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingPolicy, setEditingPolicy] = useState<any>(null)
  const [testContext, setTestContext] = useState('')
  const [testResult, setTestResult] = useState<any>(null)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: policies, isLoading } = useQuery({
    queryKey: ['access-policies'],
    queryFn: () => governanceAPI.getAccessPolicies(),
  })

  const createPolicy = useMutation({
    mutationFn: (data: any) => governanceAPI.createAccessPolicy(data),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['access-policies'] }); setShowCreateModal(false); success('Policy created') },
  })

  const updatePolicy = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => governanceAPI.updateAccessPolicy(id, data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['access-policies'] }),
  })

  const deletePolicy = useMutation({
    mutationFn: (id: string) => governanceAPI.deleteAccessPolicy(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['access-policies'] }),
  })

  const testPolicy = useMutation({
    mutationFn: () => governanceAPI.testAccessPolicy(editingPolicy?.id, JSON.parse(testContext)),
    onSuccess: (res) => setTestResult(res.data),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Shield size={20} className="text-neon-mint" /> Access Policy Editor</h1><p className="text-xs text-slate-400">RBAC & ABAC policy management</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Policy</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4"><div className="grid md:grid-cols-3 gap-4"><div className="space-y-3">{policies?.map(p => (<TerminalCard key={p.id} className="p-3"><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><span className="font-mono text-xs text-slate-400">{p.id.slice(0, 8)}</span><span className="font-bold text-white">{p.name}</span><Badge variant={p.effect === 'allow' ? 'success' : 'danger'}>{p.effect}</Badge><Badge variant={p.enabled ? 'success' : 'secondary'}>{p.enabled ? 'Enabled' : 'Disabled'}</Badge></div><div className="text-xs text-slate-400 mt-1 line-clamp-2">{p.description}</div><div className="flex gap-2 mt-2"><Badge variant="info">Priority: {p.priority}</Badge></div></div><div className="flex gap-1"><button onClick={() => { setEditingPolicy(p); setTestContext(''); setTestResult(null) }}><Edit size={14} className="text-slate-400" /></button><button onClick={() => deletePolicy.mutate(p.id)}><Trash2 size={14} className="text-red-400" /></button></div></div></TerminalCard>))}</div>
          <div className="lg:col-span-2">{editingPolicy ? (<div className="space-y-4"><TerminalCard className="p-4"><Input label="Policy Name" value={editingPolicy.name} onChange={e => setEditingPolicy({ ...editingPolicy, name: e.target.value })} /><Input label="Description" value={editingPolicy.description} onChange={e => setEditingPolicy({ ...editingPolicy, description: e.target.value })} /><Select label="Effect" options={[{ value: 'allow', label: 'Allow' }, { value: 'deny', label: 'Deny' }]} value={editingPolicy.effect} onChange={e => setEditingPolicy({ ...editingPolicy, effect: e.target.value })} /><div className="flex justify-between items-center"><span className="text-sm text-slate-400">Enabled</span><Switch checked={editingPolicy.enabled} onChange={checked => setEditingPolicy({ ...editingPolicy, enabled: checked })} /></div></TerminalCard>
            <PolicyMatrix policy={editingPolicy} onChange={setEditingPolicy} />
            <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Policy Test Simulator</h3><textarea rows={3} className="w-full bg-white/5 border border-white/10 rounded p-2 text-sm font-mono" placeholder='{"user": "alice", "resource": "dataset-123", "action": "read"}' value={testContext} onChange={e => setTestContext(e.target.value)} /><div className="flex gap-2 mt-2"><Button size="sm" variant="ghost" onClick={() => testPolicy.mutate()}>Test Policy</Button></div>{testResult && <div className="mt-3 p-2 rounded bg-white/5 text-sm"><div className="text-neon-mint">Result: {testResult.allowed ? '✅ Allowed' : '❌ Denied'}</div><div className="text-xs text-slate-400">Matched rule: {testResult.matched_rule}</div></div>}</TerminalCard>
            <div className="flex justify-end gap-2"><Button onClick={() => updatePolicy.mutate({ id: editingPolicy.id, data: editingPolicy })}>Save Changes</Button><Button variant="ghost" onClick={() => setEditingPolicy(null)}>Cancel</Button></div>
          </div>) : <div className="text-center text-slate-500 py-12">Select a policy to edit or create a new one</div>}</div>
      </div></div>
    </div>
  )
}