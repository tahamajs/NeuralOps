// src/features/integrations/pages/WebhookListPage.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { integrationsAPI } from '../services/api'
import { WebhookTable } from '../components/WebhookTable'
import { WebhookForm } from '../components/WebhookForm'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Webhook, Plus, RefreshCw, Activity } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function WebhookListPage({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: webhooks, isLoading, refetch } = useQuery({
    queryKey: ['webhooks'],
    queryFn: () => integrationsAPI.getWebhooks(),
  })

  const deleteWebhook = useMutation({
    mutationFn: (id: string) => integrationsAPI.deleteWebhook(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['webhooks'] })
      success('Webhook deleted')
    },
  })

  const testWebhook = useMutation({
    mutationFn: (id: string) => integrationsAPI.testWebhook(id),
    onSuccess: () => success('Test event sent'),
    onError: () => error('Failed to send test event'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Webhook size={20} className="text-neon-mint" /> Webhook Management</h1><p className="text-xs text-slate-400">Configure outgoing event notifications</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> Create Webhook</Button><Button size="sm" variant="ghost" onClick={() => refetch()}><RefreshCw size={14} /></Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4"><WebhookTable webhooks={webhooks || []} onEdit={setEditingWebhook} onDelete={(id) => deleteWebhook.mutate(id)} onTest={(id) => testWebhook.mutate(id)} /></div>
      <Modal open={showCreateModal || !!editingWebhook} onClose={() => { setShowCreateModal(false); setEditingWebhook(null) }} title={editingWebhook ? 'Edit Webhook' : 'Create Webhook'}>
        <WebhookForm initialData={editingWebhook || undefined} onSuccess={() => { setShowCreateModal(false); setEditingWebhook(null); queryClient.invalidateQueries({ queryKey: ['webhooks'] }) }} />
      </Modal>
    </div>
  )
}