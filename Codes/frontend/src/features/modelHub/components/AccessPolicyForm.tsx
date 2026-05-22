import { useState } from 'react'
import { Modal } from '../../../components/ui/Modal'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import { modelHubAPI } from '../services/api'
import { useToast } from '../../../components/ui/Toast'

export function AccessPolicyForm({ open, onClose, modelId, onSuccess }: any) {
  const [userId, setUserId] = useState('')
  const [permission, setPermission] = useState('view')
  const [expiresAt, setExpiresAt] = useState('')
  const { success, error } = useToast()

  const handleSubmit = async () => {
    try {
      await modelHubAPI.grantAccess(modelId, userId, permission, expiresAt || undefined)
      success('Access granted')
      onSuccess()
      onClose()
    } catch { error('Failed to grant access') }
  }

  return (
    <Modal open={open} onClose={onClose} title="Grant Access">
      <div className="space-y-4">
        <Input label="User ID or Email" value={userId} onChange={e => setUserId(e.target.value)} />
        <Select label="Permission" options={[{ value: 'view', label: 'View' }, { value: 'download', label: 'Download' }, { value: 'deploy', label: 'Deploy' }, { value: 'manage', label: 'Manage' }]} value={permission} onChange={e => setPermission(e.target.value)} />
        <Input label="Expires At (optional)" type="datetime-local" value={expiresAt} onChange={e => setExpiresAt(e.target.value)} />
        <Button onClick={handleSubmit}>Grant</Button>
      </div>
    </Modal>
  )
}