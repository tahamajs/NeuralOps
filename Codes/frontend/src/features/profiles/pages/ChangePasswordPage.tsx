import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Input, Button } from '../../../components/ui'

export default function ChangePasswordPage() {
  const [oldPw, setOldPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [confirm, setConfirm] = useState('')
  const change = useMutation({
    mutationFn: () => api.post('/auth/change-password/', { old_password: oldPw, new_password: newPw }),
    onSuccess: () => alert('Password changed'),
  })
  return (
    <Card className="max-w-md mx-auto">
      <h1 className="text-xl font-bold text-white mb-4">Change Password</h1>
      <Input type="password" label="Current Password" value={oldPw} onChange={e => setOldPw(e.target.value)} />
      <Input type="password" label="New Password" value={newPw} onChange={e => setNewPw(e.target.value)} />
      <Input type="password" label="Confirm" value={confirm} onChange={e => setConfirm(e.target.value)} />
      <Button onClick={() => change.mutate()} disabled={!oldPw || !newPw || newPw !== confirm}>Update</Button>
    </Card>
  )
}