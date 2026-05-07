import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'

export default function ChangePasswordPage() {
  const [form, setForm] = useState({ old_password: '', new_password: '' })
  const mutation = useMutation({
    mutationFn: async () => api.post('/api/auth/change-password/', form),
    onSuccess: () => alert('Password changed'),
  })

  return (
    <div className="space-y-6 animate-fade-in max-w-xl mx-auto">
      <h1 className="text-2xl font-black text-white">Change Password</h1>
      <Card>
        <div className="space-y-4">
          <Input label="Current Password" type="password" value={form.old_password} onChange={(e) => setForm({ ...form, old_password: e.target.value })} />
          <Input label="New Password" type="password" value={form.new_password} onChange={(e) => setForm({ ...form, new_password: e.target.value })} />
          <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>Change Password</Button>
        </div>
      </Card>
    </div>
  )
}