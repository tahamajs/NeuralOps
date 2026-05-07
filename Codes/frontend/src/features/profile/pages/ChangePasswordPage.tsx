import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../../api/client'
import { Card } from '../../components/ui/Card'
import { Button } from '../../components/ui/Button'
import { Input } from '../../components/ui/Input'

export default function ChangePasswordPage() {
  const [form, setForm] = useState({ old_password: '', new_password: '' })
  const mutation = useMutation({
    mutationFn: async () => api.post('/api/auth/change-password/', form),
    onSuccess: () => {
      alert('Password changed')
      setForm({ old_password: '', new_password: '' })
    },
    onError: (err: any) => alert(err.response?.data?.error || 'Error'),
  })

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="w-full max-w-md p-8 space-y-6">
        <h1 className="text-2xl font-black text-white text-center">
          CHANGE <span className="text-neon-blue">PASSWORD</span>
        </h1>
        <Input label="Current Password" type="password" value={form.old_password} onChange={e => setForm({ ...form, old_password: e.target.value })} />
        <Input label="New Password (min 6 characters)" type="password" value={form.new_password} onChange={e => setForm({ ...form, new_password: e.target.value })} />
        <Button className="w-full" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
          {mutation.isPending ? 'Updating...' : 'Update Password'}
        </Button>
      </Card>
    </div>
  )
}