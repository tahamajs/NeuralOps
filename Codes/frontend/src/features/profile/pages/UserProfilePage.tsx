import { useQuery, useMutation } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'

export default function UserProfilePage() {
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '' })
  const { data: user, isLoading } = useQuery({
    queryKey: ['userInfo'],
    queryFn: async () => (await api.get('/api/auth/user/')).data,
  })

  useEffect(() => {
    if (user) setForm({ first_name: user.first_name || '', last_name: user.last_name || '', email: user.email || '' })
  }, [user])

  const mutation = useMutation({
    mutationFn: async () => api.patch('/api/auth/profile/', form),
    onSuccess: () => alert('Profile updated'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-xl mx-auto">
      <h1 className="text-2xl font-black text-white">Profile</h1>
      <Card>
        <div className="space-y-4">
          <Input label="First Name" value={form.first_name} onChange={(e) => setForm({ ...form, first_name: e.target.value })} />
          <Input label="Last Name" value={form.last_name} onChange={(e) => setForm({ ...form, last_name: e.target.value })} />
          <Input label="Email" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>Save Profile</Button>
        </div>
      </Card>
    </div>
  )
}