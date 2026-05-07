import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { useAuthStore } from '../../auth/store/authStore'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { User, Mail, Calendar, Shield, Activity } from 'lucide-react'

export default function UserProfilePage() {
  const { user: authUser } = useAuthStore()
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '' })

  const { data: profile, isLoading } = useQuery({
    queryKey: ['userProfile'],
    queryFn: async () => (await api.get('/auth/user/')).data,
  })

  useEffect(() => {
    if (profile) {
      setForm({
        first_name: profile.first_name || '',
        last_name: profile.last_name || '',
        email: profile.email || '',
      })
    }
  }, [profile])

  const updateMut = useMutation({
    mutationFn: (data: any) => api.patch('/auth/profile/', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userProfile'] })
      setEditing(false)
    },
  })

  if (isLoading) return <Spinner />

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      {/* Banner */}
      <div className="glass p-8 rounded-2xl bg-gradient-to-br from-neon-blue/10 to-purple-500/10 text-center">
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center mx-auto text-3xl font-bold text-white shadow-glow-md">
          {profile?.username?.[0]?.toUpperCase() || 'U'}
        </div>
        <h1 className="text-2xl font-display font-bold text-white mt-4">{profile?.username}</h1>
        <p className="text-slate-400 text-sm">{profile?.email}</p>
        <div className="flex justify-center gap-3 mt-4">
          <Button size="sm" variant="ghost" onClick={() => setEditing(!editing)}>
            {editing ? 'Cancel' : 'Edit Profile'}
          </Button>
          <Link to="/profile/change-password">
            <Button size="sm" variant="ghost">Change Password</Button>
          </Link>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="flex items-center gap-3 p-4">
          <User size={20} className="text-neon-blue" />
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Username</p>
            <p className="text-sm text-white">{profile?.username}</p>
          </div>
        </Card>
        <Card className="flex items-center gap-3 p-4">
          <Mail size={20} className="text-neon-purple" />
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Email</p>
            <p className="text-sm text-white">{profile?.email || '—'}</p>
          </div>
        </Card>
        <Card className="flex items-center gap-3 p-4">
          <Calendar size={20} className="text-neon-green" />
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Joined</p>
            <p className="text-sm text-white">{new Date(profile?.date_joined).toLocaleDateString()}</p>
          </div>
        </Card>
        <Card className="flex items-center gap-3 p-4">
          <Shield size={20} className="text-neon-orange" />
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Role</p>
            <p className="text-sm text-white">{profile?.is_staff ? 'Admin' : 'Researcher'}</p>
          </div>
        </Card>
      </div>

      {/* Edit Form (conditional) */}
      {editing && (
        <Card className="animate-scale-in">
          <h2 className="text-sm font-bold text-white mb-4">Edit Profile</h2>
          <div className="space-y-4">
            <Input
              label="First Name"
              value={form.first_name}
              onChange={e => setForm({ ...form, first_name: e.target.value })}
            />
            <Input
              label="Last Name"
              value={form.last_name}
              onChange={e => setForm({ ...form, last_name: e.target.value })}
            />
            <Input
              label="Email"
              type="email"
              value={form.email}
              onChange={e => setForm({ ...form, email: e.target.value })}
            />
            <Button onClick={() => updateMut.mutate(form)} disabled={updateMut.isPending}>
              Save Changes
            </Button>
          </div>
        </Card>
      )}
    </div>
  )
}