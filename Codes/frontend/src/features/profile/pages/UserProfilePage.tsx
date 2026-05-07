import { useQuery, useMutation } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { useAuthStore } from '../../auth/store/authStore'
import { Link } from 'react-router-dom'
import ActivityFeedPage from '../../activity/pages/ActivityFeedPage' // we can reuse part or inline stats

export default function UserProfilePage() {
  const { user } = useAuthStore()
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '' })

  const { data: profile, isLoading } = useQuery({
    queryKey: ['userInfo'],
    queryFn: async () => (await api.get('/api/auth/user/')).data,
  })

  const { data: notesCount } = useQuery({
    queryKey: ['notesCount'],
    queryFn: async () => (await api.get('/api/notes/v1/notes/statistics/')).data,
  })

  const { data: experiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get('/api/experiments/')).data,
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

  const updateMutation = useMutation({
    mutationFn: async (payload: any) => api.patch('/api/auth/profile/', payload),
    onSuccess: () => alert('Profile updated'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="animate-fade-in">
      {/* Full‑screen background hero */}
      <div className="relative mb-8">
        <div className="absolute inset-0 bg-gradient-to-r from-neon-blue/20 to-purple-500/20 rounded-3xl" />
        <div className="relative glass p-8 md:p-12 flex flex-col md:flex-row gap-8 items-center md:items-start">
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-4xl font-black text-white shadow-lg">
            {profile?.username?.[0]?.toUpperCase() || 'U'}
          </div>
          <div className="flex-1 text-center md:text-left">
            <h1 className="text-3xl font-black text-white">{profile?.username}</h1>
            <p className="text-slate-400 text-sm mt-1">{profile?.email}</p>
            <div className="flex gap-3 mt-4 justify-center md:justify-start">
              <Link to="/profile/change-password"><Button variant="secondary" size="sm">Change Password</Button></Link>
              <Button variant="ghost" size="sm" onClick={() => alert('Logout')}>Logout</Button>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-6 text-center">
            <div>
              <div className="text-2xl font-black text-white">{experiments?.length || 0}</div>
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Experiments</div>
            </div>
            <div>
              <div className="text-2xl font-black text-white">{notesCount?.total_notes || 0}</div>
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Notes</div>
            </div>
            <div>
              <div className="text-2xl font-black text-white">{profile?.is_staff ? 'Admin' : 'User'}</div>
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Role</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-2">
          <h2 className="text-sm font-black text-white uppercase tracking-wider mb-6">Edit Profile</h2>
          <div className="space-y-4">
            <Input label="First Name" value={form.first_name} onChange={e => setForm({ ...form, first_name: e.target.value })} />
            <Input label="Last Name" value={form.last_name} onChange={e => setForm({ ...form, last_name: e.target.value })} />
            <Input label="Email" type="email" value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
            <Button onClick={() => updateMutation.mutate(form)} disabled={updateMutation.isPending}>
              Save Changes
            </Button>
          </div>
        </Card>
        <Card>
          <h2 className="text-sm font-black text-white uppercase tracking-wider mb-4">Quick Links</h2>
          <div className="space-y-2 text-sm">
            <Link to="/my-jobs" className="block text-neon-blue hover:underline">My Scheduled Jobs</Link>
            <Link to="/notes" className="block text-neon-blue hover:underline">My Notes</Link>
            <Link to="/activity" className="block text-neon-blue hover:underline">Activity Feed</Link>
          </div>
        </Card>
      </div>
    </div>
  )
}