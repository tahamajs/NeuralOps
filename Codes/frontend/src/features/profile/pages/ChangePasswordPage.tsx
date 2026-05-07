import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Card } from '../../../components/ui/Card'
import { Lock } from 'lucide-react'

export default function ChangePasswordPage() {
  const [form, setForm] = useState({ old_password: '', new_password: '' })
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setSuccess('')
    try {
      const res = await api.post('/auth/change-password/', form)
      setSuccess('Password changed successfully.')
      setForm({ old_password: '', new_password: '' })
      // Optionally update token
      if (res.data.token) {
        localStorage.setItem('token', res.data.token)
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to change password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="w-full max-w-md p-8">
        <div className="text-center mb-6">
          <Lock size={32} className="mx-auto text-neon-blue mb-2" />
          <h1 className="text-xl font-display font-bold text-white">Change Password</h1>
        </div>
        {error && <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
        {success && <div className="mb-4 p-3 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 text-xs">{success}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input label="Current Password" type="password" value={form.old_password} onChange={e => setForm({ ...form, old_password: e.target.value })} required />
          <Input label="New Password" type="password" value={form.new_password} onChange={e => setForm({ ...form, new_password: e.target.value })} required />
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Updating...' : 'Update Password'}
          </Button>
        </form>
      </Card>
    </div>
  )
}