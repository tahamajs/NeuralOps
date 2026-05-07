import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../../../api/client'
import { useAuthStore } from '../store/authStore'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Card } from '../../../components/ui/Card'
import { Eye, EyeOff, UserPlus } from 'lucide-react'

export default function RegisterPage() {
  const [form, setForm] = useState({
    username: '', email: '', password: '', confirmPassword: '',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const setAuth = useAuthStore((s) => s.setAuth)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await api.post('/auth/register/', {
        username: form.username,
        email: form.email,
        password: form.password,
      })
      const { token, ...user } = res.data
      localStorage.setItem('token', token)
      document.cookie = `authToken=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`
      setAuth(user, token)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err.response?.data?.error || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface cyber-grid">
      <Card className="w-full max-w-md p-8 animate-scale-in border border-white/10">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-neon-blue to-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-glow-md">
            <span className="text-white font-black text-xl">NO</span>
          </div>
          <h1 className="text-2xl font-display font-bold text-white">Create account</h1>
          <p className="text-slate-400 text-sm mt-1">Join the NeuralOps platform</p>
        </div>
        {error && (
          <div className="mb-6 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit} className="space-y-5">
          <Input label="Username" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} placeholder="Choose a username" required />
          <Input label="Email (optional)" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="you@example.com" />
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Password</label>
            <div className="relative">
              <input type={showPassword ? 'text' : 'password'} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} placeholder="At least 6 characters" required className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors pr-10" />
              <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white">
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <Input label="Confirm Password" type="password" value={form.confirmPassword} onChange={(e) => setForm({ ...form, confirmPassword: e.target.value })} placeholder="Repeat your password" required />
          <Button type="submit" className="w-full" size="lg" loading={loading}>
            <UserPlus size={18} className="mr-2" />
            {loading ? 'Creating account...' : 'Create account'}
          </Button>
        </form>
        <div className="mt-6 text-center text-xs text-slate-500">
          Already have an account?{' '}
          <Link to="/login" className="text-neon-blue hover:text-neon-blue/80 font-semibold transition-colors">
            Sign in
          </Link>
        </div>
      </Card>
    </div>
  )
}