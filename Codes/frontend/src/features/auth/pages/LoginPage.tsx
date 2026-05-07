import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../../../api/client'
import { useAuthStore } from '../store/authStore'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Card } from '../../../components/ui/Card'
import { Eye, EyeOff, LogIn } from 'lucide-react'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const setAuth = useAuthStore((s) => s.setAuth)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.post('/auth/login/', { username, password })
      const { token, ...user } = res.data
      localStorage.setItem('token', token)
      document.cookie = `authToken=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`
      setAuth(user, token)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err.response?.data?.error || 'Invalid credentials')
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
          <h1 className="text-2xl font-display font-bold text-white">Welcome back</h1>
          <p className="text-slate-400 text-sm mt-1">Sign in to your account</p>
        </div>

        {error && (
          <div className="mb-6 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <Input
            label="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your username"
            required
          />
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Password</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <Button type="submit" className="w-full" size="lg" loading={loading}>
            <LogIn size={18} className="mr-2" />
            {loading ? 'Signing in...' : 'Sign in'}
          </Button>
        </form>

        <div className="mt-6 text-center text-xs text-slate-500">
          Don't have an account?{' '}
          <Link to="/register" className="text-neon-blue hover:text-neon-blue/80 font-semibold transition-colors">
            Create one
          </Link>
        </div>
      </Card>
    </div>
  )
}