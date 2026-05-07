import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { HardwareProvider } from '../../features/hardware/context/HardwareContext'
import { useAuthStore } from '../../features/auth/store/authStore'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'

const navItems = [
  { label: 'Dashboard', path: '/dashboard', icon: '◉' },
  { label: 'Experiments', path: '/experiments', icon: '⊙' },
  { label: 'Launcher', path: '/launcher', icon: '⟐' },
  { label: 'Notes', path: '/notes', icon: '▣' },
]

const secondaryItems = [
  { label: 'Cluster', path: '/cluster', icon: '⊞' },
  { label: 'Analytics', path: '/analytics', icon: '⊟' },
  { label: 'Prompts', path: '/prompts', icon: '✦' },
  { label: 'Search', path: '/search', icon: '⌕' },
]

export const MainLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  return (
    <HardwareProvider>
      <div className="min-h-screen bg-cyber-900 cyber-grid flex flex-col">
        {/* ───────────── HEADER ───────────── */}
        <header className="sticky top-0 z-50 bg-cyber-900/80 backdrop-blur-xl border-b border-white/5">
          <div className="max-w-full mx-auto px-4 md:px-6 py-3 flex items-center justify-between">
            {/* Left: Logo + Hamburger */}
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="md:hidden text-slate-400 hover:text-white p-1"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <NavLink to="/dashboard" className="flex items-center gap-2 group">
                <div className="w-9 h-9 bg-gradient-to-br from-neon-blue to-purple-500 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                  <span className="text-white font-black text-xs">ET</span>
                </div>
                <span className="text-xl font-black text-white tracking-tighter hidden sm:block">
                  NEURAL<span className="text-neon-blue">OPS</span>
                </span>
              </NavLink>
            </div>

            {/* Center: Quick Search */}
            <div className="hidden lg:block flex-1 max-w-md mx-8">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search experiments, notes, models..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-xs text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const q = (e.target as HTMLInputElement).value
                      if (q.trim()) navigate(`/search?q=${encodeURIComponent(q)}`)
                    }
                  }}
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-600 text-xs">⌕</span>
              </div>
            </div>

            {/* Right: User Menu + Status */}
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-[10px] font-bold text-green-400 uppercase tracking-widest">Online</span>
              </div>
              <div className="relative group">
                <button className="flex items-center gap-2 glass py-1.5 px-3 rounded-xl">
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                    {user?.username?.[0]?.toUpperCase() || 'U'}
                  </div>
                  <span className="text-xs text-slate-300 hidden sm:block">{user?.username || 'User'}</span>
                </button>
                {/* Dropdown */}
                <div className="absolute right-0 mt-2 w-48 glass border border-white/10 rounded-xl overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                  <NavLink to="/profile" className="block px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white">Profile</NavLink>
                  <NavLink to="/profile/change-password" className="block px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white">Change Password</NavLink>
                  <NavLink to="/my-jobs" className="block px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white">My Jobs</NavLink>
                  <hr className="border-white/5" />
                  <button onClick={logout} className="w-full text-left px-4 py-2 text-xs text-red-400 hover:bg-white/5">Logout</button>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* ───────────── BODY (Sidebar + Content) ───────────── */}
        <div className="flex-1 flex">
          {/* Sidebar */}
          <aside className={`
            fixed md:sticky top-[57px] left-0 z-40 h-[calc(100vh-57px)] w-64 glass border-r border-white/5
            transform transition-transform duration-300
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
            overflow-y-auto scrollbar-thin
          `}>
            <div className="p-4 space-y-6">
              {/* Main Nav */}
              <div>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 px-2">Main</p>
                <nav className="space-y-1">
                  {navItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={() => setSidebarOpen(false)}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                          isActive
                            ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20'
                            : 'text-slate-400 hover:text-white hover:bg-white/5'
                        }`
                      }
                    >
                      <span className="text-lg">{item.icon}</span>
                      {item.label}
                    </NavLink>
                  ))}
                </nav>
              </div>

              {/* Secondary */}
              <div>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 px-2">Tools</p>
                <nav className="space-y-1">
                  {secondaryItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={() => setSidebarOpen(false)}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                          isActive
                            ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20'
                            : 'text-slate-400 hover:text-white hover:bg-white/5'
                        }`
                      }
                    >
                      <span className="text-lg">{item.icon}</span>
                      {item.label}
                    </NavLink>
                  ))}
                </nav>
              </div>

              {/* Activity */}
              <div>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 px-2">System</p>
                <nav className="space-y-1">
                  <NavLink to="/activity" onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${isActive ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
                    <span>↯</span> Activity Feed
                  </NavLink>
                  <NavLink to="/system" onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${isActive ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
                    <span>⬡</span> System Status
                  </NavLink>
                  <NavLink to="/templates" onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${isActive ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
                    <span>⊡</span> Templates
                  </NavLink>
                </nav>
              </div>
            </div>
          </aside>

          {/* Overlay for mobile sidebar */}
          {sidebarOpen && (
            <div className="fixed inset-0 bg-black/50 z-30 md:hidden" onClick={() => setSidebarOpen(false)} />
          )}

          {/* ───────────── MAIN CONTENT ───────────── */}
          <main className="flex-1 p-4 md:p-8 animate-fade-in overflow-x-hidden">
            <Outlet />
          </main>
        </div>

        {/* ───────────── FOOTER ───────────── */}
        <footer className="bg-cyber-900/80 backdrop-blur-xl border-t border-white/5">
          <div className="max-w-full mx-auto px-6 py-4">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-4">
                <span>© 2026 NeuralOps Control Suite</span>
                <span className="hidden md:inline text-white/20">|</span>
                <span>API v1 · Django + React</span>
              </div>
              <div className="flex items-center gap-4">
                <NavLink to="/health" className="hover:text-neon-blue transition-colors">Health</NavLink>
                <NavLink to="/version" className="hover:text-neon-blue transition-colors">Version</NavLink>
                <NavLink to="/config" className="hover:text-neon-blue transition-colors">Config</NavLink>
                <a href="/admin/" target="_blank" rel="noopener noreferrer" className="hover:text-neon-blue transition-colors">Admin</a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </HardwareProvider>
  )
}