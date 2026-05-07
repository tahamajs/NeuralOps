import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { HardwareProvider } from '../../features/hardware/context/HardwareContext'
import { useAuthStore } from '../../features/auth/store/authStore'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Badge } from '../ui/Badge'

// ... (keep existing imports, add these new ones)
import { Bell, Search, Sun, Moon, ChevronLeft, ChevronRight, User, Settings, LogOut } from 'lucide-react'

export const MainLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [darkMode, setDarkMode] = useState(true)
  const [notifications, setNotifications] = useState([
    { id: 1, text: 'Experiment "bert-mrpc" completed', time: '5 min ago', read: false },
    { id: 2, text: 'New note shared with you', time: '1 hour ago', read: true },
  ])
  const [showNotifications, setShowNotifications] = useState(false)
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const navItems = [
    { label: 'Dashboard', path: '/dashboard', icon: 'LayoutDashboard' },
    { label: 'Experiments', path: '/experiments', icon: 'FlaskConical' },
    { label: 'Launcher', path: '/launcher', icon: 'Rocket' },
    { label: 'Notes', path: '/notes/notebook', icon: 'BookOpen' },
    { label: 'Analysis', path: '/analysis', icon: 'BarChart3' },
    { label: 'Papers', path: '/paper/new', icon: 'FileText' },
  ]

  // Icons mapping
  const iconComponents: Record<string, React.ReactNode> = {
    LayoutDashboard: <LayoutDashboard size={18} />,
    FlaskConical: <FlaskConical size={18} />,
    Rocket: <Rocket size={18} />,
    BookOpen: <BookOpen size={18} />,
    BarChart3: <BarChart3 size={18} />,
    FileText: <FileText size={18} />,
  }

  return (
    <HardwareProvider>
      <div className={`min-h-screen flex ${darkMode ? 'bg-surface' : 'bg-gray-50'}`}>
        {/* Sidebar */}
        <aside className={`${sidebarCollapsed ? 'w-20' : 'w-64'} glass border-r border-white/5 flex flex-col transition-all duration-300 sticky top-0 h-screen z-40`}>
          <div className="p-4 flex items-center justify-between">
            {!sidebarCollapsed && (
              <NavLink to="/dashboard" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-neon-blue to-purple-500 rounded-lg flex items-center justify-center shadow-glow-sm">
                  <span className="text-white font-black text-xs">NO</span>
                </div>
                <span className="font-display font-bold text-white tracking-tight">NeuralOps</span>
              </NavLink>
            )}
            <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} className="text-slate-400 hover:text-white p-1 rounded-lg">
              {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </button>
          </div>
          
          <nav className="flex-1 px-3 space-y-1 overflow-y-auto">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-neon-blue/10 text-neon-blue shadow-glow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  } ${sidebarCollapsed ? 'justify-center' : ''}`
                }
                title={sidebarCollapsed ? item.label : undefined}
              >
                {iconComponents[item.icon]}
                {!sidebarCollapsed && item.label}
              </NavLink>
            ))}
          </nav>

          <div className="p-4 border-t border-white/5">
            <div className={`flex items-center gap-3 ${sidebarCollapsed ? 'justify-center' : ''}`}>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </div>
              {!sidebarCollapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">{user?.username || 'User'}</p>
                  <p className="text-[10px] text-slate-500">Researcher</p>
                </div>
              )}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Header */}
          <header className={`sticky top-0 z-30 ${darkMode ? 'bg-surface/80' : 'bg-white/80'} backdrop-blur-xl border-b border-white/5`}>
            <div className="px-6 py-3 flex items-center justify-between gap-4">
              {/* Global Search */}
              <div className="flex-1 max-w-xl relative">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search experiments, notes, papers..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-sm text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') navigate(`/search?q=${encodeURIComponent((e.target as HTMLInputElement).value)}`)
                  }}
                />
              </div>

              <div className="flex items-center gap-3">
                <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
                  {darkMode ? <Sun size={18} /> : <Moon size={18} />}
                </button>
                
                {/* Notifications */}
                <div className="relative">
                  <button onClick={() => setShowNotifications(!showNotifications)} className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors relative">
                    <Bell size={18} />
                    {notifications.filter(n => !n.read).length > 0 && (
                      <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                    )}
                  </button>
                  {showNotifications && (
                    <div className="absolute right-0 top-12 w-80 glass border border-white/10 rounded-2xl shadow-2xl p-4 z-50 animate-scale-in">
                      <h3 className="text-sm font-bold text-white mb-3">Notifications</h3>
                      {notifications.map(n => (
                        <div key={n.id} className={`py-2 border-b border-white/5 last:border-0 ${n.read ? 'opacity-60' : ''}`}>
                          <p className="text-xs text-white">{n.text}</p>
                          <p className="text-[10px] text-slate-500">{n.time}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="relative group">
                  <button className="flex items-center gap-2 glass py-1.5 px-3 rounded-xl">
                    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                      {user?.username?.[0]?.toUpperCase() || 'U'}
                    </div>
                    <ChevronDown size={14} className="text-slate-400" />
                  </button>
                  <div className="absolute right-0 mt-2 w-48 glass border border-white/10 rounded-xl overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                    <NavLink to="/profile" className="flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white"><User size={14} /> Profile</NavLink>
                    <NavLink to="/profile/change-password" className="flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white"><Key size={14} /> Password</NavLink>
                    <hr className="border-white/5" />
                    <button onClick={logout} className="w-full flex items-center gap-2 px-4 py-2 text-xs text-red-400 hover:bg-white/5"><LogOut size={14} /> Logout</button>
                  </div>
                </div>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6 md:p-8 animate-fade-in">
            <Outlet />
          </main>
        </div>
      </div>
    </HardwareProvider>
  )
}