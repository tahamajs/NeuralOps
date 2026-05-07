import { NavLink, Outlet } from 'react-router-dom'
import { HardwareProvider } from '../../features/hardware/context/HardwareContext'

const navItems = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Experiments', path: '/experiments' },
  { label: 'Launcher', path: '/launcher' },
  { label: 'Notes', path: '/notes' },
]]

export const MainLayout = () => (
  <HardwareProvider>
    <div className="min-h-screen bg-cyber-900 cyber-grid">
      <nav className="sticky top-0 z-50 bg-cyber-900/80 backdrop-blur-xl border-b border-white/5 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-8">
            <NavLink to="/dashboard" className="flex items-center gap-2 group">
              <div className="w-9 h-9 bg-gradient-to-br from-neon-blue to-purple-500 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                <span className="text-white font-black text-xs">ET</span>
              </div>
              <span className="text-xl font-black text-white tracking-tighter">
                NEURAL<span className="text-neon-blue">OPS</span>
              </span>
            </NavLink>
            <div className="hidden md:flex gap-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${
                      isActive
                        ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/20'
                        : 'text-slate-500 hover:text-white hover:bg-white/5'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:block text-xs text-slate-400 font-mono">Cluster: Active</div>
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto p-4 md:p-8 animate-fade-in">
        <Outlet />
      </main>
    </div>
  </HardwareProvider>
)