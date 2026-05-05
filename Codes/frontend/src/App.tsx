import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom'
import DashboardPage from './pages/DashboardPage'
import ExperimentListPage from './pages/ExperimentListPage'
import ExperimentDetailPage from './pages/ExperimentDetailPage'
import LauncherPage from './pages/LauncherPage'

export default function App() {
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Experiments', path: '/experiments' },
    { label: 'Launcher', path: '/launcher' },
  ];

  return (
    <div className="min-h-screen bg-[#0a0c10] cyber-grid">
      <nav className="sticky top-0 z-50 bg-[#0a0c10]/70 backdrop-blur-xl border-b border-white/5 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-12">
            <Link to="/dashboard" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-110 transition-transform">
                <span className="text-white font-black text-sm tracking-tighter">ET</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xl font-black text-white leading-none tracking-tighter">
                  NEURAL<span className="text-blue-500">OPS</span>
                </span>
                <span className="text-[10px] font-bold text-slate-500 tracking-[0.3em] uppercase">Control Suite</span>
              </div>
            </Link>
            
            <div className="hidden md:flex items-center gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${
                    location.pathname.startsWith(item.path)
                      ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20'
                      : 'text-slate-500 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-6">
             <div className="hidden sm:flex flex-col items-end">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Global Status</span>
                <span className="text-xs font-black text-green-400 uppercase tracking-tighter">Operational</span>
             </div>
             <div className="h-10 w-[1px] bg-white/5 mx-2" />
             <div className="w-8 h-8 rounded-full bg-white/10 border border-white/10 flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
             </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-4 md:p-10">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/experiments" element={<ExperimentListPage />} />
          <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
          <Route path="/launcher" element={<LauncherPage />} />
        </Routes>
      </main>
    </div>
  )
}