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
    <div className="min-h-screen bg-slate-50">
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md shadow-sm border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link to="/dashboard" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-black text-xs">ET</span>
              </div>
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                Toolkit
              </span>
            </Link>
            
            <div className="hidden md:flex items-center gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    location.pathname.startsWith(item.path)
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
             <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
             <span className="text-xs font-mono text-slate-500 uppercase tracking-widest">System Online</span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-4 md:p-8 animate-fade-in">
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