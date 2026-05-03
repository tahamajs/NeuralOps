import { Routes, Route, Navigate } from 'react-router-dom'
import DashboardPage from './pages/DashboardPage'
import ExperimentListPage from './pages/ExperimentListPage'
import ExperimentDetailPage from './pages/ExperimentDetailPage'
import LauncherPage from './pages/LauncherPage'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center gap-6">
          <a href="/dashboard" className="text-xl font-bold text-gray-800">ET Toolkit</a>
          <a href="/dashboard" className="text-sm text-gray-600 hover:text-gray-900">Dashboard</a>
          <a href="/experiments" className="text-sm text-gray-600 hover:text-gray-900">Experiments</a>
          <a href="/launcher" className="text-sm text-gray-600 hover:text-gray-900">Launcher</a>
        </div>
      </nav>
      <div className="max-w-7xl mx-auto p-6">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/experiments" element={<ExperimentListPage />} />
          <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
          <Route path="/launcher" element={<LauncherPage />} />
        </Routes>
      </div>
    </div>
  )
}