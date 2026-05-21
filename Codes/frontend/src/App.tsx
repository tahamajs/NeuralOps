import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

// Import ALL pages directly (no lazy loading)
import MainLayout from './components/layout/MainLayout'
import LoginPage from './features/auth/pages/LoginPage'
import RegisterPage from './features/auth/pages/RegisterPage'
import DashboardPage from './features/dashboard/pages/DashboardPage'
import ExperimentListPage from './features/experiments/pages/ExperimentListPage'
import ExperimentDetailPage from './features/experiments/pages/ExperimentDetailPage'
import LauncherPage from './features/launcher/pages/LauncherPage'
import MarketplacePage from './features/marketplace/pages/MarketplacePage'
import MyOrdersPage from './features/marketplace/pages/MyOrdersPage'
import MyJobsPage from './features/marketplace/pages/MyJobsPage'
import JobDetailPage from './features/marketplace/pages/JobDetailPage'
import BillingPage from './features/marketplace/pages/BillingPage'
import UsagePage from './features/marketplace/pages/UsagePage'
import LeaderboardPage from './features/marketplace/pages/LeaderboardPage'
import ReputationPage from './features/marketplace/pages/ReputationPage'
import NotesListPage from './features/notes/pages/NotesListPage'
import NoteDetailPage from './features/notes/pages/NoteDetailPage'
import NoteCreatePage from './features/notes/pages/NoteCreatePage'
import PaperEditorPage from './features/paper/pages/PaperEditorPage'
import UserProfilePage from './features/profile/pages/UserProfilePage'
import SystemStatusPage from './features/system/pages/SystemStatusPage'
import NotFoundPage from './features/NotFoundPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Main layout (protected) */}
        <Route element={<MainLayout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />

          <Route path="/experiments" element={<ExperimentListPage />} />
          <Route path="/experiments/:id" element={<ExperimentDetailPage />} />

          <Route path="/launcher" element={<LauncherPage />} />

          <Route path="/marketplace" element={<MarketplacePage />} />
          <Route path="/marketplace/orders" element={<MyOrdersPage />} />
          <Route path="/marketplace/jobs" element={<MyJobsPage />} />
          <Route path="/marketplace/jobs/:jobId" element={<JobDetailPage />} />
          <Route path="/marketplace/billing" element={<BillingPage />} />
          <Route path="/marketplace/usage" element={<UsagePage />} />
          <Route path="/marketplace/leaderboard" element={<LeaderboardPage />} />
          <Route path="/marketplace/reputation" element={<ReputationPage />} />

          <Route path="/notes" element={<NotesListPage />} />
          <Route path="/notes/new" element={<NoteCreatePage />} />
          <Route path="/notes/:id" element={<NoteDetailPage />} />

          <Route path="/paper/editor" element={<PaperEditorPage />} />

          <Route path="/profile" element={<UserProfilePage />} />
          <Route path="/system" element={<SystemStatusPage />} />

          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}