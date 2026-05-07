import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { MainLayout } from './components/layout/MainLayout'
import { Spinner } from './components/ui/Spinner'

const DashboardPage = lazy(() => import('./features/dashboard/pages/DashboardPage'))
const ExperimentListPage = lazy(() => import('./features/experiments/pages/ExperimentListPage'))
const ExperimentDetailPage = lazy(() => import('./features/experiments/pages/ExperimentDetailPage'))
const LauncherPage = lazy(() => import('./features/launcher/pages/LauncherPage'))
const LoginPage = lazy(() => import('./features/auth/pages/LoginPage'))
const NotesListPage = lazy(() => import('./features/notes/pages/NotesListPage'))
const NoteDetailPage = lazy(() => import('./features/notes/pages/NoteDetailPage'))
const NoteCreatePage = lazy(() => import('./features/notes/pages/NoteCreatePage'))

export const router = createBrowserRouter([
  {
    path: '/login',
    element: (
      <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-cyber-900"><Spinner /></div>}>
        <LoginPage />
      </Suspense>
    ),
  },
  {
    element: <MainLayout />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/experiments', element: <ExperimentListPage /> },
      { path: '/experiments/:id', element: <ExperimentDetailPage /> },
      { path: '/launcher', element: <LauncherPage /> },
      { path: '/notes', element: <NotesListPage /> },
      { path: '/notes/new', element: <NoteCreatePage /> },
      { path: '/notes/:id', element: <NoteDetailPage /> },
    ],
  },
])