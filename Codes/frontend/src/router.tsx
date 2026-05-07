import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { MainLayout } from './components/layout/MainLayout'
import { Spinner } from './components/ui/Spinner'

const DashboardPage = lazy(() => import('./features/dashboard/pages/DashboardPage'))
const ExperimentListPage = lazy(() => import('./features/experiments/pages/ExperimentListPage'))
const ExperimentDetailPage = lazy(() => import('./features/experiments/pages/ExperimentDetailPage'))
const LauncherPage = lazy(() => import('./features/launcher/pages/LauncherPage'))
const LoginPage = lazy(() => import('./features/auth/pages/LoginPage'))

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
      {
        path: '/dashboard',
        element: (
          <Suspense fallback={<Spinner />}>
            <DashboardPage />
          </Suspense>
        ),
      },
      {
        path: '/experiments',
        element: (
          <Suspense fallback={<Spinner />}>
            <ExperimentListPage />
          </Suspense>
        ),
      },
      {
        path: '/experiments/:id',
        element: (
          <Suspense fallback={<Spinner />}>
            <ExperimentDetailPage />
          </Suspense>
        ),
      },
      {
        path: '/launcher',
        element: (
          <Suspense fallback={<Spinner />}>
            <LauncherPage />
          </Suspense>
        ),
      },
    ],
  },
])