import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { MainLayout } from './components/layout/MainLayout'
import { Spinner } from './components/ui/Spinner'

const DashboardPage = lazy(() => import('./features/dashboard/pages/DashboardPage'))
const ExperimentListPage = lazy(() => import('./features/experiments/pages/ExperimentListPage'))
const ExperimentDetailPage = lazy(() => import('./features/experiments/pages/ExperimentDetailPage'))
const LauncherPage = lazy(() => import('./features/launcher/pages/LauncherPage'))
const LoginPage = lazy(() => import('./features/auth/pages/LoginPage'))
const RegisterPage = lazy(() => import('./features/auth/pages/RegisterPage'))
const NotesListPage = lazy(() => import('./features/notes/pages/NotesListPage'))
const NoteDetailPage = lazy(() => import('./features/notes/pages/NoteDetailPage'))
const NoteCreatePage = lazy(() => import('./features/notes/pages/NoteCreatePage'))
const NotFoundPage = lazy(() => import('./features/NotFoundPage'))

const ModelVersionsPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionsPage'))
const ModelVersionComparePage = lazy(() => import('./features/modelVersioning/pages/ModelVersionComparePage'))
const ModelLineagePage = lazy(() => import('./features/modelVersioning/pages/ModelLineagePage'))
const DriftCheckPage = lazy(() => import('./features/drift/pages/DriftCheckPage'))
const FairnessPage = lazy(() => import('./features/fairness/pages/FairnessPage'))
const RobustnessPage = lazy(() => import('./features/robustness/pages/RobustnessPage'))
const PromptsListPage = lazy(() => import('./features/prompts/pages/PromptsListPage'))
const PromptDetailPage = lazy(() => import('./features/prompts/pages/PromptDetailPage'))
const ClusterOverviewPage = lazy(() => import('./features/scheduler/pages/ClusterOverviewPage'))
const GlobalSearchPage = lazy(() => import('./features/search/pages/GlobalSearchPage'))
const ActivityFeedPage = lazy(() => import('./features/activity/pages/ActivityFeedPage'))
const PaperGeneratorPage = lazy(() => import('./features/paper/pages/PaperGeneratorPage'))
const PaperEditorPage = lazy(() => import('./features/paper/pages/PaperEditorPage'))
const AdvancedAnalyticsPage = lazy(() => import('./features/analytics/pages/AdvancedAnalyticsPage'))
const TemplatesListPage = lazy(() => import('./features/templates/pages/TemplatesListPage'))
const TemplateDetailPage = lazy(() => import('./features/templates/pages/TemplateDetailPage'))
const HPOStatusPage = lazy(() => import('./features/hpo/pages/HPOStatusPage'))
const FederatedStatusPage = lazy(() => import('./features/federated/pages/FederatedStatusPage'))
const UserProfilePage = lazy(() => import('./features/profile/pages/UserProfilePage'))
const ChangePasswordPage = lazy(() => import('./features/profile/pages/ChangePasswordPage'))
// new imports
const MyScheduledJobsPage = lazy(() => import('./features/jobs/pages/MyScheduledJobsPage'))
const ConfigHistoryPage = lazy(() => import('./features/jobs/pages/ConfigHistoryPage'))
const CustomMetricsPage = lazy(() => import('./features/jobs/pages/CustomMetricsPage')) // or experiments, but use same
const EnvironmentInfoPage = lazy(() => import('./features/jobs/pages/EnvironmentInfoPage'))
const DashboardWidgetsConfigPage = lazy(() => import('./features/dashboard/pages/DashboardWidgetsConfigPage'))
const StopAllJobsPage = lazy(() => import('./features/admin/pages/StopAllJobsPage'))

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-cyber-900"><Spinner /></div>}><LoginPage /></Suspense>,
  },
  {
    path: '/register',
    element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-cyber-900"><Spinner /></div>}><RegisterPage /></Suspense>,
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

      // Model Versioning
      { path: '/runs/:runId/model-versions', element: <ModelVersionsPage /> },
      { path: '/runs/:runId/model-versions/compare', element: <ModelVersionComparePage /> },
      { path: '/models/:versionId/lineage', element: <ModelLineagePage /> },

      // Drift / Fairness / Robustness
      { path: '/runs/:runId/drift', element: <DriftCheckPage /> },
      { path: '/runs/:runId/fairness', element: <FairnessPage /> },
      { path: '/runs/:runId/robustness', element: <RobustnessPage /> },

      // Prompts
      { path: '/prompts', element: <PromptsListPage /> },
      { path: '/prompts/:name', element: <PromptDetailPage /> },

      // Cluster & Scheduler
      { path: '/cluster', element: <ClusterOverviewPage /> },

      // Search
      { path: '/search', element: <GlobalSearchPage /> },

      // Activity
      { path: '/activity', element: <ActivityFeedPage /> },

      // Paper & Reports
      { path: '/runs/:runId/paper', element: <PaperGeneratorPage /> },
      { path: '/runs/:runId/paper/editor', element: <PaperEditorPage /> },

      // Analytics
      { path: '/analytics', element: <AdvancedAnalyticsPage /> },

      // Templates
      { path: '/templates', element: <TemplatesListPage /> },
      { path: '/templates/:templateId', element: <TemplateDetailPage /> },

      // HPO / Federated
      { path: '/hpo-status', element: <HPOStatusPage /> },
      { path: '/federated-status', element: <FederatedStatusPage /> },

      // Profile
      { path: '/profile', element: <UserProfilePage /> },
      { path: '/profile/change-password', element: <ChangePasswordPage /> },

      // Health / Version (could be a simple page)
      { path: '/health', element: <div className="text-white">Health OK</div> },
      { path: '/version', element: <div className="text-white">API v1</div> },

      // Fallback
      { path: '*', element: <NotFoundPage /> },
      { path: '/my-jobs', element: <MyScheduledJobsPage /> },
{ path: '/profile/change-password', element: <ChangePasswordPage /> },
{ path: '/jobs/:jobId/environment', element: <EnvironmentInfoPage /> },
{ path: '/runs/:runId/custom-metrics', element: <CustomMetricsPage /> },
{ path: '/my-jobs', element: <MyScheduledJobsPage /> },
{ path: '/jobs/:jobId/config-history', element: <ConfigHistoryPage /> },
{ path: '/dashboard/widgets-config', element: <DashboardWidgetsConfigPage /> },
{ path: '/admin/stop-all-jobs', element: <StopAllJobsPage /> },
    ],
  },
])