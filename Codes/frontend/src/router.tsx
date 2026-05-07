import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { MainLayout } from './components/layout/MainLayout'
import { Spinner } from './components/ui/Spinner'

const LoginPage = lazy(() => import('./features/auth/pages/LoginPage'))
const RegisterPage = lazy(() => import('./features/auth/pages/RegisterPage'))
const DashboardPage = lazy(() => import('./features/dashboard/pages/DashboardPage'))
const DashboardWidgetsConfigPage = lazy(() => import('./features/dashboard/pages/DashboardWidgetsConfigPage'))
const ExperimentListPage = lazy(() => import('./features/experiments/pages/ExperimentListPage'))
const ExperimentDetailPage = lazy(() => import('./features/experiments/pages/ExperimentDetailPage'))
const ExperimentComparePage = lazy(() => import('./features/experiments/pages/ExperimentComparePage'))
const ExperimentDiffPage = lazy(() => import('./features/experiments/pages/ExperimentDiffPage'))
const CustomMetricsPage = lazy(() => import('./features/experiments/pages/CustomMetricsPage'))
const DataLineagePage = lazy(() => import('./features/experiments/pages/DataLineagePage'))
const ModelCardPage = lazy(() => import('./features/experiments/pages/ModelCardPage'))
const RunAnalysisOverviewPage = lazy(() => import('./features/experiments/pages/RunAnalysisOverviewPage'))
const RunArtifactsPage = lazy(() => import('./features/experiments/pages/RunArtifactsPage'))
const ModelDownloadPage = lazy(() => import('./features/experiments/pages/ModelDownloadPage'))
const LauncherPage = lazy(() => import('./features/launcher/pages/LauncherPage'))
const NotesListPage = lazy(() => import('./features/notes/pages/NotesListPage'))
const NoteDetailPage = lazy(() => import('./features/notes/pages/NoteDetailPage'))
const NoteCreatePage = lazy(() => import('./features/notes/pages/NoteCreatePage'))
const NoteStatisticsPage = lazy(() => import('./features/notes/pages/NoteStatisticsPage'))
const RecentNotesPage = lazy(() => import('./features/notes/pages/RecentNotesPage'))
const BulkNotesPage = lazy(() => import('./features/notes/pages/BulkNotesPage'))
const WebhookManagerPage = lazy(() => import('./features/notes/pages/WebhookManagerPage'))
const JobDetailPage = lazy(() => import('./features/jobs/pages/JobDetailPage'))
const JobHistoryPage = lazy(() => import('./features/jobs/pages/JobHistoryPage'))
const JobSchedulingPage = lazy(() => import('./features/jobs/pages/JobSchedulingPage'))
const ConfigHistoryPage = lazy(() => import('./features/jobs/pages/ConfigHistoryPage'))
const EnvironmentInfoPage = lazy(() => import('./features/jobs/pages/EnvironmentInfoPage'))
const MyScheduledJobsPage = lazy(() => import('./features/jobs/pages/MyScheduledJobsPage'))
const ModelVersionsPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionsPage'))
const ModelVersionDetailPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionDetailPage'))
const ModelVersionCompareFromRunPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionCompareFromRunPage'))
const ModelLineagePage = lazy(() => import('./features/modelVersioning/pages/ModelLineagePage'))
const ModelVersionCleanupPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionCleanupPage'))
const PublishModelPage = lazy(() => import('./features/modelVersioning/pages/PublishModelPage'))
const ArchiveModelPage = lazy(() => import('./features/modelVersioning/pages/ArchiveModelPage'))
const SetProductionVersionPage = lazy(() => import('./features/modelVersioning/pages/SetProductionVersionPage'))
const DriftCheckPage = lazy(() => import('./features/drift/pages/DriftCheckPage'))
const FairnessPage = lazy(() => import('./features/fairness/pages/FairnessPage'))
const RobustnessPage = lazy(() => import('./features/robustness/pages/RobustnessPage'))
const PromptsListPage = lazy(() => import('./features/prompts/pages/PromptsListPage'))
const PromptDetailPage = lazy(() => import('./features/prompts/pages/PromptDetailPage'))
const PromptHistoryPage = lazy(() => import('./features/prompts/pages/PromptHistoryPage'))
const ClusterOverviewPage = lazy(() => import('./features/scheduler/pages/ClusterOverviewPage'))
const GlobalSearchPage = lazy(() => import('./features/search/pages/GlobalSearchPage'))
const ActivityFeedPage = lazy(() => import('./features/activity/pages/ActivityFeedPage'))
const PaperGeneratorPage = lazy(() => import('./features/paper/pages/PaperGeneratorPage'))
const PaperEditorPage = lazy(() => import('./features/paper/pages/PaperEditorPage'))
const PaperPreviewPage = lazy(() => import('./features/paper/pages/PaperPreviewPage'))
const AdvancedAnalyticsPage = lazy(() => import('./features/analytics/pages/AdvancedAnalyticsPage'))
const TemplatesListPage = lazy(() => import('./features/templates/pages/TemplatesListPage'))
const TemplateDetailPage = lazy(() => import('./features/templates/pages/TemplateDetailPage'))
const HPOStatusPage = lazy(() => import('./features/hpo/pages/HPOStatusPage'))
const FederatedStatusPage = lazy(() => import('./features/federated/pages/FederatedStatusPage'))
const UserProfilePage = lazy(() => import('./features/profile/pages/UserProfilePage'))
const ChangePasswordPage = lazy(() => import('./features/profile/pages/ChangePasswordPage'))
const AdminDashboard = lazy(() => import('./features/admin/pages/AdminDashboard'))
const AdminJobsPage = lazy(() => import('./features/admin/pages/AdminJobsPage'))
const AdminModelVersionsPage = lazy(() => import('./features/admin/pages/AdminModelVersionsPage'))
const AdminUsersPage = lazy(() => import('./features/admin/pages/AdminUsersPage'))
const StopAllJobsPage = lazy(() => import('./features/admin/pages/StopAllJobsPage'))
const SystemStatusPage = lazy(() => import('./features/system/pages/SystemStatusPage'))
const HealthPage = lazy(() => import('./features/system/pages/HealthPage'))
const VersionPage = lazy(() => import('./features/system/pages/VersionPage'))
const ConfigPage = lazy(() => import('./features/config/pages/ConfigPage'))
const NotFoundPage = lazy(() => import('./features/NotFoundPage'))

export const router = createBrowserRouter([
  { path: '/login', element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-cyber-900"><Spinner /></div>}><LoginPage /></Suspense> },
  { path: '/register', element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-cyber-900"><Spinner /></div>}><RegisterPage /></Suspense> },
  {
    element: <MainLayout />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/dashboard/widgets-config', element: <DashboardWidgetsConfigPage /> },
      { path: '/statistics', element: <DashboardPage /> }, // reuse dashboard stats if needed
      { path: '/experiments', element: <ExperimentListPage /> },
      { path: '/experiments/:id', element: <ExperimentDetailPage /> },
      { path: '/experiments/:id/custom-metrics', element: <CustomMetricsPage /> },
      { path: '/experiments/compare', element: <ExperimentComparePage /> },
      { path: '/experiments/diff', element: <ExperimentDiffPage /> },
      { path: '/runs/:runId/analysis', element: <RunAnalysisOverviewPage /> },
      { path: '/runs/:runId/artifacts', element: <RunArtifactsPage /> },
      { path: '/runs/:runId/model', element: <ModelDownloadPage /> },
      { path: '/runs/:runId/lineage', element: <DataLineagePage /> },
      { path: '/runs/:runId/model-card', element: <ModelCardPage /> },
      { path: '/runs/:runId/drift', element: <DriftCheckPage /> },
      { path: '/runs/:runId/fairness', element: <FairnessPage /> },
      { path: '/runs/:runId/robustness', element: <RobustnessPage /> },
      { path: '/runs/:runId/paper', element: <PaperGeneratorPage /> },
      { path: '/runs/:runId/paper/editor', element: <PaperEditorPage /> },
      { path: '/runs/:runId/paper/preview', element: <PaperPreviewPage /> },
      { path: '/runs/:runId/model-versions', element: <ModelVersionsPage /> },
      { path: '/runs/:runId/model-versions/compare', element: <ModelVersionCompareFromRunPage /> },
      { path: '/runs/:runId/model-versions/cleanup', element: <ModelVersionCleanupPage /> },
      { path: '/runs/:runId/model-versions/set-production', element: <SetProductionVersionPage /> },
      { path: '/model-versions/:versionId', element: <ModelVersionDetailPage /> },
      { path: '/model-versions/:versionId/publish', element: <PublishModelPage /> },
      { path: '/model-versions/:versionId/archive', element: <ArchiveModelPage /> },
      { path: '/models/:versionId/lineage', element: <ModelLineagePage /> },
      { path: '/jobs/:jobId', element: <JobDetailPage /> },
      { path: '/jobs/:jobId/history', element: <JobHistoryPage /> },
      { path: '/jobs/:jobId/scheduling', element: <JobSchedulingPage /> },
      { path: '/jobs/:jobId/config-history', element: <ConfigHistoryPage /> },
      { path: '/jobs/:jobId/environment', element: <EnvironmentInfoPage /> },
      { path: '/my-jobs', element: <MyScheduledJobsPage /> },
      { path: '/launcher', element: <LauncherPage /> },
      { path: '/notes', element: <NotesListPage /> },
      { path: '/notes/new', element: <NoteCreatePage /> },
      { path: '/notes/:id', element: <NoteDetailPage /> },
      { path: '/notes/statistics', element: <NoteStatisticsPage /> },
      { path: '/notes/recent', element: <RecentNotesPage /> },
      { path: '/notes/bulk', element: <BulkNotesPage /> },
      { path: '/notes/webhooks', element: <WebhookManagerPage /> },
      { path: '/prompts', element: <PromptsListPage /> },
      { path: '/prompts/:name', element: <PromptDetailPage /> },
      { path: '/prompts/:name/history', element: <PromptHistoryPage /> },
      { path: '/cluster', element: <ClusterOverviewPage /> },
      { path: '/search', element: <GlobalSearchPage /> },
      { path: '/activity', element: <ActivityFeedPage /> },
      { path: '/analytics', element: <AdvancedAnalyticsPage /> },
      { path: '/templates', element: <TemplatesListPage /> },
      { path: '/templates/:templateId', element: <TemplateDetailPage /> },
      { path: '/hpo-status', element: <HPOStatusPage /> },
      { path: '/federated-status', element: <FederatedStatusPage /> },
      { path: '/profile', element: <UserProfilePage /> },
      { path: '/profile/change-password', element: <ChangePasswordPage /> },
      { path: '/admin/dashboard', element: <AdminDashboard /> },
      { path: '/admin/jobs', element: <AdminJobsPage /> },
      { path: '/admin/model-versions', element: <AdminModelVersionsPage /> },
      { path: '/admin/users', element: <AdminUsersPage /> },
      { path: '/admin/stop-all-jobs', element: <StopAllJobsPage /> },
      { path: '/system', element: <SystemStatusPage /> },
      { path: '/health', element: <HealthPage /> },
      { path: '/version', element: <VersionPage /> },
      { path: '/config', element: <ConfigPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])