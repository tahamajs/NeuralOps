import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { MainLayout } from './components/layout/MainLayout'
import { Spinner } from './components/ui/Spinner'

// Auth
const LoginPage = lazy(() => import('./features/auth/pages/LoginPage'))
const RegisterPage = lazy(() => import('./features/auth/pages/RegisterPage'))

// Dashboard
const DashboardPage = lazy(() => import('./features/dashboard/pages/DashboardPage'))
const DashboardWidgetsConfigPage = lazy(() => import('./features/dashboard/pages/DashboardWidgetsConfigPage'))

// Experiments
const ExperimentListPage = lazy(() => import('./features/experiments/pages/ExperimentListPage'))
const ExperimentDetailPage = lazy(() => import('./features/experiments/pages/ExperimentDetailPage'))
const ExperimentComparePage = lazy(() => import('./features/experiments/pages/ExperimentComparePage'))
const ExperimentDiffPage = lazy(() => import('./features/experiments/pages/ExperimentDiffPage'))
const ExperimentMonitorPage = lazy(() => import('./features/experiments/pages/ExperimentMonitorPage'))
const ExperimentReportPage = lazy(() => import('./features/experiments/pages/ExperimentReportPage'))
const CustomMetricsPage = lazy(() => import('./features/experiments/pages/CustomMetricsPage'))
const DataLineagePage = lazy(() => import('./features/experiments/pages/DataLineagePage'))
const ModelCardPage = lazy(() => import('./features/experiments/pages/ModelCardPage'))
const RunAnalysisOverviewPage = lazy(() => import('./features/experiments/pages/RunAnalysisOverviewPage'))
const RunArtifactsPage = lazy(() => import('./features/experiments/pages/RunArtifactsPage'))
const ModelDownloadPage = lazy(() => import('./features/experiments/pages/ModelDownloadPage'))
const TrainingSnapshotsPage = lazy(() => import('./features/experiments/pages/TrainingSnapshotsPage'))
const UpdateRunConfigPage = lazy(() => import('./features/experiments/pages/UpdateRunConfigPage'))
const MarketplacePage = lazy(() => import('./features/marketplace/pages/MarketplacePage'))
const MyOrdersPage = lazy(() => import('./features/marketplace/pages/MyOrdersPage'))
const MyJobsPage = lazy(() => import('./features/marketplace/pages/MyJobsPage'))
const BillingPage = lazy(() => import('./features/marketplace/pages/BillingPage'))
const UsagePage = lazy(() => import('./features/marketplace/pages/UsagePage'))
const LeaderboardPage = lazy(() => import('./features/marketplace/pages/LeaderboardPage'))
const ReputationPage = lazy(() => import('./features/marketplace/pages/ReputationPage'))

// Launcher
const LauncherPage = lazy(() => import('./features/launcher/pages/LauncherPage'))

// Notes
const NotesListPage = lazy(() => import('./features/notes/pages/NotesListPage'))
const NoteDetailPage = lazy(() => import('./features/notes/pages/NoteDetailPage'))
const NoteCreatePage = lazy(() => import('./features/notes/pages/NoteCreatePage'))
const NoteStatisticsPage = lazy(() => import('./features/notes/pages/NoteStatisticsPage'))
const RecentNotesPage = lazy(() => import('./features/notes/pages/RecentNotesPage'))
const BulkNotesPage = lazy(() => import('./features/notes/pages/BulkNotesPage'))
const WebhookManagerPage = lazy(() => import('./features/notes/pages/WebhookManagerPage'))
const NoteVersionsPage = lazy(() => import('./features/notes/pages/NoteVersionsPage'))
const NoteVersionDetailPage = lazy(() => import('./features/notes/pages/NoteVersionDetailPage'))
const NoteRepliesPage = lazy(() => import('./features/notes/pages/NoteRepliesPage'))
const NoteMentionsPage = lazy(() => import('./features/notes/pages/NoteMentionsPage'))

// Jobs
const JobDetailPage = lazy(() => import('./features/jobs/pages/JobDetailPage'))
const JobHistoryPage = lazy(() => import('./features/jobs/pages/JobHistoryPage'))
const JobSchedulingPage = lazy(() => import('./features/jobs/pages/JobSchedulingPage'))
const ConfigHistoryPage = lazy(() => import('./features/jobs/pages/ConfigHistoryPage'))
const EnvironmentInfoPage = lazy(() => import('./features/jobs/pages/EnvironmentInfoPage'))
const MyScheduledJobsPage = lazy(() => import('./features/jobs/pages/MyScheduledJobsPage'))
const JobConfigComparePage = lazy(() => import('./features/jobs/pages/JobConfigComparePage'))

// Model Versioning
const ModelVersionsPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionsPage'))
const ModelVersionDetailPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionDetailPage'))
const ModelVersionCompareFromRunPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionCompareFromRunPage'))
const ModelLineagePage = lazy(() => import('./features/modelVersioning/pages/ModelLineagePage'))
const ModelVersionCleanupPage = lazy(() => import('./features/modelVersioning/pages/ModelVersionCleanupPage'))
const PublishModelPage = lazy(() => import('./features/modelVersioning/pages/PublishModelPage'))
const ArchiveModelPage = lazy(() => import('./features/modelVersioning/pages/ArchiveModelPage'))
const SetProductionVersionPage = lazy(() => import('./features/modelVersioning/pages/SetProductionVersionPage'))

// Drift / Fairness / Robustness
const DriftCheckPage = lazy(() => import('./features/drift/pages/DriftCheckPage'))
const FairnessPage = lazy(() => import('./features/fairness/pages/FairnessPage'))
const RobustnessPage = lazy(() => import('./features/robustness/pages/RobustnessPage'))

// Prompts
const PromptsListPage = lazy(() => import('./features/prompts/pages/PromptsListPage'))
const PromptDetailPage = lazy(() => import('./features/prompts/pages/PromptDetailPage'))
const PromptHistoryPage = lazy(() => import('./features/prompts/pages/PromptHistoryPage'))

// Cluster / Scheduler
const ClusterOverviewPage = lazy(() => import('./features/scheduler/pages/ClusterOverviewPage'))

// Search / Activity
const GlobalSearchPage = lazy(() => import('./features/search/pages/GlobalSearchPage'))
const ActivityFeedPage = lazy(() => import('./features/activity/pages/ActivityFeedPage'))

// Paper
const PaperEditorPage = lazy(() => import('./features/paper/pages/PaperEditorPage'))
const OverleafEditorPage = lazy(() => import('./features/paper/pages/OverleafEditorPage'))
const NewPaperProjectPage = lazy(() => import('./features/paper/pages/NewPaperProjectPage'))
const PaperPreviewPage = lazy(() => import('./features/paper/pages/PaperPreviewPage'))

// Analysis / Stats
const AdvancedAnalyticsPage = lazy(() => import('./features/analytics/pages/AdvancedAnalyticsPage'))
const ResearchAnalysisPage = lazy(() => import('./features/analysis/pages/ResearchAnalysisPage'))

// Templates
const TemplatesListPage = lazy(() => import('./features/templates/pages/TemplatesListPage'))
const TemplateDetailPage = lazy(() => import('./features/templates/pages/TemplateDetailPage'))

// HPO / Federated
const HPOStatusPage = lazy(() => import('./features/hpo/pages/HPOStatusPage'))
const FederatedStatusPage = lazy(() => import('./features/federated/pages/FederatedStatusPage'))

// Profile
const UserProfilePage = lazy(() => import('./features/profile/pages/UserProfilePage'))
const ChangePasswordPage = lazy(() => import('./features/profile/pages/ChangePasswordPage'))

// Admin
const AdminDashboard = lazy(() => import('./features/admin/pages/AdminDashboard'))
const AdminJobsPage = lazy(() => import('./features/admin/pages/AdminJobsPage'))
const AdminModelVersionsPage = lazy(() => import('./features/admin/pages/AdminModelVersionsPage'))
const AdminUsersPage = lazy(() => import('./features/admin/pages/AdminUsersPage'))
const StopAllJobsPage = lazy(() => import('./features/admin/pages/StopAllJobsPage'))

// System
const SystemStatusPage = lazy(() => import('./features/system/pages/SystemStatusPage'))
const HealthPage = lazy(() => import('./features/system/pages/HealthPage'))
const VersionPage = lazy(() => import('./features/system/pages/VersionPage'))
const ConfigPage = lazy(() => import('./features/config/pages/ConfigPage'))
const ModelSearchPage = lazy(() => import('./features/config/pages/ModelSearchPage'))

// Agent
const AgentChatPage = lazy(() => import('./features/agent/pages/AgentChatPage'))
const AgentSessionsListPage = lazy(() => import('./features/agent/pages/AgentSessionsListPage'))
const PaperLibraryPage = lazy(() => import('./features/agent/pages/PaperLibraryPage'))
const PromptTestingPage = lazy(() => import('./features/agent/pages/PromptTestingPage'))

// Sweeps
const SweepDashboardPage = lazy(() => import('./features/sweeps/pages/SweepDashboardPage'))
const SweepCreator = lazy(() => import('./features/sweeps/components/SweepCreator'))
// Organizations
const OrganizationListPage = lazy(() => import('../features/organizations/pages/OrganizationListPage'))
const OrganizationDetailPage = lazy(() => import('../features/organizations/pages/OrganizationDetailPage'))
const InviteAcceptPage = lazy(() => import('../features/organizations/pages/InviteAcceptPage'))

// In the router children array:
// Fallback
const NotFoundPage = lazy(() => import('./features/NotFoundPage'))

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-surface"><Spinner /></div>}><LoginPage /></Suspense>,
  },
  {
    path: '/register',
    element: <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-surface"><Spinner /></div>}><RegisterPage /></Suspense>,
  },
  {
    element: <MainLayout />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },

      // Dashboard
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/dashboard/widgets-config', element: <DashboardWidgetsConfigPage /> },
      { path: '/statistics', element: <DashboardPage /> },

      // Experiments
      { path: '/experiments', element: <ExperimentListPage /> },
      { path: '/experiments/compare', element: <ExperimentComparePage /> },
      { path: '/experiments/diff', element: <ExperimentDiffPage /> },
      { path: '/experiments/:id', element: <ExperimentDetailPage /> },
      { path: '/experiments/:id/monitor', element: <ExperimentMonitorPage /> },
      { path: '/experiments/:id/report', element: <ExperimentReportPage /> },
      { path: '/experiments/:id/custom-metrics', element: <CustomMetricsPage /> },
      { path: '/runs/:runId/snapshots', element: <TrainingSnapshotsPage /> },
      { path: '/runs/:runId/analysis', element: <RunAnalysisOverviewPage /> },
      { path: '/runs/:runId/artifacts', element: <RunArtifactsPage /> },
      { path: '/runs/:runId/model', element: <ModelDownloadPage /> },
      { path: '/runs/:runId/lineage', element: <DataLineagePage /> },
      { path: '/runs/:runId/model-card', element: <ModelCardPage /> },
      { path: '/runs/:runId/config/update', element: <UpdateRunConfigPage /> },

      // Drift / Fairness / Robustness
      { path: '/runs/:runId/drift', element: <DriftCheckPage /> },
      { path: '/runs/:runId/fairness', element: <FairnessPage /> },
      { path: '/runs/:runId/robustness', element: <RobustnessPage /> },

      // Paper
      { path: '/runs/:runId/paper', element: <PaperEditorPage /> },
      { path: '/runs/:runId/paper/preview', element: <PaperPreviewPage /> },
      { path: '/paper/editor', element: <PaperEditorPage /> },
      { path: '/paper/editor/:runId', element: <PaperEditorPage /> },
      { path: '/paper/editor/:projectId', element: <OverleafEditorPage /> },
      { path: '/paper/new', element: <NewPaperProjectPage /> },

      // Model Versioning
      { path: '/runs/:runId/model-versions', element: <ModelVersionsPage /> },
      { path: '/runs/:runId/model-versions/compare', element: <ModelVersionCompareFromRunPage /> },
      { path: '/runs/:runId/model-versions/cleanup', element: <ModelVersionCleanupPage /> },
      { path: '/runs/:runId/model-versions/set-production', element: <SetProductionVersionPage /> },
      { path: '/model-versions/:versionId', element: <ModelVersionDetailPage /> },
      { path: '/model-versions/:versionId/publish', element: <PublishModelPage /> },
      { path: '/model-versions/:versionId/archive', element: <ArchiveModelPage /> },
      { path: '/models/:versionId/lineage', element: <ModelLineagePage /> },

      // Jobs
      { path: '/jobs/:jobId', element: <JobDetailPage /> },
      { path: '/jobs/:jobId/history', element: <JobHistoryPage /> },
      { path: '/jobs/:jobId/scheduling', element: <JobSchedulingPage /> },
      { path: '/jobs/:jobId/config-history', element: <ConfigHistoryPage /> },
      { path: '/jobs/:jobId/config-compare', element: <JobConfigComparePage /> },
      { path: '/jobs/:jobId/environment', element: <EnvironmentInfoPage /> },
      { path: '/my-jobs', element: <MyScheduledJobsPage /> },

      // Launcher
      { path: '/launcher', element: <LauncherPage /> },

      // Notes
      { path: '/notes', element: <NotesListPage /> },
      { path: '/notes/new', element: <NoteCreatePage /> },
      { path: '/notes/:id', element: <NoteDetailPage /> },
      { path: '/notes/:noteId/versions', element: <NoteVersionsPage /> },
      { path: '/notes/:noteId/versions/:versionNumber', element: <NoteVersionDetailPage /> },
      { path: '/notes/:noteId/replies', element: <NoteRepliesPage /> },
      { path: '/notes/:noteId/mentions', element: <NoteMentionsPage /> },
      { path: '/notes/statistics', element: <NoteStatisticsPage /> },
      { path: '/notes/recent', element: <RecentNotesPage /> },
      { path: '/notes/bulk', element: <BulkNotesPage /> },
      { path: '/notes/webhooks', element: <WebhookManagerPage /> },

      // Prompts
      { path: '/prompts', element: <PromptsListPage /> },
      { path: '/prompts/:name', element: <PromptDetailPage /> },
      { path: '/prompts/:name/history', element: <PromptHistoryPage /> },

      // Cluster / Scheduler
      { path: '/cluster', element: <ClusterOverviewPage /> },

      // Search / Activity
      { path: '/search', element: <GlobalSearchPage /> },
      { path: '/activity', element: <ActivityFeedPage /> },

      // Analytics
      { path: '/analytics', element: <AdvancedAnalyticsPage /> },
      { path: '/analysis', element: <ResearchAnalysisPage /> },

      // Templates
      { path: '/templates', element: <TemplatesListPage /> },
      { path: '/templates/:templateId', element: <TemplateDetailPage /> },

      // HPO / Federated
      { path: '/hpo-status', element: <HPOStatusPage /> },
      { path: '/federated-status', element: <FederatedStatusPage /> },

      // Profile
      { path: '/profile', element: <UserProfilePage /> },
      { path: '/profile/change-password', element: <ChangePasswordPage /> },

      // Admin
      { path: '/admin/dashboard', element: <AdminDashboard /> },
      { path: '/admin/jobs', element: <AdminJobsPage /> },
      { path: '/admin/model-versions', element: <AdminModelVersionsPage /> },
      { path: '/admin/users', element: <AdminUsersPage /> },
      { path: '/admin/stop-all-jobs', element: <StopAllJobsPage /> },

      // System
      { path: '/system', element: <SystemStatusPage /> },
      { path: '/health', element: <HealthPage /> },
      { path: '/version', element: <VersionPage /> },
      { path: '/config', element: <ConfigPage /> },
      { path: '/models/search', element: <ModelSearchPage /> },

      // Agent
      { path: '/agent', element: <AgentChatPage /> },
      { path: '/agent/:sessionId', element: <AgentChatPage /> },
      { path: '/agent/sessions', element: <AgentSessionsListPage /> },
      { path: '/agent/papers', element: <PaperLibraryPage /> },
      { path: '/agent/prompts', element: <PromptTestingPage /> },

      // Sweeps
      { path: '/sweeps/create', element: <SweepCreator onCreated={() => {}} /> },
      { path: '/sweeps/:sweepId', element: <SweepDashboardPage /> },
      { path: '/marketplace', element: <MarketplacePage /> },
      { path: '/marketplace/orders', element: <MyOrdersPage /> },
      { path: '/marketplace/jobs', element: <MyJobsPage /> },
      { path: '/marketplace/billing', element: <BillingPage /> },
      { path: '/marketplace/usage', element: <UsagePage /> },
      { path: '/marketplace/leaderboard', element: <LeaderboardPage /> },
      { path: '/marketplace/reputation', element: <ReputationPage /> },
{ path: '/organizations', element: <OrganizationListPage /> },
{ path: '/orgs/:orgId', element: <OrganizationDetailPage /> },
{ path: '/invite/:token', element: <InviteAcceptPage /> },

      // 404
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])