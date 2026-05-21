#!/bin/bash
# create-placeholders.sh – create all missing feature pages

BASE="src/features"

declare -A PAGES=(
  # activity / search
  ["$BASE/activity/pages/ActivityFeedPage.tsx"]="export default function ActivityFeedPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Activity Feed</h1></div> }"
  ["$BASE/search/pages/GlobalSearchPage.tsx"]="export default function GlobalSearchPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Global Search</h1></div> }"

  # paper
  ["$BASE/paper/pages/PaperEditorPage.tsx"]="export default function PaperEditorPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Paper Editor</h1></div> }"
  ["$BASE/paper/pages/OverleafEditorPage.tsx"]="export default function OverleafEditorPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Overleaf Editor</h1></div> }"
  ["$BASE/paper/pages/NewPaperProjectPage.tsx"]="export default function NewPaperProjectPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>New Paper Project</h1></div> }"
  ["$BASE/paper/pages/PaperPreviewPage.tsx"]="export default function PaperPreviewPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Paper Preview</h1></div> }"

  # analytics / analysis
  ["$BASE/analytics/pages/AdvancedAnalyticsPage.tsx"]="export default function AdvancedAnalyticsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Advanced Analytics</h1></div> }"
  ["$BASE/analysis/pages/ResearchAnalysisPage.tsx"]="export default function ResearchAnalysisPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Research Analysis</h1></div> }"

  # templates
  ["$BASE/templates/pages/TemplatesListPage.tsx"]="export default function TemplatesListPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Templates</h1></div> }"
  ["$BASE/templates/pages/TemplateDetailPage.tsx"]="export default function TemplateDetailPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Template Detail</h1></div> }"

  # scheduler
  ["$BASE/scheduler/pages/ClusterOverviewPage.tsx"]="export default function ClusterOverviewPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Cluster Overview</h1></div> }"

  # experiments (all missing)
  ["$BASE/experiments/pages/ExperimentComparePage.tsx"]="export default function ExperimentComparePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Compare Experiments</h1></div> }"
  ["$BASE/experiments/pages/ExperimentDiffPage.tsx"]="export default function ExperimentDiffPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Experiment Diff</h1></div> }"
  ["$BASE/experiments/pages/CustomMetricsPage.tsx"]="export default function CustomMetricsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Custom Metrics</h1></div> }"
  ["$BASE/experiments/pages/DataLineagePage.tsx"]="export default function DataLineagePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Data Lineage</h1></div> }"
  ["$BASE/experiments/pages/ModelCardPage.tsx"]="export default function ModelCardPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Model Card</h1></div> }"
  ["$BASE/experiments/pages/RunAnalysisOverviewPage.tsx"]="export default function RunAnalysisOverviewPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Run Analysis</h1></div> }"
  ["$BASE/experiments/pages/RunArtifactsPage.tsx"]="export default function RunArtifactsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Run Artifacts</h1></div> }"
  ["$BASE/experiments/pages/ModelDownloadPage.tsx"]="export default function ModelDownloadPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Model Download</h1></div> }"
  ["$BASE/experiments/pages/TrainingSnapshotsPage.tsx"]="export default function TrainingSnapshotsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Training Snapshots</h1></div> }"
  ["$BASE/experiments/pages/UpdateRunConfigPage.tsx"]="export default function UpdateRunConfigPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Update Run Config</h1></div> }"

  # model versioning
  ["$BASE/modelVersioning/pages/ModelVersionsPage.tsx"]="export default function ModelVersionsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Model Versions</h1></div> }"
  ["$BASE/modelVersioning/pages/ModelVersionDetailPage.tsx"]="export default function ModelVersionDetailPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Version Detail</h1></div> }"
  ["$BASE/modelVersioning/pages/ModelVersionCompareFromRunPage.tsx"]="export default function ModelVersionCompareFromRunPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Compare Versions</h1></div> }"
  ["$BASE/modelVersioning/pages/ModelLineagePage.tsx"]="export default function ModelLineagePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Model Lineage</h1></div> }"
  ["$BASE/modelVersioning/pages/ModelVersionCleanupPage.tsx"]="export default function ModelVersionCleanupPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Version Cleanup</h1></div> }"
  ["$BASE/modelVersioning/pages/PublishModelPage.tsx"]="export default function PublishModelPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Publish Model</h1></div> }"
  ["$BASE/modelVersioning/pages/ArchiveModelPage.tsx"]="export default function ArchiveModelPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Archive Model</h1></div> }"
  ["$BASE/modelVersioning/pages/SetProductionVersionPage.tsx"]="export default function SetProductionVersionPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Set Production</h1></div> }"

  # jobs
  ["$BASE/jobs/pages/JobDetailPage.tsx"]="export default function JobDetailPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Job Detail</h1></div> }"
  ["$BASE/jobs/pages/JobHistoryPage.tsx"]="export default function JobHistoryPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Job History</h1></div> }"
  ["$BASE/jobs/pages/JobSchedulingPage.tsx"]="export default function JobSchedulingPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Job Scheduling</h1></div> }"
  ["$BASE/jobs/pages/ConfigHistoryPage.tsx"]="export default function ConfigHistoryPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Config History</h1></div> }"
  ["$BASE/jobs/pages/EnvironmentInfoPage.tsx"]="export default function EnvironmentInfoPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Environment Info</h1></div> }"
  ["$BASE/jobs/pages/MyScheduledJobsPage.tsx"]="export default function MyScheduledJobsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>My Scheduled Jobs</h1></div> }"
  ["$BASE/jobs/pages/JobConfigComparePage.tsx"]="export default function JobConfigComparePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Config Compare</h1></div> }"

  # launcher
  ["$BASE/launcher/pages/LauncherPage.tsx"]="export default function LauncherPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Experiment Launcher</h1></div> }"

  # notes (most already exist? but ensure)
  ["$BASE/notes/pages/NotesListPage.tsx"]="export default function NotesListPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Notes</h1></div> }"
  ["$BASE/notes/pages/NoteDetailPage.tsx"]="export default function NoteDetailPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Note Detail</h1></div> }"
  ["$BASE/notes/pages/NoteCreatePage.tsx"]="export default function NoteCreatePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Create Note</h1></div> }"
  ["$BASE/notes/pages/NoteStatisticsPage.tsx"]="export default function NoteStatisticsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Note Statistics</h1></div> }"
  ["$BASE/notes/pages/RecentNotesPage.tsx"]="export default function RecentNotesPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Recent Notes</h1></div> }"
  ["$BASE/notes/pages/BulkNotesPage.tsx"]="export default function BulkNotesPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Bulk Notes</h1></div> }"
  ["$BASE/notes/pages/WebhookManagerPage.tsx"]="export default function WebhookManagerPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Webhooks</h1></div> }"
  ["$BASE/notes/pages/NoteVersionsPage.tsx"]="export default function NoteVersionsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Note Versions</h1></div> }"
  ["$BASE/notes/pages/NoteVersionDetailPage.tsx"]="export default function NoteVersionDetailPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Version Detail</h1></div> }"
  ["$BASE/notes/pages/NoteRepliesPage.tsx"]="export default function NoteRepliesPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Replies</h1></div> }"
  ["$BASE/notes/pages/NoteMentionsPage.tsx"]="export default function NoteMentionsPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Mentions</h1></div> }"

  # dashboard
  ["$BASE/dashboard/pages/DashboardPage.tsx"]="export default function DashboardPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Dashboard</h1></div> }"
  ["$BASE/dashboard/pages/DashboardWidgetsConfigPage.tsx"]="export default function DashboardWidgetsConfigPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Widget Config</h1></div> }"

  # profile
  ["$BASE/profile/pages/UserProfilePage.tsx"]="export default function UserProfilePage() { return <div className='p-6'><h1 className='text-2xl font-bold'>User Profile</h1></div> }"
  ["$BASE/profile/pages/ChangePasswordPage.tsx"]="export default function ChangePasswordPage() { return <div className='p-6'><h1 className='text-2xl font-bold'>Change Password</h1></div> }"

  # 404
  ["$BASE/NotFoundPage.tsx"]="export default function NotFoundPage() { return <div className='min-h-screen flex items-center justify-center'><h1 className='text-4xl'>404</h1></div> }"
)

for file in "${!PAGES[@]}"; do
  mkdir -p "$(dirname "$file")"
  echo "${PAGES[$file]}" > "$file"
done

echo "Placeholder pages created."