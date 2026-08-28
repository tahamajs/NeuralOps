#!/bin/bash
set -e

echo "Creating placeholder feature pages..."

BASE="src/features"
mkdir -p "$BASE"

# List of all files that the router expects (relative to src/features/)
FILES=(
  # activity / search
  "activity/pages/ActivityFeedPage.tsx"
  "search/pages/GlobalSearchPage.tsx"

  # paper
  "paper/pages/PaperEditorPage.tsx"
  "paper/pages/OverleafEditorPage.tsx"
  "paper/pages/NewPaperProjectPage.tsx"
  "paper/pages/PaperPreviewPage.tsx"

  # analytics / analysis
  "analytics/pages/AdvancedAnalyticsPage.tsx"
  "analysis/pages/ResearchAnalysisPage.tsx"

  # templates
  "templates/pages/TemplatesListPage.tsx"
  "templates/pages/TemplateDetailPage.tsx"

  # scheduler / cluster
  "scheduler/pages/ClusterOverviewPage.tsx"

  # experiments
  "experiments/pages/ExperimentComparePage.tsx"
  "experiments/pages/ExperimentDiffPage.tsx"
  "experiments/pages/ExperimentMonitorPage.tsx"
  "experiments/pages/ExperimentReportPage.tsx"
  "experiments/pages/CustomMetricsPage.tsx"
  "experiments/pages/DataLineagePage.tsx"
  "experiments/pages/ModelCardPage.tsx"
  "experiments/pages/RunAnalysisOverviewPage.tsx"
  "experiments/pages/RunArtifactsPage.tsx"
  "experiments/pages/ModelDownloadPage.tsx"
  "experiments/pages/TrainingSnapshotsPage.tsx"
  "experiments/pages/UpdateRunConfigPage.tsx"

  # model versioning
  "modelVersioning/pages/ModelVersionsPage.tsx"
  "modelVersioning/pages/ModelVersionDetailPage.tsx"
  "modelVersioning/pages/ModelVersionCompareFromRunPage.tsx"
  "modelVersioning/pages/ModelLineagePage.tsx"
  "modelVersioning/pages/ModelVersionCleanupPage.tsx"
  "modelVersioning/pages/PublishModelPage.tsx"
  "modelVersioning/pages/ArchiveModelPage.tsx"
  "modelVersioning/pages/SetProductionVersionPage.tsx"

  # jobs
  "jobs/pages/JobDetailPage.tsx"
  "jobs/pages/JobHistoryPage.tsx"
  "jobs/pages/JobSchedulingPage.tsx"
  "jobs/pages/ConfigHistoryPage.tsx"
  "jobs/pages/EnvironmentInfoPage.tsx"
  "jobs/pages/MyScheduledJobsPage.tsx"
  "jobs/pages/JobConfigComparePage.tsx"

  # launcher
  "launcher/pages/LauncherPage.tsx"

  # notes
  "notes/pages/NotesListPage.tsx"
  "notes/pages/NoteDetailPage.tsx"
  "notes/pages/NoteCreatePage.tsx"
  "notes/pages/NoteStatisticsPage.tsx"
  "notes/pages/RecentNotesPage.tsx"
  "notes/pages/BulkNotesPage.tsx"
  "notes/pages/WebhookManagerPage.tsx"
  "notes/pages/NoteVersionsPage.tsx"
  "notes/pages/NoteVersionDetailPage.tsx"
  "notes/pages/NoteRepliesPage.tsx"
  "notes/pages/NoteMentionsPage.tsx"

  # dashboard
  "dashboard/pages/DashboardPage.tsx"
  "dashboard/pages/DashboardWidgetsConfigPage.tsx"

  # profile
  "profile/pages/UserProfilePage.tsx"
  "profile/pages/ChangePasswordPage.tsx"

  # hpo / federated
  "hpo/pages/HPOStatusPage.tsx"
  "federated/pages/FederatedStatusPage.tsx"

  # robustness / fairness / drift
  "robustness/pages/RobustnessPage.tsx"
  "fairness/pages/FairnessPage.tsx"
  "drift/pages/DriftCheckPage.tsx"

  # prompts
  "prompts/pages/PromptsListPage.tsx"
  "prompts/pages/PromptDetailPage.tsx"
  "prompts/pages/PromptHistoryPage.tsx"

  # system / config
  "system/pages/SystemStatusPage.tsx"
  "system/pages/HealthPage.tsx"
  "system/pages/VersionPage.tsx"
  "config/pages/ConfigPage.tsx"
  "config/pages/ModelSearchPage.tsx"

  # agent
  "agent/pages/AgentChatPage.tsx"
  "agent/pages/AgentSessionsListPage.tsx"
  "agent/pages/PaperLibraryPage.tsx"
  "agent/pages/PromptTestingPage.tsx"

  # sweeps
  "sweeps/pages/SweepDashboardPage.tsx"

  # admin
  "admin/pages/AdminDashboard.tsx"
  "admin/pages/AdminJobsPage.tsx"
  "admin/pages/AdminModelVersionsPage.tsx"
  "admin/pages/AdminUsersPage.tsx"
  "admin/pages/StopAllJobsPage.tsx"

  # 404
  "../NotFoundPage.tsx"
)

for rel in "${FILES[@]}"; do
  filepath="$BASE/$rel"
  dir=$(dirname "$filepath")
  filename=$(basename "$rel" .tsx)

  mkdir -p "$dir"

  # Only create the file if it doesn't already exist
  if [ ! -f "$filepath" ]; then
    cat > "$filepath" <<EOF
export default function ${filename}() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">${filename}</h1>
      <p className="text-slate-400 mt-2">Coming soon.</p>
    </div>
  )
}
EOF
    echo "Created $filepath"
  else
    echo "Skipped (exists) $filepath"
  fi
done

echo ""
echo "All placeholder pages are in place."
echo "You can now run: npm run dev"