// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HardwareProvider } from './features/hardware/context/HardwareContext'
import { MainLayout } from './components/layout/MainLayout'
import FloatingWindowRenderer from './components/floating/FloatingWindow'
import FloatingWindowBar from './components/floating/FloatingWindowBar'
import { useWindowStore } from './store/windowStore'
import { DockableWindow } from './components/DockableWindow'
import { WindowLauncher } from './components/WindowLauncher'
import { WorkspaceManager } from './components/WorkspaceManager'
import { Toaster } from './components/ui/Toast'
import { ChatPanel } from './features/chat/components/ChatPanel'


// ========================
// 1. Core / Layout Pages (used in Router)
// ========================
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

// ========================
// 2. Floating Windows (used in DockableWindow componentMap)
// ========================
// Terminal & Chat
import { TerminalPanel } from './features/terminal/components/TerminalPanel'
import { ChatPanel } from './features/chat/components/ChatPanel'
import { FloatingNotesEditor } from './features/notes/components/FloatingNotesEditor'
import { PaperEditorPanel } from './features/paper/components/PaperEditorPanel'

// Marketplace & GPU
import { GPUMarketplacePage } from './features/marketplace/pages/GPUMarketplacePage'
import GPUTradingTerminal from './features/marketplace/pages/GPUTradingTerminal'
import AgenticMarketplacePage from './features/marketplaceAI/pages/AgenticMarketplacePage'
import ComparisonBuilder from './features/marketplaceAI/pages/ComparisonBuilder'

// Experiments & Analysis
import { ExperimentListPage as ExperimentListWindow } from './features/experiments/pages/ExperimentListPage'
import { ResearchAnalysisPage } from './features/analysis/pages/ResearchAnalysisPage'
import { AdvancedAnalyticsPage } from './features/analytics/pages/AdvancedAnalyticsPage'
import AcceleratorDashboard from './features/accelerator/pages/AcceleratorDashboard'
import HPODashboard from './features/hpo/pages/HPODashboard'
import SweepDetailPage from './features/hpo/pages/SweepDetailPage'
import RecommendationListPage from './features/recommender/pages/RecommendationListPage'
import FailureAnalysisPage from './features/forensics/pages/FailureAnalysisPage'

// AI & Research Co‑pilot
import IdeaTournamentPage from './features/coscientist/pages/IdeaTournamentPage'
import HypothesisBrowser from './features/coscientist/pages/HypothesisBrowser'
import DeepResearchPage from './features/research/pages/DeepResearchPage'
import AutonomousPipelineMonitor from './features/autonomous/pages/AutonomousPipelineMonitor'
import StepEditor from './features/autonomous/pages/StepEditor'
import CodeAssistantPage from './features/codingAssistant/pages/CodeAssistantPage'

// Literature & Discovery
import LiteratureMapPage from './features/literature/pages/LiteratureMapPage'
import CitationGraphViewer from './features/literature/pages/CitationGraphViewer'

// Impact Predictor
import ImpactScorePage from './features/impactPredictor/pages/ImpactScorePage'
import ProjectLeaderboard from './features/impactPredictor/pages/ProjectLeaderboard'

// Collaborator Matching & Expert Wall
import CollaboratorSuggestionPage from './features/collaborator/pages/CollaboratorSuggestionPage'
import MatchHistoryList from './features/collaborator/pages/MatchHistoryList'
import ExpertDirectoryPage from './features/expertWall/pages/ExpertDirectoryPage'
import ProjectApplicationPanel from './features/expertWall/pages/ProjectApplicationPanel'

// Gamification & Streaks
import StreakHeatmapPage from './features/streaks/pages/StreakHeatmapPage'
import WeeklySummaryViewer from './features/streaks/pages/WeeklySummaryViewer'
import BadgeWallPage from './features/gamification/pages/BadgeWallPage'
import ContributionLeaderboard from './features/gamification/pages/ContributionLeaderboard'

// Admin, Organizations, Community (Floating versions – reuse pages)
import { AdminDashboard } from './features/admin/pages/AdminDashboard'
import { ClusterOverviewPage } from './features/cluster/pages/ClusterOverviewPage'
import { OrganizationListPage } from './features/organizations/pages/OrganizationListPage'
import { QuestionListPage } from './features/community/pages/QuestionListPage'
import { UserProfilePage as UserProfileWindow } from './features/profile/pages/UserProfilePage'

// 3D Model Viewer & Federated Learning
import ModelViewerPage from './features/modelViewer/pages/ModelViewerPage'
import FLRunPage from './features/federated/pages/FLRunPage'

// Anomaly & HPO (floating versions)
import AnomalyListPage from './features/anomaly/pages/AnomalyListPage'
import AnomalyDetailPage from './features/anomaly/pages/AnomalyDetailPage'
import HPORunsPage from './features/hpo/pages/HPORunsPage'

// ========================
// 3. Component Map for DockableWindow
// ========================
const componentMap: Record<string, React.ComponentType<any>> = {
  // Core
  terminal: TerminalPanel,
  chat: ChatPanel,

  notes: FloatingNotesEditor,
  'paper-editor': PaperEditorPanel,
  marketplace: GPUMarketplacePage,
  gpumarket: GPUTradingTerminal,
  'marketplace-ai': AgenticMarketplacePage,
  'comparison-builder': ComparisonBuilder,

  // Experiments & Analysis
  experiments: ExperimentListWindow,
  analysis: ResearchAnalysisPage,
  analytics: AdvancedAnalyticsPage,
  accelerator: AcceleratorDashboard,
  hpo: HPODashboard,
  sweep_detail: SweepDetailPage,
  recommendations: RecommendationListPage,
  forensics: FailureAnalysisPage,

  // AI & Research
  tournament: IdeaTournamentPage,
  hypotheses: HypothesisBrowser,
  research: DeepResearchPage,
  'pipeline-monitor': AutonomousPipelineMonitor,
  'step-editor': StepEditor,
  'coding-assistant': CodeAssistantPage,

  // Literature
  'literature-map': LiteratureMapPage,
  'citation-graph': CitationGraphViewer,

  // Impact
  'impact-score': ImpactScorePage,
  'project-leaderboard': ProjectLeaderboard,

  // Collaboration
  'collaborator-suggestions': CollaboratorSuggestionPage,
  'match-history': MatchHistoryList,
  'expert-directory': ExpertDirectoryPage,
  'project-application': ProjectApplicationPanel,

  // Gamification & Streaks
  'streak-heatmap': StreakHeatmapPage,
  'weekly-summary': WeeklySummaryViewer,
  'badge-wall': BadgeWallPage,
  leaderboard: ContributionLeaderboard,

  // Admin & Org
  admin: AdminDashboard,
  cluster: ClusterOverviewPage,
  organizations: OrganizationListPage,
  community: QuestionListPage,
  profile: UserProfileWindow,

  // 3D & Federated
  model3d: ModelViewerPage,
  federated: FLRunPage,

  // Anomaly & HPO (additional)
  anomaly: AnomalyListPage,
  anomaly_detail: AnomalyDetailPage,
  hpo_runs: HPORunsPage,
}

// ========================
// 4. QueryClient
// ========================
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 10000,
    },
  },
})

// ========================
// 5. Main App Component
// ========================
function App() {
  const { windows } = useWindowStore()

  return (
    <QueryClientProvider client={queryClient}>
      <HardwareProvider>
        {/* Main Router for full‑page routes (authentication, dashboard, etc.) */}
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
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

        {/* Floating window system (always on top, independent of router) */}
        <div className="relative w-screen h-screen overflow-hidden bg-surface pointer-events-none">
          <div className="pointer-events-auto">
            {windows.map(win => {
              const Component = componentMap[win.type]
              if (!Component) return null
              return (
                <DockableWindow key={win.id} window={win}>
                  <Component onClose={() => useWindowStore.getState().removeWindow(win.id)} />
                </DockableWindow>
              )
            })}
            <WindowLauncher />
            <WorkspaceManager />
            <FloatingWindowRenderer />
            <FloatingWindowBar />
            <Toaster />
          </div>
        </div>
      </HardwareProvider>
    </QueryClientProvider>
  )
}

export default App



// src/App.tsx (partial)
import LabsListPage from './features/communityLab/pages/LabsListPage'
import LabDetailPage from './features/communityLab/pages/LabDetailPage'

const componentMap = {
  // ...
  labs: LabsListPage,
  lab_detail: LabDetailPage,
}


import DatasetListPage from './features/datasets/pages/DatasetListPage'
import DatasetDetailPage from './features/datasets/pages/DatasetDetailPage'

const componentMap = {
  // ...
  datasets: DatasetListPage,
  dataset_detail: DatasetDetailPage,
}



import LLMProviderList from './features/llmHub/pages/LLMProviderList'
import LLMPlayground from './features/llmHub/pages/LLMPlayground'
import PromptLibrary from './features/llmHub/pages/PromptLibrary'
import EvaluationSuiteRunner from './features/llmHub/pages/EvaluationSuiteRunner'
import CostDashboard from './features/llmHub/pages/CostDashboard'
import MemoryManager from './features/llmHub/pages/MemoryManager'
import ToolCatalog from './features/llmHub/pages/ToolCatalog'
import ArenaBattlePage from './features/llmHub/pages/ArenaBattlePage'
import GuardrailEditor from './features/llmHub/pages/GuardrailEditor'

const componentMap = {
  // ...
  'llm-providers': LLMProviderList,
  'playground': LLMPlayground,
  'prompts': PromptLibrary,
  'evaluations': EvaluationSuiteRunner,
  'costs': CostDashboard,
  'memory': MemoryManager,
  'tools': ToolCatalog,
  'arena': ArenaBattlePage,
  'guardrails': GuardrailEditor,
}



import FeatureStoreBrowser from './features/mlops/pages/FeatureStoreBrowser'
import DriftMonitorList from './features/mlops/pages/DriftMonitorList'
import AlertDashboard from './features/mlops/pages/AlertDashboard'
import PerformanceSnapshotView from './features/mlops/pages/PerformanceSnapshotView'

const componentMap = {
  // ...
  'feature-store': FeatureStoreBrowser,
  'drift-monitors': DriftMonitorList,
  'alerts': AlertDashboard,
  'performance': PerformanceSnapshotView,
}