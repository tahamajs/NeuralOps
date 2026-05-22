import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HardwareProvider } from './features/hardware/context/HardwareContext'
import { MainLayout } from './components/layout/MainLayout'
import FloatingWindowRenderer from './components/floating/FloatingWindow'
import FloatingWindowBar from './components/floating/FloatingWindowBar'
import { useWindowStore } from './store/windowStore'
import { DockableWindow } from './components/DockableWindow'
import { WindowLauncher } from './components/WindowLauncher'
import { TerminalPanel } from './features/terminal/components/TerminalPanel'
import { PaperEditorPanel } from './features/paper/components/PaperEditorPanel'
import { FloatingNotesEditor } from './features/notes/components/FloatingNotesEditor'
import { GPUMarketplacePage } from './features/marketplace/pages/GPUMarketplacePage'
const componentMap: Record<string, React.ComponentType<any>> = {
  terminal: TerminalPanel,
  'paper-editor': PaperEditorPanel,
  notes: FloatingNotesEditor,
  marketplace: GPUMarketplacePage,
  experiments: ExperimentListPage,
  analysis: ResearchAnalysisPage,
  profile: UserProfilePage,
  community: QuestionListPage,
  // add all others
}
import AnomalyListPage from './features/anomaly/pages/AnomalyListPage'
import AnomalyDetailPage from './features/anomaly/pages/AnomalyDetailPage'
import HPORunsPage from './features/hpo/pages/HPORunsPage'

const componentMap = {
  // ...
  anomaly: AnomalyListPage,
  anomaly_detail: AnomalyDetailPage,
  hpo: HPORunsPage,
}
// Pages
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
function App() {
  const { windows } = useWindowStore()

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-surface">
      {windows.map(win => {
        const Component = componentMap[win.type]
        if (!Component) return null
        return (
          <DockableWindow key={win.id} window={win}>
            <Component />
          </DockableWindow>
        )
      })}
      <WindowLauncher />
      {/* Optional: keep a minimal sidebar with quick actions */}
      <div className="fixed bottom-4 left-4 z-[100] flex gap-2">
        <button className="glass p-2 rounded-full" onClick={() => useWindowStore.getState().addWindow('terminal', 'Terminal', '>_')}>Terminal</button>
        <button className="glass p-2 rounded-full" onClick={() => useWindowStore.getState().addWindow('notes', 'Notes', '📝')}>Notes</button>
      </div>
    </div>
  )
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 10000,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HardwareProvider>
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
        {/* Floating windows – always on top */}
        <FloatingWindowRenderer />
        <FloatingWindowBar />
      </HardwareProvider>
    </QueryClientProvider>
  )
}

export default App



// src/App.tsx
import { useWindowStore } from './store/windowStore'
import { DockableWindow } from './components/DockableWindow'
import { WindowLauncher } from './components/WindowLauncher'
import { WorkspaceManager } from './components/WorkspaceManager'
import { ChatPanel } from './features/chat/components/ChatPanel'
import { TerminalPanel } from './features/terminal/components/TerminalPanel'
import { PaperEditorPanel } from './features/paper/components/PaperEditorPanel'
import { FloatingNotesEditor } from './features/notes/components/FloatingNotesEditor'
import { GPUMarketplacePage } from './features/marketplace/pages/GPUMarketplacePage'
import { ExperimentListPage } from './features/experiments/pages/ExperimentListPage'
import { UserProfilePage } from './features/profile/pages/UserProfilePage'
import { QuestionListPage } from './features/community/pages/QuestionListPage'
// import all other components you need

const componentMap: Record<string, React.ComponentType<any>> = {
  terminal: TerminalPanel,
  'paper-editor': PaperEditorPanel,
  notes: FloatingNotesEditor,
  marketplace: GPUMarketplacePage,
  experiments: ExperimentListPage,
  profile: UserProfilePage,
  community: QuestionListPage,
  chat: ChatPanel,
  // add more as needed
}

function App() {
  const { windows } = useWindowStore()

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-surface">
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
    </div>
  )
}

export default App


import { useWindowStore } from './store/windowStore'
import { DockableWindow } from './components/DockableWindow'
import { WindowLauncher } from './components/WindowLauncher'
import { WorkspaceManager } from './components/WorkspaceManager'
import { Toaster } from './components/ui/Toast'

// Import all windowable panels
import { TerminalPanel } from './features/terminal/components/TerminalPanel'
import { ChatPanel } from './features/chat/components/ChatPanel'
import { FloatingNotesEditor } from './features/notes/components/FloatingNotesEditor'
import { PaperEditorPanel } from './features/paper/components/PaperEditorPanel'
import { GPUMarketplacePage } from './features/marketplace/pages/GPUMarketplacePage'
import { ExperimentListPage } from './features/experiments/pages/ExperimentListPage'
import { UserProfilePage } from './features/profile/pages/UserProfilePage'
import { QuestionListPage } from './features/community/pages/QuestionListPage'
import { ResearchAnalysisPage } from './features/analysis/pages/ResearchAnalysisPage'
import { AdvancedAnalyticsPage } from './features/analytics/pages/AdvancedAnalyticsPage'
import { AdminDashboard } from './features/admin/pages/AdminDashboard'
import { ClusterOverviewPage } from './features/cluster/pages/ClusterOverviewPage'
import { OrganizationListPage } from './features/organizations/pages/OrganizationListPage'
import { SweepDashboardPage } from './features/sweeps/pages/SweepDashboardPage'

const componentMap: Record<string, React.ComponentType<any>> = {
  terminal: TerminalPanel,
  chat: ChatPanel,
  notes: FloatingNotesEditor,
  'paper-editor': PaperEditorPanel,
  marketplace: GPUMarketplacePage,
  experiments: ExperimentListPage,
  profile: UserProfilePage,
  community: QuestionListPage,
  analysis: ResearchAnalysisPage,
  analytics: AdvancedAnalyticsPage,
  admin: AdminDashboard,
  cluster: ClusterOverviewPage,
  organizations: OrganizationListPage,
  sweeps: SweepDashboardPage,
}

function App() {
  const { windows } = useWindowStore()
  return (
    <div className="relative w-screen h-screen overflow-hidden bg-surface">
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
      <Toaster />
    </div>
  )
}
export default App




import DeepResearchPage from './features/research/pages/DeepResearchPage';
// ...
const componentMap = {
  // ...
  research: DeepResearchPage,
  // ...
};

// In src/App.tsx
import { Model3DViewer } from './features/modelViewer/components/Model3DViewer';

const componentMap = {
  // ...
  model3d: Model3DViewer,
};

// In src/components/WindowLauncher.tsx
const AVAILABLE_WINDOWS = [
  // ...
  { type: 'model3d', title: '3D Model Explorer', icon: '🧊' },
];


import ModelViewerPage from './features/modelViewer/pages/ModelViewerPage';
import FLRunPage from './features/federated/pages/FLRunPage';

const componentMap: Record<string, React.ComponentType<any>> = {
  // ... existing
  model3d: ModelViewerPage,
  federated: FLRunPage,
};


import GPUTradingTerminal from './features/marketplace/pages/GPUTradingTerminal';
import { useWindowStore } from './store/windowStore';
const componentMap = {
  // ...
  gpumarket: GPUTradingTerminal,
};


import HPODashboard from './features/hpo/pages/HPODashboard'
import SweepDetailPage from './features/hpo/pages/SweepDetailPage'

const componentMap = {
  // ...
  hpo: HPODashboard,
  sweep_detail: SweepDetailPage,
}