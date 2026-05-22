// components/WindowLauncher.tsx
import { useState, useEffect } from 'react'
import { useWindowStore } from '../store/windowStore'
import { Command, Terminal, FileText, StickyNote, BarChart3, Users, Settings, Search } from 'lucide-react'

const AVAILABLE_WINDOWS = [
  { type: 'terminal', title: 'Terminal', icon: '>_', component: 'TerminalPanel' },
  { type: 'paper-editor', title: 'Paper Editor', icon: '📄', component: 'PaperEditorPanel' },
  { type: 'notes', title: 'Notes', icon: '📝', component: 'FloatingNotesEditor' },
  { type: 'marketplace', title: 'GPU Marketplace', icon: '🖥️', component: 'GPUMarketplacePage' },
  { type: 'experiments', title: 'Experiments', icon: '🔬', component: 'ExperimentListPage' },
  { type: 'analysis', title: 'Analysis', icon: '📊', component: 'ResearchAnalysisPage' },
  { type: 'profile', title: 'Profile', icon: '👤', component: 'UserProfilePage' },
  { type: 'community', title: 'Q&A Forum', icon: '💬', component: 'QuestionListPage' },
]

export function WindowLauncher() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const { addWindow } = useWindowStore()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'p') {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const filtered = AVAILABLE_WINDOWS.filter(w =>
    w.title.toLowerCase().includes(query.toLowerCase())
  )

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[1000] bg-black/50 backdrop-blur-sm flex items-start justify-center pt-20" onClick={() => setOpen(false)}>
      <div className="glass w-full max-w-lg rounded-xl shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-2 p-4 border-b border-white/10">
          <Command size={18} />
          <input
            autoFocus
            type="text"
            placeholder="Open a window... (Terminal, Notes, Marketplace)"
            className="flex-1 bg-transparent outline-none text-white"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
        <div className="max-h-96 overflow-y-auto p-2">
          {filtered.map(item => (
            <button
              key={item.type}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 rounded-lg transition-colors text-left"
              onClick={() => {
                addWindow(item.type, item.title, item.icon)
                setOpen(false)
              }}
            >
              <span className="text-2xl">{item.icon}</span>
              <div>
                <div className="text-white">{item.title}</div>
                <div className="text-xs text-slate-500">Open as new window</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}