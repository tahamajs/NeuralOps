import { useState, useEffect } from 'react'
import { useWindowStore } from '../store/windowStore'

const AVAILABLE_WINDOWS = [
  { type: 'terminal', title: 'Terminal', icon: '>_' },
  { type: 'chat', title: 'AI Chat', icon: '💬' },
  { type: 'notes', title: 'Notes', icon: '📝' },
  { type: 'paper-editor', title: 'Paper Editor', icon: '📄' },
  { type: 'marketplace', title: 'GPU Marketplace', icon: '🖥️' },
  { type: 'experiments', title: 'Experiments', icon: '🔬' },
  { type: 'community', title: 'Q&A Forum', icon: '❓' },
  { type: 'profile', title: 'Profile', icon: '👤' },
  { type: 'analysis', title: 'Research Analysis', icon: '📊' },
  { type: 'analytics', title: 'Advanced Analytics', icon: '📈' },
  { type: 'admin', title: 'Admin Dashboard', icon: '⚙️' },
  { type: 'cluster', title: 'Cluster Overview', icon: '🌐' },
  { type: 'organizations', title: 'Organizations', icon: '🏢' },
  { type: 'sweeps', title: 'Sweeps', icon: '🎯' },
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

  if (!open) return null

  const filtered = AVAILABLE_WINDOWS.filter(w => w.title.toLowerCase().includes(query.toLowerCase()))

  return (
    <div className="fixed inset-0 z-[1000] bg-black/50 backdrop-blur-sm flex items-start justify-center pt-20" onClick={() => setOpen(false)}>
      <div className="glass-terminal w-full max-w-md rounded-xl shadow-2xl" onClick={e => e.stopPropagation()}>
        <input
          autoFocus
          type="text"
          placeholder="Open a window... (Terminal, Chat, Notes)"
          className="w-full bg-transparent border-b border-white/10 p-4 outline-none text-white"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <div className="max-h-80 overflow-y-auto p-2">
          {filtered.map(item => (
            <button
              key={item.type}
              className="w-full flex items-center gap-3 px-3 py-2 hover:bg-white/5 rounded text-left"
              onClick={() => {
                addWindow(item.type, item.title, item.icon)
                setOpen(false)
              }}
            >
              <span className="text-xl">{item.icon}</span>
              <span>{item.title}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

const AVAILABLE_WINDOWS = [
  // ... existing
  { type: 'model3d', title: '3D Model Explorer', icon: '🧊' },
  { type: 'federated', title: 'Federated Learning Dashboard', icon: '🔗' },
];

// In WindowLauncher.tsx
const AVAILABLE_WINDOWS = [
  // ... existing entries
  { type: 'support', title: 'AI Support Hub', icon: '🎓' },
];


const AVAILABLE_WINDOWS = [
  // ... existing entries
  { type: 'gpumarket', title: 'GPU Stock Market', icon: '📈', component: GPUTradingTerminal },
];



const AVAILABLE_WINDOWS = [
  // ... existing
  { type: 'anomaly', title: 'Anomaly Detection', icon: '⚠️' },
  { type: 'hpo', title: 'HPO Sweeps', icon: '🎯' },
]


const AVAILABLE_WINDOWS = [
  // ... existing
  { type: 'hpo', title: 'HPO Dashboard', icon: '🎯' },
]


const AVAILABLE_WINDOWS = [
  // ...
  { type: 'recommendations', title: 'Experiment Recommendations', icon: '💡' },
]


const AVAILABLE_WINDOWS = [
  // ... existing
  { type: 'forensics', title: 'Failure Forensics', icon: '🛡️' },
]



// src/components/WindowLauncher.tsx
const AVAILABLE_WINDOWS = [
  // ... existing
  { type: 'tournament', title: 'Idea Tournament', icon: '🏆' },
  { type: 'hypotheses', title: 'Hypothesis Browser', icon: '🔬' },
]


// WindowLauncher.tsx
const AVAILABLE_WINDOWS = [
  { type: 'literature-map', title: 'Literature Map', icon: '🗺️' },
  { type: 'citation-graph', title: 'Citation Graph Viewer', icon: '🔗' },
]