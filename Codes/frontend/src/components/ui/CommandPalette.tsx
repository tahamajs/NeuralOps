import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, FileText, FlaskConical, BookOpen, BarChart3, Settings, Terminal, Users } from 'lucide-react'

interface Command {
  id: string
  title: string
  description?: string
  icon?: React.ReactNode
  action: () => void
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  const commands: Command[] = [
    { id: 'dashboard', title: 'Go to Dashboard', icon: <BarChart3 size={16} />, action: () => navigate('/dashboard') },
    { id: 'experiments', title: 'View Experiments', icon: <FlaskConical size={16} />, action: () => navigate('/experiments') },
    { id: 'launcher', title: 'Launch New Experiment', icon: <FlaskConical size={16} />, action: () => navigate('/launcher') },
    { id: 'notes', title: 'Open Notes', icon: <BookOpen size={16} />, action: () => navigate('/notes') },
    { id: 'paper', title: 'New Paper', icon: <FileText size={16} />, action: () => navigate('/paper/new') },
    { id: 'terminal', title: 'Open Terminal', icon: <Terminal size={16} />, action: () => useFloatingWindowStore.getState().addWindow({ id: 'terminal', title: 'Terminal', icon: '>_', content: 'terminal' }) },
    { id: 'profile', title: 'Edit Profile', icon: <Users size={16} />, action: () => navigate('/profile') },
    { id: 'settings', title: 'Settings', icon: <Settings size={16} />, action: () => navigate('/profile/settings') },
  ]

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const filtered = commands.filter(cmd =>
    cmd.title.toLowerCase().includes(search.toLowerCase()) ||
    cmd.description?.toLowerCase().includes(search.toLowerCase())
  )

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[400] flex items-start justify-center pt-20 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)}>
      <div className="glass w-full max-w-xl rounded-2xl shadow-2xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 p-4 border-b border-white/10">
          <Search size={18} className="text-slate-400" />
          <input
            type="text"
            autoFocus
            placeholder="Type a command..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 bg-transparent outline-none text-white placeholder-slate-500"
          />
          <kbd className="px-2 py-0.5 rounded bg-white/10 text-slate-400 text-xs">ESC</kbd>
        </div>
        <div className="max-h-96 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <div className="text-center text-slate-500 py-8">No commands found</div>
          ) : (
            filtered.map((cmd) => (
              <button
                key={cmd.id}
                onClick={() => {
                  cmd.action()
                  setOpen(false)
                }}
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <span className="text-slate-400">{cmd.icon}</span>
                <div className="text-left">
                  <div className="text-white text-sm">{cmd.title}</div>
                  {cmd.description && <div className="text-xs text-slate-500">{cmd.description}</div>}
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  )
}