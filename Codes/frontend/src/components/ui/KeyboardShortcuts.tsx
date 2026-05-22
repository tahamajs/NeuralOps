import { useState, useEffect } from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { Command } from 'lucide-react'

const shortcuts = [
  { keys: ['⌘', 'K'], description: 'Open command palette' },
  { keys: ['⌘', 'S'], description: 'Save current file/note' },
  { keys: ['⌘', 'F'], description: 'Search' },
  { keys: ['⌘', 'Shift', 'P'], description: 'Show command palette' },
  { keys: ['⌘', 'N'], description: 'New experiment/note' },
  { keys: ['⌘', 'B'], description: 'Toggle sidebar' },
  { keys: ['⌘', '⌫'], description: 'Delete selected item' },
  { keys: ['Esc'], description: 'Close modal / cancel' },
  { keys: ['⌘', '?'], description: 'Show this help' },
]

export function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === '?') {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <Modal open={open} onClose={() => setOpen(false)} title="Keyboard Shortcuts">
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {shortcuts.map((shortcut, idx) => (
          <div key={idx} className="flex justify-between items-center py-2 border-b border-white/5">
            <div className="flex gap-1">
              {shortcut.keys.map((key, i) => (
                <kbd
                  key={i}
                  className="px-2 py-0.5 rounded bg-white/10 text-white text-xs font-mono"
                >
                  {key}
                </kbd>
              ))}
            </div>
            <span className="text-sm text-slate-300">{shortcut.description}</span>
          </div>
        ))}
      </div>
      <Button variant="ghost" size="sm" className="mt-4" onClick={() => setOpen(false)}>
        Close
      </Button>
    </Modal>
  )
}