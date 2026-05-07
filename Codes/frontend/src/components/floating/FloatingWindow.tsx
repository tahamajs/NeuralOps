import { Suspense, lazy } from 'react'
import { useFloatingWindowStore } from './FloatingWindowManager'

// Lazy‑loaded content renderers
const TerminalPanel = lazy(() => import('../../features/terminal/components/TerminalPanel'))
const PaperEditorPanel = lazy(() => import('../../features/paper/components/PaperEditorPanel'))
const NotesEditorPanel = lazy(() => import('../../features/notes/components/FloatingNotesEditor'))
const SystemMonitorPanel = lazy(() => import('../../features/hardware/components/SystemMonitorPanel'))

const CONTENT_RENDERERS: Record<string, React.ComponentType<any>> = {
  terminal: TerminalPanel,
  'paper-editor': PaperEditorPanel,
  'notes-editor': NotesEditorPanel,
  'system-monitor': SystemMonitorPanel,
}

export default function FloatingWindowRenderer() {
  const { windows, removeWindow, toggleMinimize, focusWindow, bringToFront, focusedId } =
    useFloatingWindowStore()

  return (
    <>
      {windows.map((win) => {
        if (win.minimized) {
          return (
            <div
              key={win.id}
              onClick={() => {
                bringToFront(win.id)
                toggleMinimize(win.id)
              }}
              className={`fixed bottom-4 right-4 z-[200] glass px-4 py-2 rounded-xl cursor-pointer flex items-center gap-2 text-xs font-bold text-white transition-all hover:scale-105 ${
                focusedId === win.id ? 'border-neon-blue shadow-neon-blue/20 shadow-lg' : ''
              }`}
            >
              <span>{win.icon || '▣'}</span>
              {win.title}
            </div>
          )
        }

        const ContentRenderer = CONTENT_RENDERERS[win.content as string] || null

        return (
          <div
            key={win.id}
            className={`fixed glass border rounded-2xl overflow-hidden shadow-2xl transition-shadow ${
              focusedId === win.id ? 'border-neon-blue/50 shadow-neon-blue/20' : 'border-white/10'
            }`}
            style={{
              left: win.x ?? 100,
              top: win.y ?? 100,
              width: win.width ?? 600,
              height: win.height ?? 500,
              minWidth: 300,
              minHeight: 200,
              zIndex: win.zIndex ?? 100,
            }}
            onMouseDown={() => {
              focusWindow(win.id)
              bringToFront(win.id)
            }}
          >
            {/* Custom Title Bar */}
            <div className="h-9 bg-cyber-800 border-b border-white/5 flex items-center justify-between px-3 cursor-grab active:cursor-grabbing select-none">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
                <span>{win.icon || '▣'}</span>
                <span className="uppercase tracking-wider">{win.title}</span>
              </div>
              <div className="flex gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleMinimize(win.id)
                  }}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 text-slate-400 hover:text-white text-xs"
                  title="Minimize"
                >
                  —
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeWindow(win.id)
                  }}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-red-500/20 text-slate-400 hover:text-red-400 text-xs"
                  title="Close"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="h-[calc(100%-36px)] overflow-auto">
              <Suspense
                fallback={
                  <div className="flex items-center justify-center h-full text-slate-400 text-xs">
                    Loading...
                  </div>
                }
              >
                {ContentRenderer ? (
                  <ContentRenderer onClose={() => removeWindow(win.id)} />
                ) : (
                  <div className="p-4 text-slate-400">Unknown window type</div>
                )}
              </Suspense>
            </div>
          </div>
        )
      })}
    </>
  )
}