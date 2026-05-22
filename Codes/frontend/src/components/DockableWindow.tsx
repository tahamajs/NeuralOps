import { useRef } from 'react'
import { Rnd } from 'react-rnd'
import { useWindowStore } from '../store/windowStore'
import { X, Minus, Square } from 'lucide-react'

export function DockableWindow({ window: win, children }: { window: any; children: React.ReactNode }) {
  const { updateWindow, bringToFront, toggleMinimize, toggleMaximize, removeWindow } = useWindowStore()
  const rndRef = useRef<any>()

  if (win.minimized) {
    return (
      <div
        className="fixed bottom-4 left-4 z-50 glass-terminal px-3 py-1 rounded cursor-pointer text-sm font-mono"
        onClick={() => toggleMinimize(win.id)}
      >
        {win.icon} {win.title}
      </div>
    )
  }

  const rndProps = win.maximized
    ? {
        position: { x: 0, y: 0 },
        size: { width: '100%', height: '100%' },
        disableDragging: true,
        enableResizing: false,
        style: { position: 'fixed', top: 0, left: 0, zIndex: win.zIndex },
      }
    : {
        position: { x: win.x, y: win.y },
        size: { width: win.width, height: win.height },
        onDragStop: (e, d) => updateWindow(win.id, { x: d.x, y: d.y }),
        onResizeStop: (e, dir, ref, delta, pos) =>
          updateWindow(win.id, { width: ref.offsetWidth, height: ref.offsetHeight, x: pos.x, y: pos.y }),
        minWidth: 300,
        minHeight: 200,
        style: { zIndex: win.zIndex },
      }

  return (
    <Rnd ref={rndRef} {...rndProps} className="glass-terminal flex flex-col overflow-hidden" onMouseDown={() => bringToFront(win.id)}>
      <div className="flex items-center justify-between bg-black/40 px-3 py-2 cursor-move border-b border-white/10">
        <div className="flex items-center gap-2 text-sm font-mono text-neon-mint">
          {win.icon && <span>{win.icon}</span>}
          <span>{win.title}</span>
        </div>
        <div className="flex gap-1">
          <button onClick={() => toggleMinimize(win.id)} className="p-1 hover:bg-white/10 rounded"><Minus size={14} /></button>
          <button onClick={() => toggleMaximize(win.id)} className="p-1 hover:bg-white/10 rounded"><Square size={14} /></button>
          <button onClick={() => removeWindow(win.id)} className="p-1 hover:bg-red-500/20 rounded"><X size={14} /></button>
        </div>
      </div>
      <div className="flex-1 overflow-auto bg-black/20">{children}</div>
    </Rnd>
  )
}