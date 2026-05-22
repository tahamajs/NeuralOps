// components/DockableWindow.tsx
import { useRef, useState, useEffect } from 'react'
import { Rnd } from 'react-rnd'
import { useWindowStore } from '../store/windowStore'
import { X, Minus, Square, Maximize2 } from 'lucide-react'

const SNAP_THRESHOLD = 20
const DOCK_THRESHOLD = 50

export function DockableWindow({ window: win, children }: { window: WindowItem; children: React.ReactNode }) {
  const { updateWindow, bringToFront, toggleMinimize, toggleMaximize, removeWindow, groupWindows } = useWindowStore()
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const rndRef = useRef<any>()

  const handleDragStop = (e: any, d: { x: number; y: number }) => {
    // Check for docking with other windows (simplified: just check if overlapping with any other)
    const otherWindows = useWindowStore.getState().windows.filter(w => w.id !== win.id && !w.groupId)
    for (const other of otherWindows) {
      const overlapX = Math.abs(d.x - other.x) < DOCK_THRESHOLD
      const overlapY = Math.abs(d.y - other.y) < DOCK_THRESHOLD
      if (overlapX && overlapY) {
        groupWindows(win.id, other.id)
        return
      }
    }
    // Save position
    updateWindow(win.id, { x: d.x, y: d.y })
  }

  // Window controls
  const handleMinimize = () => toggleMinimize(win.id)
  const handleMaximize = () => toggleMaximize(win.id)
  const handleClose = () => removeWindow(win.id)

  if (win.minimized) {
    return (
      <div
        className="fixed bottom-4 left-4 z-50 glass px-3 py-1 rounded-lg cursor-pointer text-sm"
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
        onDragStop: handleDragStop,
        onResizeStop: (e, dir, ref, delta, position) => {
          updateWindow(win.id, { width: ref.offsetWidth, height: ref.offsetHeight, x: position.x, y: position.y })
        },
        minWidth: 300,
        minHeight: 200,
        style: { zIndex: win.zIndex },
      }

  return (
    <Rnd
      ref={rndRef}
      {...rndProps}
      className="glass border border-white/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
      onMouseDown={() => bringToFront(win.id)}
    >
      {/* Title Bar */}
      <div className="flex items-center justify-between bg-black/30 px-3 py-2 cursor-move">
        <div className="flex items-center gap-2 text-sm font-medium">
          {win.icon && <span>{win.icon}</span>}
          <span>{win.title}</span>
        </div>
        <div className="flex gap-1">
          <button onClick={handleMinimize} className="p-1 hover:bg-white/10 rounded"><Minus size={14} /></button>
          <button onClick={handleMaximize} className="p-1 hover:bg-white/10 rounded"><Maximize2 size={14} /></button>
          <button onClick={handleClose} className="p-1 hover:bg-red-500/20 rounded"><X size={14} /></button>
        </div>
      </div>
      {/* Content */}
      <div className="flex-1 overflow-auto bg-black/20">
        {children}
      </div>
    </Rnd>
  )
}