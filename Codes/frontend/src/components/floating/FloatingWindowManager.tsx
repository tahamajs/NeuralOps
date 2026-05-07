import { create } from 'zustand'

export interface FloatingWindow {
  id: string
  title: string
  icon?: string
  content: string
  width?: number
  height?: number
  x?: number
  y?: number
  minimized?: boolean
  zIndex?: number
}

interface FloatingWindowState {
  windows: FloatingWindow[]
  focusedId: string | null
  maxZIndex: number
  addWindow: (win: FloatingWindow, avoidDuplicate?: boolean) => void
  removeWindow: (id: string) => void
  toggleMinimize: (id: string) => void
  focusWindow: (id: string) => void
  bringToFront: (id: string) => void
}

// Default positions for each window type to avoid stacking on top of each other
const defaultPositions: Record<string, { x: number; y: number }> = {
  terminal: { x: 80, y: 80 },
  'paper-editor': { x: 320, y: 120 },
  'notes-editor': { x: 560, y: 160 },
  'system-monitor': { x: 800, y: 80 },
}

export const useFloatingWindowStore = create<FloatingWindowState>((set, get) => ({
  windows: [],
  focusedId: null,
  maxZIndex: 100,

  addWindow: (win, avoidDuplicate = true) => {
    const state = get()

    // If we want to avoid duplicates, check if a window with the same content type already exists
    if (avoidDuplicate) {
      const existing = state.windows.find((w) => w.id === win.id && !w.minimized)
      if (existing) {
        // Just focus it
        set({
          focusedId: win.id,
          windows: state.windows.map((w) =>
            w.id === win.id ? { ...w, minimized: false, zIndex: state.maxZIndex + 1 } : w
          ),
          maxZIndex: state.maxZIndex + 1,
        })
        return
      }
      // If minimized, restore it
      const minimized = state.windows.find((w) => w.id === win.id && w.minimized)
      if (minimized) {
        set({
          windows: state.windows.map((w) =>
            w.id === win.id ? { ...w, minimized: false, zIndex: state.maxZIndex + 1 } : w
          ),
          focusedId: win.id,
          maxZIndex: state.maxZIndex + 1,
        })
        return
      }
    }

    // Use default position if none provided
    const defaults = defaultPositions[win.id] || { x: 100, y: 100 }

    const newZ = state.maxZIndex + 1
    set({
      windows: [...state.windows.filter((w) => w.id !== win.id), { ...win, x: win.x ?? defaults.x, y: win.y ?? defaults.y, zIndex: newZ }],
      focusedId: win.id,
      maxZIndex: newZ,
    })
  },

  removeWindow: (id) =>
    set((state) => ({
      windows: state.windows.filter((w) => w.id !== id),
      focusedId: state.focusedId === id ? null : state.focusedId,
    })),

  toggleMinimize: (id) =>
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === id ? { ...w, minimized: !w.minimized } : w
      ),
      focusedId: state.windows.find((w) => w.id === id)?.minimized ? state.focusedId : id,
    })),

  focusWindow: (id) => set({ focusedId: id }),

  bringToFront: (id) => {
    const state = get()
    const newZ = state.maxZIndex + 1
    set({
      windows: state.windows.map((w) =>
        w.id === id ? { ...w, zIndex: newZ } : w
      ),
      maxZIndex: newZ,
      focusedId: id,
    })
  },
}))

export const WINDOW_TYPES = {
  TERMINAL: 'terminal',
  PAPER_EDITOR: 'paper-editor',
  NOTES_EDITOR: 'notes-editor',
  SYSTEM_MONITOR: 'system-monitor',
} as const