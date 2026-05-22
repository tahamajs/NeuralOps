import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface WindowItem {
  id: string
  type: string   // 'terminal', 'paper-editor', 'notes', 'marketplace', etc.
  title: string
  icon?: string
  width: number
  height: number
  x: number
  y: number
  zIndex: number
  minimized: boolean
  maximized: boolean
  groupId?: string   // if part of a tab group
  isFloating: boolean
  // Store the actual component content is separate; we use type to render.
}

interface WindowStore {
  windows: WindowItem[]
  nextZIndex: number
  addWindow: (type: string, title: string, icon?: string) => void
  removeWindow: (id: string) => void
  updateWindow: (id: string, updates: Partial<WindowItem>) => void
  bringToFront: (id: string) => void
  toggleMinimize: (id: string) => void
  toggleMaximize: (id: string) => void
  // Dock / group
  groupWindows: (sourceId: string, targetId: string) => void
  ungroupWindow: (id: string) => void
  // Layout persistence
  saveLayout: (name: string) => void
  loadLayout: (name: string) => void
}

const DEFAULT_WIDTH = 800
const DEFAULT_HEIGHT = 600
const CASCADE_OFFSET = 30

export const useWindowStore = create<WindowStore>()(
  persist(
    (set, get) => ({
      windows: [],
      nextZIndex: 100,

      addWindow: (type, title, icon) => {
        const existingCount = get().windows.length
        const newZ = get().nextZIndex + 1
        const newWindow: WindowItem = {
          id: crypto.randomUUID(),
          type,
          title,
          icon,
          width: DEFAULT_WIDTH,
          height: DEFAULT_HEIGHT,
          x: 100 + (existingCount * CASCADE_OFFSET) % 300,
          y: 100 + (existingCount * CASCADE_OFFSET) % 300,
          zIndex: newZ,
          minimized: false,
          maximized: false,
          isFloating: true,
        }
        set(state => ({
          windows: [...state.windows, newWindow],
          nextZIndex: newZ,
        }))
      },

      removeWindow: (id) => set(state => ({
        windows: state.windows.filter(w => w.id !== id)
      })),

      updateWindow: (id, updates) => set(state => ({
        windows: state.windows.map(w => w.id === id ? { ...w, ...updates } : w)
      })),

      bringToFront: (id) => {
        const newZ = get().nextZIndex + 1
        set(state => ({
          windows: state.windows.map(w => w.id === id ? { ...w, zIndex: newZ } : w),
          nextZIndex: newZ,
        }))
      },

      toggleMinimize: (id) => set(state => ({
        windows: state.windows.map(w => w.id === id ? { ...w, minimized: !w.minimized } : w)
      })),

      toggleMaximize: (id) => set(state => ({
        windows: state.windows.map(w => w.id === id ? { ...w, maximized: !w.maximized } : w)
      })),

      groupWindows: (sourceId, targetId) => set(state => {
        const source = state.windows.find(w => w.id === sourceId)
        const target = state.windows.find(w => w.id === targetId)
        if (!source || !target) return state
        const groupId = target.groupId || target.id
        return {
          windows: state.windows.map(w => {
            if (w.id === sourceId) return { ...w, groupId, isFloating: false }
            if (w.id === targetId && !target.groupId) return { ...w, groupId, isFloating: false }
            return w
          })
        }
      }),

      ungroupWindow: (id) => set(state => ({
        windows: state.windows.map(w => w.id === id ? { ...w, groupId: undefined, isFloating: true } : w)
      })),

      saveLayout: (name) => {
        const layout = get().windows.map(({ id, type, title, icon, width, height, x, y, groupId }) => ({
          id, type, title, icon, width, height, x, y, groupId
        }))
        localStorage.setItem(`workspace-${name}`, JSON.stringify(layout))
      },

      loadLayout: (name) => {
        const raw = localStorage.getItem(`workspace-${name}`)
        if (raw) {
          const layout = JSON.parse(raw)
          set({ windows: layout.map((w: any) => ({ ...w, zIndex: 0, minimized: false, maximized: false })) })
        }
      },
    }),
    { name: 'neuralops-windows' }
  )
)