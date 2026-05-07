import { create } from 'zustand'
import { ReactNode } from 'react'

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
}

interface FloatingWindowState {
  windows: FloatingWindow[]
  focusedId: string | null
  addWindow: (win: FloatingWindow) => void
  removeWindow: (id: string) => void
  toggleMinimize: (id: string) => void
  focusWindow: (id: string) => void
}

export const useFloatingWindowStore = create<FloatingWindowState>((set) => ({
  windows: [],
  focusedId: null,
  addWindow: (win) =>
    set((state) => ({
      windows: [...state.windows.filter((w) => w.id !== win.id), win],
      focusedId: win.id,
    })),
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
    })),
  focusWindow: (id) => set({ focusedId: id }),
}))

export const WINDOW_TYPES = {
  TERMINAL: 'terminal',
  PAPER_EDITOR: 'paper-editor',
  NOTES_EDITOR: 'notes-editor',
  SYSTEM_MONITOR: 'system-monitor',
} as const