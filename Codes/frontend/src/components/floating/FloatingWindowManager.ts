import { create } from "zustand";

export interface FloatingWindow {
  id: string;
  title: string;
  icon?: string;
  content: string; // matches CONTENT_RENDERERS key
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  minimized?: boolean;
  maximized?: boolean;
  zIndex?: number;
  prevWidth?: number;
  prevHeight?: number;
  prevX?: number;
  prevY?: number;
}

interface FloatingWindowState {
  windows: FloatingWindow[];
  focusedId: string | null;
  maxZIndex: number;
  addWindow: (win: FloatingWindow, avoidDuplicate?: boolean) => void;
  removeWindow: (id: string) => void;
  toggleMinimize: (id: string) => void;
  toggleMaximize: (id: string) => void;
  updatePosition: (id: string, x: number, y: number) => void;
  updateSize: (id: string, width: number, height: number) => void;
  focusWindow: (id: string) => void;
  bringToFront: (id: string) => void;
}

const DEFAULT_WIDTH = 700;
const DEFAULT_HEIGHT = 500;
const MIN_WIDTH = 300;
const MIN_HEIGHT = 200;

// Default positions for new windows (cascade)
let nextX = 100;
let nextY = 100;
const STEP = 30;
const MAX_CASCADE = 5;

const getNextPosition = () => {
  const pos = { x: nextX, y: nextY };
  nextX = (nextX + STEP) % (STEP * MAX_CASCADE);
  nextY = (nextY + STEP) % (STEP * MAX_CASCADE);
  return pos;
};

export const useFloatingWindowStore = create<FloatingWindowState>(
  (set, get) => ({
    windows: [],
    focusedId: null,
    maxZIndex: 100,

    addWindow: (win, avoidDuplicate = true) => {
      const state = get();
      if (avoidDuplicate) {
        const existing = state.windows.find(
          (w) => w.id === win.id && !w.minimized,
        );
        if (existing) {
          // Restore and focus
          set({
            windows: state.windows.map((w) =>
              w.id === win.id
                ? {
                    ...w,
                    minimized: false,
                    maximized: false,
                    zIndex: state.maxZIndex + 1,
                  }
                : w,
            ),
            focusedId: win.id,
            maxZIndex: state.maxZIndex + 1,
          });
          return;
        }
        const minimized = state.windows.find(
          (w) => w.id === win.id && w.minimized,
        );
        if (minimized) {
          set({
            windows: state.windows.map((w) =>
              w.id === win.id
                ? { ...w, minimized: false, zIndex: state.maxZIndex + 1 }
                : w,
            ),
            focusedId: win.id,
            maxZIndex: state.maxZIndex + 1,
          });
          return;
        }
      }

      const newPos = getNextPosition();
      const newZ = state.maxZIndex + 1;
      set({
        windows: [
          ...state.windows.filter((w) => w.id !== win.id),
          {
            ...win,
            x: win.x ?? newPos.x,
            y: win.y ?? newPos.y,
            width: win.width ?? DEFAULT_WIDTH,
            height: win.height ?? DEFAULT_HEIGHT,
            minimized: false,
            maximized: false,
            zIndex: newZ,
          },
        ],
        focusedId: win.id,
        maxZIndex: newZ,
      });
    },

    removeWindow: (id) =>
      set((state) => ({
        windows: state.windows.filter((w) => w.id !== id),
        focusedId: state.focusedId === id ? null : state.focusedId,
      })),

    toggleMinimize: (id) =>
      set((state) => ({
        windows: state.windows.map((w) =>
          w.id === id ? { ...w, minimized: !w.minimized, maximized: false } : w,
        ),
        focusedId: state.windows.find((w) => w.id === id)?.minimized
          ? state.focusedId
          : id,
      })),

    toggleMaximize: (id) =>
      set((state) => {
        const win = state.windows.find((w) => w.id === id);
        if (!win) return state;
        if (win.maximized) {
          // Restore previous size/position (we store them in separate fields)
          return {
            windows: state.windows.map((w) =>
              w.id === id
                ? {
                    ...w,
                    maximized: false,
                    width: w.prevWidth ?? DEFAULT_WIDTH,
                    height: w.prevHeight ?? DEFAULT_HEIGHT,
                    x: w.prevX,
                    y: w.prevY,
                  }
                : w,
            ),
          };
        } else {
          // Save current dimensions, then maximize
          return {
            windows: state.windows.map((w) =>
              w.id === id
                ? {
                    ...w,
                    prevWidth: w.width,
                    prevHeight: w.height,
                    prevX: w.x,
                    prevY: w.y,
                    maximized: true,
                    width: window.innerWidth - 40, // leave some margin
                    height: window.innerHeight - 40,
                    x: 20,
                    y: 20,
                  }
                : w,
            ),
          };
        }
      }),

    updatePosition: (id, x, y) =>
      set((state) => ({
        windows: state.windows.map((w) => (w.id === id ? { ...w, x, y } : w)),
      })),

    updateSize: (id, width, height) =>
      set((state) => ({
        windows: state.windows.map((w) =>
          w.id === id ? { ...w, width, height } : w,
        ),
      })),

    focusWindow: (id) => set({ focusedId: id }),

    bringToFront: (id) => {
      const state = get();
      const newZ = state.maxZIndex + 1;
      set({
        windows: state.windows.map((w) =>
          w.id === id ? { ...w, zIndex: newZ } : w,
        ),
        maxZIndex: newZ,
        focusedId: id,
      });
    },
  }),
);

export const WINDOW_TYPES = {
  TERMINAL: "terminal",
  CHAT: "chat",
  PAPER_EDITOR: "paper-editor",
  NOTES_EDITOR: "notes-editor",
  SYSTEM_MONITOR: "system-monitor",
  EXPERIMENTS: "experiments",
  ANALYSIS: "analysis",
  ACCELERATOR: "accelerator",
  HPO: "hpo",
  ANOMALY: "anomaly",
  FORENSICS: "forensics",
  SUPPORT: "support",
} as const;
