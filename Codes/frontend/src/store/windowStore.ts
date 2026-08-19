import { create } from "zustand";
import { persist } from "zustand/middleware";

/** crypto.randomUUID() is only available in HTTPS secure contexts.
 *  This polyfill works on plain HTTP (LAN access, etc.) as well. */
const uuid = (): string => {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  // Fallback: RFC 4122 v4 UUID using Math.random
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export interface WindowItem {
  id: string;
  type: string;
  title: string;
  icon?: string;
  width: number;
  height: number;
  x: number;
  y: number;
  zIndex: number;
  minimized: boolean;
  maximized: boolean;
  groupId?: string;
}

interface WindowStore {
  windows: WindowItem[];
  nextZIndex: number;
  addWindow: (type: string, title: string, icon?: string) => void;
  openWindow: (type: string, title: string, icon?: string) => void;
  removeWindow: (id: string) => void;
  closeAllWindows: () => void;
  updateWindow: (id: string, updates: Partial<WindowItem>) => void;
  bringToFront: (id: string) => void;
  toggleMinimize: (id: string) => void;
  toggleMaximize: (id: string) => void;
  groupWindows: (sourceId: string, targetId: string) => void;
  ungroup: (id: string) => void;
  saveLayout: (name: string) => void;
  loadLayout: (name: string) => void;
}

const DEFAULT_WIDTH = 800;
const DEFAULT_HEIGHT = 600;
const CASCADE = 30;

export const useWindowStore = create<WindowStore>()(
  persist(
    (set, get) => ({
      windows: [],
      nextZIndex: 100,
      openWindow: (type, title, icon) => get().addWindow(type, title, icon),
      closeAllWindows: () => set({ windows: [] }),
      addWindow: (type, title, icon) => {
        const count = get().windows.length;
        const newZ = get().nextZIndex + 1;
        set({
          windows: [
            ...get().windows,
            {
              id: uuid(),
              type,
              title,
              icon,
              width: DEFAULT_WIDTH,
              height: DEFAULT_HEIGHT,
              x: 80 + ((count * CASCADE) % 300),
              y: 80 + ((count * CASCADE) % 300),
              zIndex: newZ,
              minimized: false,
              maximized: false,
            },
          ],
          nextZIndex: newZ,
        });
      },
      removeWindow: (id) =>
        set({ windows: get().windows.filter((w) => w.id !== id) }),
      updateWindow: (id, updates) =>
        set({
          windows: get().windows.map((w) =>
            w.id === id ? { ...w, ...updates } : w,
          ),
        }),
      bringToFront: (id) => {
        const newZ = get().nextZIndex + 1;
        set({
          windows: get().windows.map((w) =>
            w.id === id ? { ...w, zIndex: newZ } : w,
          ),
          nextZIndex: newZ,
        });
      },
      toggleMinimize: (id) =>
        set({
          windows: get().windows.map((w) =>
            w.id === id ? { ...w, minimized: !w.minimized } : w,
          ),
        }),
      toggleMaximize: (id) =>
        set({
          windows: get().windows.map((w) =>
            w.id === id ? { ...w, maximized: !w.maximized } : w,
          ),
        }),
      groupWindows: (sourceId, targetId) => {
        const target = get().windows.find((w) => w.id === targetId);
        if (!target) return;
        const groupId = target.groupId || target.id;
        set({
          windows: get().windows.map((w) =>
            w.id === sourceId
              ? { ...w, groupId, x: target.x + 20, y: target.y + 20 }
              : w.id === targetId && !target.groupId
                ? { ...w, groupId }
                : w,
          ),
        });
      },
      ungroup: (id) =>
        set({
          windows: get().windows.map((w) =>
            w.id === id ? { ...w, groupId: undefined } : w,
          ),
        }),
      saveLayout: (name) => {
        const layout = get().windows.map(
          ({ id, type, title, icon, width, height, x, y, groupId }) => ({
            id,
            type,
            title,
            icon,
            width,
            height,
            x,
            y,
            groupId,
          }),
        );
        localStorage.setItem(`workspace-${name}`, JSON.stringify(layout));
      },
      loadLayout: (name) => {
        const raw = localStorage.getItem(`workspace-${name}`);
        if (raw) {
          const layout = JSON.parse(raw);
          set({
            windows: layout.map((w: any) => ({
              ...w,
              zIndex: 0,
              minimized: false,
              maximized: false,
            })),
            nextZIndex: 100,
          });
        }
      },
    }),
    { name: "neuralops-windows" },
  ),
);
