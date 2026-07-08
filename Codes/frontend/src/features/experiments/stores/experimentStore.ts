import { create } from "zustand";

interface ExperimentState {
  status: string;
  logs: string[];
  metrics: { loss: number[]; accuracy: number[] };
  progress: { current: number; total: number; eta: number };
  updateStatus: (status: string) => void;
  addLog: (line: string) => void;
  updateMetrics: (data: any) => void;
  updateProgress: (progress: any) => void;
  reset: () => void;
}

export const useLiveStore = create<ExperimentState>((set) => ({
  status: "",
  logs: [],
  metrics: { loss: [], accuracy: [] },
  progress: { current: 0, total: 0, eta: 0 },
  updateStatus: (status) => set({ status }),
  addLog: (line) =>
    set((state) => ({ logs: [...state.logs, line].slice(-200) })),
  updateMetrics: (data) => set({ metrics: data }),
  updateProgress: (progress) => set({ progress }),
  reset: () =>
    set({
      status: "",
      logs: [],
      metrics: { loss: [], accuracy: [] },
      progress: { current: 0, total: 0, eta: 0 },
    }),
}));
