import { create } from 'zustand';
export const useLiveStore = create((set) => ({
    status: '',
    logs: [],
    metrics: { loss: [], accuracy: [] },
    progress: { current: 0, total: 0, eta: 0 },
    gpuInfo: { memoryUsed: 0, total: 0 },
    updateStatus: (status) => set({ status }),
    addLog: (msg) => set((s) => ({ logs: [...s.logs, msg] })),
    updateMetrics: (data) => set((s) => ({
        metrics: {
            loss: data.loss !== undefined ? [...s.metrics.loss, data.loss] : s.metrics.loss,
            accuracy: data.accuracy !== undefined ? [...s.metrics.accuracy, data.accuracy] : s.metrics.accuracy,
        },
    })),
    updateProgress: (data) => set((s) => ({
        progress: {
            current: data.current_step ?? s.progress.current,
            total: data.total_steps ?? s.progress.total,
            eta: data.eta_seconds ?? s.progress.eta,
        },
    })),
    updateGpu: (data) => set({ gpuInfo: { memoryUsed: data.memory_used_gb ?? 0, total: data.memory_total ?? 0 } }),
    reset: () => set({
        status: '',
        logs: [],
        metrics: { loss: [], accuracy: [] },
        progress: { current: 0, total: 0, eta: 0 },
        gpuInfo: { memoryUsed: 0, total: 0 },
    }),
}));
