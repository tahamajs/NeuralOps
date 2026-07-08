// src/components/ui/Toast.tsx
import { create } from "zustand";
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from "lucide-react";
import { cn } from "../../lib/utils";

type ToastType = "success" | "error" | "warning" | "info";

interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastStore {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, "id">) => void;
  removeToast: (id: string) => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = Math.random().toString(36).slice(2);
    set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
    setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
    }, toast.duration || 5000);
  },
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));

const icons = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const colors = {
  success: "border-green-500/50 bg-green-500/10 text-green-400",
  error: "border-red-500/50 bg-red-500/10 text-red-400",
  warning: "border-yellow-500/50 bg-yellow-500/10 text-yellow-400",
  info: "border-neon-blue/50 bg-neon-blue/10 text-neon-blue",
};

export function Toaster() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="fixed bottom-4 right-4 z-[300] flex flex-col gap-2">
      {toasts.map((toast) => {
        const Icon = icons[toast.type];
        return (
          <div
            key={toast.id}
            className={cn(
              "glass border rounded-xl p-4 max-w-sm animate-slide-in-right",
              colors[toast.type],
            )}
          >
            <div className="flex items-start gap-3">
              <Icon size={20} className="shrink-0" />
              <div className="flex-1">
                <p className="font-bold text-sm">{toast.title}</p>
                {toast.message && (
                  <p className="text-xs mt-0.5 opacity-80">{toast.message}</p>
                )}
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="shrink-0"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Hook to use toast
export const useToast = () => {
  const { addToast } = useToastStore();
  return {
    success: (title: string, message?: string) =>
      addToast({ type: "success", title, message }),
    error: (title: string, message?: string) =>
      addToast({ type: "error", title, message }),
    warning: (title: string, message?: string) =>
      addToast({ type: "warning", title, message }),
    info: (title: string, message?: string) =>
      addToast({ type: "info", title, message }),
  };
};
