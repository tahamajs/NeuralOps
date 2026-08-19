import { useWindowStore } from "../../store/windowStore";

const QUICK_LAUNCH = [
  { id: "terminal", title: "Terminal", icon: ">_", color: "text-green-400" },
  { id: "chat", title: "AI Chat", icon: "💬", color: "text-cyan-400" },
  { id: "paper-editor", title: "Paper Editor", icon: "📄", color: "text-purple-400" },
  { id: "notes", title: "Notes", icon: "📝", color: "text-yellow-400" },
];

export default function FloatingWindowBar() {
  const { addWindow, windows, toggleMinimize, removeWindow } = useWindowStore();

  const handleClick = (item: (typeof QUICK_LAUNCH)[number]) => {
    // FIX: was checking w.id === item.id (window id vs type) — must compare w.type
    const existing = windows.find((w) => w.type === item.id);
    if (existing) {
      toggleMinimize(existing.id);
    } else {
      addWindow(item.id, item.title, item.icon);
    }
  };

  const openCount = windows.filter((w) => !w.minimized).length;

  return (
    <div className="fixed bottom-6 right-6 z-[200] flex flex-col items-end gap-2">
      {/* Active windows badge */}
      {openCount > 0 && (
        <div className="flex items-center gap-1.5 bg-[#0c1222]/90 border border-white/10 rounded-full px-3 py-1 backdrop-blur-xl shadow-xl">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[10px] font-mono text-slate-400">
            {openCount} window{openCount !== 1 ? "s" : ""} open
          </span>
        </div>
      )}

      {/* Quick launch buttons */}
      <div className="flex flex-col gap-2">
        {QUICK_LAUNCH.map((item) => {
          const existing = windows.find((w) => w.type === item.id);
          const isOpen = existing && !existing.minimized;
          const isMinimized = existing && existing.minimized;

          return (
            <div key={item.id} className="relative group flex items-center gap-2 justify-end">
              {/* Tooltip label */}
              <span className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none absolute right-14 bg-[#0c1222]/95 border border-white/10 text-[10px] font-medium text-slate-200 px-2 py-1 rounded-lg whitespace-nowrap shadow-xl">
                {item.title}
                {isOpen && <span className="ml-1 text-emerald-400">● open</span>}
                {isMinimized && <span className="ml-1 text-amber-400">⊡ minimized</span>}
              </span>

              <div className="flex items-center gap-1">
                {/* Close button (only if window exists) */}
                {existing && (
                  <button
                    onClick={() => removeWindow(existing.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity w-5 h-5 rounded-lg flex items-center justify-center text-slate-500 hover:text-red-400 hover:bg-red-500/10 text-[10px]"
                    title="Close window"
                  >
                    ✕
                  </button>
                )}

                {/* Main button */}
                <button
                  onClick={() => handleClick(item)}
                  title={item.title}
                  className={`relative w-11 h-11 rounded-2xl flex items-center justify-center text-base transition-all shadow-xl ${
                    isOpen
                      ? "bg-cyan-500/20 border border-cyan-500/50 shadow-cyan-500/20"
                      : isMinimized
                      ? "bg-amber-500/10 border border-amber-500/30"
                      : "glass border border-white/10 hover:bg-white/10 hover:border-white/20"
                  }`}
                >
                  <span className={isOpen ? "text-white" : item.color}>{item.icon}</span>
                  {isMinimized && (
                    <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-amber-400 border-2 border-[#080d19]" />
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
