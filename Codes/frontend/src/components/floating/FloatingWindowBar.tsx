import { useFloatingWindowStore, WINDOW_TYPES } from "./FloatingWindowManager";

const QUICK_LAUNCH = [
  {
    id: WINDOW_TYPES.TERMINAL,
    title: "Terminal",
    icon: ">_",
    color: "text-green-400",
  },
  {
    id: WINDOW_TYPES.PAPER_EDITOR,
    title: "Paper",
    icon: "📄",
    color: "text-purple-400",
  },
  {
    id: WINDOW_TYPES.NOTES_EDITOR,
    title: "Notes",
    icon: "📝",
    color: "text-yellow-400",
  },
  {
    id: WINDOW_TYPES.SYSTEM_MONITOR,
    title: "Monitor",
    icon: "⬡",
    color: "text-blue-400",
  },
];

export default function FloatingWindowBar() {
  const { addWindow, windows, toggleMinimize } = useFloatingWindowStore();

  const handleClick = (item: (typeof QUICK_LAUNCH)[number]) => {
    const existing = windows.find((w) => w.id === item.id);
    if (existing) {
      if (existing.minimized) toggleMinimize(item.id);
      else toggleMinimize(item.id); // or just focus: we can bring to front
    } else {
      addWindow({
        id: item.id,
        title: item.title,
        icon: item.icon,
        content: item.id,
        width: 700,
        height: 500,
      });
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-[200] flex flex-col gap-2">
      {QUICK_LAUNCH.map((item) => {
        const isOpen = windows.some((w) => w.id === item.id && !w.minimized);
        return (
          <button
            key={item.id}
            onClick={() => handleClick(item)}
            title={item.title}
            className={`w-10 h-10 rounded-xl flex items-center justify-center text-sm transition-all
              ${
                isOpen
                  ? "bg-neon-blue/20 border border-neon-blue/40 shadow-lg shadow-neon-blue/20"
                  : "glass hover:bg-white/10 text-slate-400 hover:text-white"
              }`}
          >
            <span className={item.color}>{item.icon}</span>
          </button>
        );
      })}
    </div>
  );
}
