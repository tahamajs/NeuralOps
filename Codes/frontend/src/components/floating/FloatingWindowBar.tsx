import { useWindowStore } from "../../store/windowStore";

const QUICK_LAUNCH = [
  {
    id: "terminal",
    title: "Terminal",
    icon: ">_",
    color: "text-green-400",
  },
  {
    id: "paper-editor",
    title: "Paper",
    icon: "📄",
    color: "text-purple-400",
  },
  {
    id: "notes",
    title: "Notes",
    icon: "📝",
    color: "text-yellow-400",
  },
  {
    id: "system",
    title: "Monitor",
    icon: "⬡",
    color: "text-blue-400",
  },
];

export default function FloatingWindowBar() {
  const { addWindow, windows, toggleMinimize } = useWindowStore();

  const handleClick = (item: (typeof QUICK_LAUNCH)[number]) => {
    const existing = windows.find((w) => w.type === item.id);
    if (existing) {
      if (existing.minimized) toggleMinimize(existing.id);
      else toggleMinimize(existing.id);
    } else {
      addWindow(item.id, item.title, item.icon);
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
