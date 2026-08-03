import { useState } from "react";
import { useWindowStore } from "../store/windowStore";
import { Save, FolderOpen, LayoutTemplate, X } from "lucide-react";

export function WorkspaceManager() {
  const { saveLayout, loadLayout } = useWindowStore();
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState("");
  const [feedback, setFeedback] = useState("");

  const savedLayouts = Object.keys(localStorage)
    .filter((k) => k.startsWith("workspace-"))
    .map((k) => k.replace("workspace-", ""));

  const handleSave = () => {
    if (!name.trim()) return;
    saveLayout(name);
    setFeedback("Saved!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const handleLoad = (layoutName: string) => {
    loadLayout(layoutName);
    setName(layoutName);
    setFeedback("Loaded!");
    setTimeout(() => setFeedback(""), 2000);
  };

  return (
    <div className="fixed top-20 right-4 z-[100] flex flex-col items-end gap-2">
      {/* Toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-10 h-10 rounded-full bg-cyber-900/90 hover:bg-neon-blue/20 border border-white/10 hover:border-neon-blue/50 flex items-center justify-center text-slate-300 hover:text-neon-blue shadow-lg backdrop-blur-md transition-all group duration-300"
        title="Workspace Layouts"
      >
        {isOpen ? (
          <X size={18} />
        ) : (
          <LayoutTemplate
            size={18}
            className="group-hover:rotate-12 transition-transform"
          />
        )}
      </button>

      {/* Expanded panel */}
      {isOpen && (
        <div className="bg-cyber-900/95 backdrop-blur-md border border-white/10 rounded-xl p-3 shadow-2xl flex flex-col gap-2 w-64 animate-fade-in">
          <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider px-1">
            <span>Workspace Layouts</span>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="New layout name..."
              className="flex-1 bg-black/50 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-blue transition-colors"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
            />
            <button
              onClick={handleSave}
              title="Save Layout"
              className="p-1.5 bg-neon-blue/20 hover:bg-neon-blue/40 text-neon-blue rounded transition-colors"
            >
              <Save size={14} />
            </button>
          </div>
          {savedLayouts.length > 0 && (
            <div className="flex flex-col gap-1 mt-1 max-h-32 overflow-y-auto scrollbar-thin">
              {savedLayouts.map((layout) => (
                <div
                  key={layout}
                  className="flex items-center justify-between group px-2 py-1.5 hover:bg-white/5 rounded text-xs text-slate-300 cursor-pointer"
                  onClick={() => handleLoad(layout)}
                >
                  <span>{layout}</span>
                  <FolderOpen
                    size={12}
                    className="opacity-0 group-hover:opacity-100 text-neon-blue transition-opacity"
                  />
                </div>
              ))}
            </div>
          )}
          {feedback && (
            <div className="text-[10px] text-green-400 font-medium text-center animate-pulse">
              {feedback}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
