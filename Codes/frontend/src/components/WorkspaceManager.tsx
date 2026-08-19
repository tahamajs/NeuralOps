import { useState } from "react";
import { useWindowStore } from "../store/windowStore";
import {
  Save,
  FolderOpen,
  LayoutTemplate,
  X,
  Sparkles,
  Grid,
  Columns,
  Maximize2,
  Trash2,
  Check,
  Bot,
  Zap,
  BarChart2,
  FileText,
} from "lucide-react";

export function WorkspaceManager() {
  const { openWindow, saveLayout, loadLayout, closeAllWindows } = useWindowStore();
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState("");
  const [feedback, setFeedback] = useState("");

  const savedLayouts = Object.keys(localStorage)
    .filter((k) => k.startsWith("workspace-"))
    .map((k) => k.replace("workspace-", ""));

  const handleSave = () => {
    if (!name.trim()) return;
    saveLayout(name.trim());
    setFeedback(`Layout "${name}" saved!`);
    setName("");
    setTimeout(() => setFeedback(""), 2500);
  };

  const handleLoad = (layoutName: string) => {
    loadLayout(layoutName);
    setFeedback(`Loaded "${layoutName}"!`);
    setTimeout(() => setFeedback(""), 2000);
  };

  const handleDelete = (layoutName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    localStorage.removeItem(`workspace-${layoutName}`);
    setFeedback(`Deleted "${layoutName}"`);
    setTimeout(() => setFeedback(""), 2000);
  };

  // Pre-configured Research Workspace Presets
  const applyPreset = (presetKey: string) => {
    closeAllWindows?.();
    setTimeout(() => {
      if (presetKey === "coscientist") {
        openWindow("terminal", { title: "Terminal", x: 50, y: 80, width: 620, height: 420 });
        openWindow("chat", { title: "AI Chat", x: 700, y: 80, width: 520, height: 600 });
        openWindow("notes", { title: "Notes", x: 50, y: 520, width: 620, height: 350 });
      } else if (presetKey === "gpu_market") {
        openWindow("marketplace", { title: "GPU Marketplace", x: 60, y: 80, width: 750, height: 550 });
        openWindow("gpumarket", { title: "GPU Trading Terminal", x: 830, y: 80, width: 500, height: 550 });
        openWindow("terminal", { title: "Terminal", x: 60, y: 650, width: 1270, height: 280 });
      } else if (presetKey === "svd_lab") {
        openWindow("analysis", { title: "Research Analysis", x: 50, y: 80, width: 780, height: 580 });
        openWindow("accelerator", { title: "Experiment Accelerator", x: 850, y: 80, width: 480, height: 580 });
        openWindow("terminal", { title: "Terminal", x: 50, y: 680, width: 1280, height: 260 });
      } else if (presetKey === "paper_suite") {
        openWindow("paper-editor", { title: "Paper Editor", x: 60, y: 80, width: 760, height: 620 });
        openWindow("citation-graph", { title: "Citation Graph", x: 840, y: 80, width: 480, height: 620 });
        openWindow("notes", { title: "Notes", x: 60, y: 720, width: 1260, height: 220 });
      }
      setFeedback("Preset workspace applied!");
      setTimeout(() => setFeedback(""), 2000);
    }, 100);
  };

  return (
    <div className="fixed top-20 right-4 z-[100] flex flex-col items-end gap-2">
      {/* Floating Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-3 py-2 rounded-xl bg-cyber-900/90 hover:bg-cyan-500/20 border border-white/10 hover:border-cyan-500/50 flex items-center gap-2 text-slate-300 hover:text-cyan-300 shadow-2xl backdrop-blur-xl transition-all group duration-300"
        title="Workspace Layouts"
      >
        {isOpen ? (
          <X size={16} />
        ) : (
          <LayoutTemplate
            size={16}
            className="text-cyan-400 group-hover:rotate-12 transition-transform"
          />
        )}
        <span className="text-xs font-bold font-display hidden sm:inline">Workspace Layouts</span>
      </button>

      {/* Expanded Layout Studio Panel */}
      {isOpen && (
        <div className="bg-[#0b1329]/95 backdrop-blur-2xl border border-cyan-500/30 rounded-2xl p-4 shadow-2xl flex flex-col gap-3 w-80 animate-fade-in text-white">
          <div className="flex items-center justify-between border-b border-white/10 pb-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-cyan-300 uppercase tracking-wider">
              <Sparkles size={14} className="text-cyan-400" />
              <span>Workspace Studio</span>
            </div>
            <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white text-xs">
              ✕
            </button>
          </div>

          {/* Curated Pro Layout Presets */}
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">
              Pro Research Presets
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={() => applyPreset("coscientist")}
                className="glass-hover p-2 rounded-xl border border-white/5 hover:border-cyan-500/30 text-left transition-all group"
              >
                <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-300 mb-0.5">
                  <Bot size={13} className="text-cyan-400" /> Co-Scientist
                </div>
                <div className="text-[9px] text-slate-400">Agent + Shell + Notes</div>
              </button>

              <button
                onClick={() => applyPreset("gpu_market")}
                className="glass-hover p-2 rounded-xl border border-white/5 hover:border-purple-500/30 text-left transition-all group"
              >
                <div className="flex items-center gap-1.5 text-xs font-semibold text-purple-300 mb-0.5">
                  <Zap size={13} className="text-purple-400" /> GPU Terminal
                </div>
                <div className="text-[9px] text-slate-400">Market + Trading + SSH</div>
              </button>

              <button
                onClick={() => applyPreset("svd_lab")}
                className="glass-hover p-2 rounded-xl border border-white/5 hover:border-emerald-500/30 text-left transition-all group"
              >
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-300 mb-0.5">
                  <BarChart2 size={13} className="text-emerald-400" /> SVD HPO Lab
                </div>
                <div className="text-[9px] text-slate-400">Analysis + Accelerator</div>
              </button>

              <button
                onClick={() => applyPreset("paper_suite")}
                className="glass-hover p-2 rounded-xl border border-white/5 hover:border-amber-500/30 text-left transition-all group"
              >
                <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-300 mb-0.5">
                  <FileText size={13} className="text-amber-400" /> Paper Suite
                </div>
                <div className="text-[9px] text-slate-400">LaTeX + Citations</div>
              </button>
            </div>
          </div>

          {/* Custom Saved Layouts */}
          <div className="pt-2 border-t border-white/10">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">
              Save Current Arrangement
            </div>
            <div className="flex gap-1.5">
              <input
                type="text"
                placeholder="Name your layout..."
                className="flex-1 bg-black/50 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500/50 transition-colors"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSave()}
              />
              <button
                onClick={handleSave}
                title="Save Layout"
                className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold rounded-xl text-xs flex items-center gap-1 shadow-md shadow-cyan-500/20"
              >
                <Save size={13} />
              </button>
            </div>

            {savedLayouts.length > 0 && (
              <div className="flex flex-col gap-1 mt-2 max-h-36 overflow-y-auto scrollbar-thin">
                {savedLayouts.map((layout) => (
                  <div
                    key={layout}
                    className="flex items-center justify-between group px-2.5 py-1.5 hover:bg-white/5 rounded-lg text-xs text-slate-200 cursor-pointer border border-transparent hover:border-white/10"
                    onClick={() => handleLoad(layout)}
                  >
                    <span className="truncate">{layout}</span>
                    <div className="flex items-center gap-1.5">
                      <FolderOpen
                        size={12}
                        className="opacity-60 group-hover:opacity-100 text-cyan-400"
                      />
                      <button
                        onClick={(e) => handleDelete(layout, e)}
                        className="text-slate-500 hover:text-red-400 p-0.5"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {feedback && (
            <div className="text-[11px] text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 p-1.5 rounded-lg font-medium text-center animate-fade-in flex items-center justify-center gap-1">
              <Check size={12} />
              {feedback}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
