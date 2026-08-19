import { useState, useMemo } from "react";
import { useWindowStore, WindowItem } from "../store/windowStore";
import { useNavigate } from "react-router-dom";
import { AVAILABLE_WINDOWS } from "./WindowLauncher";
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
  Layers,
  Cpu,
  ShieldCheck,
  Search,
  Download,
  Upload,
  Plus,
  Minimize2,
  RotateCcw,
  Sliders,
  ExternalLink,
  HelpCircle,
  Copy,
} from "lucide-react";

export function WorkspaceManager() {
  const {
    windows,
    openWindow,
    saveLayout,
    loadLayout,
    closeAllWindows,
    updateWindow,
  } = useWindowStore();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<"presets" | "custom" | "tools" | "tile">("presets");

  const savedLayouts = useMemo(() => {
    return Object.keys(localStorage)
      .filter((k) => k.startsWith("workspace-"))
      .map((k) => k.replace("workspace-", ""));
  }, [feedback, isOpen]);

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

  const handleExportJSON = () => {
    const layout = windows.map(({ type, title, icon, width, height, x, y }) => ({
      type,
      title,
      icon,
      width,
      height,
      x,
      y,
    }));
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(layout, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `neuralops-workspace-${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    setFeedback("Workspace exported to JSON!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const handleImportJSON = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (Array.isArray(parsed)) {
          closeAllWindows?.();
          setTimeout(() => {
            parsed.forEach((w: any) => {
              openWindow(w.type, {
                title: w.title,
                icon: w.icon,
                x: w.x,
                y: w.y,
                width: w.width,
                height: w.height,
              });
            });
            setFeedback("Imported workspace successfully!");
            setTimeout(() => setFeedback(""), 2500);
          }, 100);
        }
      } catch (err) {
        setFeedback("Failed to parse JSON file.");
        setTimeout(() => setFeedback(""), 2500);
      }
    };
    reader.readAsText(file);
  };

  // Smart Auto-Tiling Algorithms
  const tile2x2 = () => {
    if (windows.length === 0) return;
    const w = Math.floor((window.innerWidth - 120) / 2);
    const h = Math.floor((window.innerHeight - 160) / 2);
    const positions = [
      { x: 50, y: 80 },
      { x: 60 + w, y: 80 },
      { x: 50, y: 90 + h },
      { x: 60 + w, y: 90 + h },
    ];
    windows.forEach((win, idx) => {
      const pos = positions[idx % 4];
      updateWindow(win.id, {
        x: pos.x,
        y: pos.y,
        width: w,
        height: h,
        minimized: false,
        maximized: false,
      });
    });
    setFeedback("Tiled 2x2 Grid!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const tileSplitHorizontal = () => {
    if (windows.length === 0) return;
    const count = Math.min(windows.length, 3);
    const w = Math.floor((window.innerWidth - 120) / count);
    const h = window.innerHeight - 160;
    windows.slice(0, count).forEach((win, idx) => {
      updateWindow(win.id, {
        x: 50 + idx * (w + 10),
        y: 80,
        width: w,
        height: h,
        minimized: false,
        maximized: false,
      });
    });
    setFeedback("Split Horizontally!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const tileSplitVertical = () => {
    if (windows.length === 0) return;
    const count = Math.min(windows.length, 3);
    const w = window.innerWidth - 120;
    const h = Math.floor((window.innerHeight - 160) / count);
    windows.slice(0, count).forEach((win, idx) => {
      updateWindow(win.id, {
        x: 50,
        y: 80 + idx * (h + 10),
        width: w,
        height: h,
        minimized: false,
        maximized: false,
      });
    });
    setFeedback("Split Vertically!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const cascadeWindows = () => {
    if (windows.length === 0) return;
    const defaultW = Math.min(800, window.innerWidth - 200);
    const defaultH = Math.min(560, window.innerHeight - 200);
    windows.forEach((win, idx) => {
      updateWindow(win.id, {
        x: 60 + (idx * 35) % 300,
        y: 80 + (idx * 35) % 250,
        width: defaultW,
        height: defaultH,
        minimized: false,
        maximized: false,
      });
    });
    setFeedback("Cascaded Windows!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const minimizeAll = () => {
    windows.forEach((w) => updateWindow(w.id, { minimized: true }));
    setFeedback("Minimized All Windows!");
    setTimeout(() => setFeedback(""), 2000);
  };

  const restoreAll = () => {
    windows.forEach((w) => updateWindow(w.id, { minimized: false }));
    setFeedback("Restored All Windows!");
    setTimeout(() => setFeedback(""), 2000);
  };

  // 12+ Curated Research & Engineering Presets
  const presets = [
    {
      id: "coscientist",
      title: "Co-Scientist Workspace",
      desc: "Agent Chat + Interactive Shell + Notes + Citation Graph",
      cat: "ai",
      icon: <Bot size={15} className="text-cyan-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("terminal", { title: "Terminal Shell", x: 50, y: 80, width: 620, height: 420 });
          openWindow("chat", { title: "AI Co-Scientist Chat", x: 700, y: 80, width: 540, height: 600 });
          openWindow("notes", { title: "Research Notes", x: 50, y: 520, width: 620, height: 350 });
          openWindow("citation-graph", { title: "Citation Graph Viewer", x: 700, y: 520, width: 540, height: 350 });
        }, 100);
      },
    },
    {
      id: "gpu_market",
      title: "GPU Supercompute Terminal",
      desc: "Marketplace + Stock Trading + Multi-Node SSH Shell",
      cat: "gpu",
      icon: <Zap size={15} className="text-purple-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("marketplace", { title: "GPU Compute Marketplace", x: 60, y: 80, width: 750, height: 550 });
          openWindow("gpumarket", { title: "GPU Trading Terminal", x: 830, y: 80, width: 520, height: 550 });
          openWindow("terminal", { title: "Cluster SSH Console", x: 60, y: 650, width: 1290, height: 280 });
        }, 100);
      },
    },
    {
      id: "svd_lab",
      title: "SVD Compression & HPO Lab",
      desc: "Research Analysis + Experiment Accelerator + Sweeps",
      cat: "svd",
      icon: <BarChart2 size={15} className="text-emerald-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("analysis", { title: "SVD Energy Analysis", x: 50, y: 80, width: 780, height: 580 });
          openWindow("accelerator", { title: "Experiment Accelerator", x: 850, y: 80, width: 500, height: 580 });
          openWindow("hpo", { title: "Hyperparameter Sweeps", x: 50, y: 680, width: 1300, height: 280 });
        }, 100);
      },
    },
    {
      id: "paper_suite",
      title: "LaTeX Authoring & Publishing",
      desc: "LaTeX Monograph Editor + Citation Graph + Literature Map",
      cat: "paper",
      icon: <FileText size={15} className="text-amber-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("paper-editor", { title: "LaTeX Monograph Editor", x: 60, y: 80, width: 780, height: 620 });
          openWindow("citation-graph", { title: "Citation Graph", x: 860, y: 80, width: 480, height: 620 });
          openWindow("literature-map", { title: "ArXiv Literature Map", x: 60, y: 720, width: 1280, height: 240 });
        }, 100);
      },
    },
    {
      id: "llm_arena",
      title: "LLM Battle & Evaluation Studio",
      desc: "LLM Playground + Evaluation Suite + Tool Catalog",
      cat: "ai",
      icon: <Sparkles size={15} className="text-rose-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("llm-playground", { title: "LLM Playground", x: 60, y: 80, width: 700, height: 560 });
          openWindow("llm-eval", { title: "Evaluation Suite", x: 780, y: 80, width: 560, height: 560 });
          openWindow("prompt-library", { title: "Prompt Catalog", x: 60, y: 660, width: 1280, height: 260 });
        }, 100);
      },
    },
    {
      id: "cluster_ops",
      title: "Distributed Cluster & MLOps",
      desc: "Cluster Overview + Drift Monitor + Health Telemetry",
      cat: "mlops",
      icon: <Cpu size={15} className="text-blue-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("cluster", { title: "Cluster Overview", x: 60, y: 80, width: 760, height: 550 });
          openWindow("drift", { title: "MLOps Drift Monitor", x: 840, y: 80, width: 500, height: 550 });
          openWindow("health", { title: "Health Telemetry Dashboard", x: 60, y: 650, width: 1280, height: 270 });
        }, 100);
      },
    },
    {
      id: "model_3d",
      title: "3D Model & Architecture Viewer",
      desc: "3D Transformer Explorer + Matrix Visualizer",
      cat: "svd",
      icon: <Layers size={15} className="text-indigo-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("model-viewer", { title: "3D Model Architecture", x: 60, y: 80, width: 800, height: 580 });
          openWindow("analysis", { title: "Spectral Decomposition", x: 880, y: 80, width: 460, height: 580 });
        }, 100);
      },
    },
    {
      id: "tournament_arena",
      title: "Idea Tournament & Hypothesis Arena",
      desc: "Tournament Brackets + Hypothesis Browser + Deep Research",
      cat: "ai",
      icon: <Sparkles size={15} className="text-yellow-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("tournament", { title: "Idea Tournament", x: 60, y: 80, width: 720, height: 560 });
          openWindow("hypotheses", { title: "Hypothesis Browser", x: 800, y: 80, width: 540, height: 560 });
          openWindow("research", { title: "Deep Research Agent", x: 60, y: 660, width: 1280, height: 260 });
        }, 100);
      },
    },
  ];

  const filteredPresets =
    activeCategory === "all"
      ? presets
      : presets.filter((p) => p.cat === activeCategory);

  const filteredTools = useMemo(() => {
    if (!searchQuery.trim()) return AVAILABLE_WINDOWS.slice(0, 16);
    return AVAILABLE_WINDOWS.filter((w) =>
      w.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      w.type.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery]);

  return (
    <div className="fixed top-20 right-4 z-[100] flex flex-col items-end gap-2">
      {/* Floating Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-3.5 py-2 rounded-2xl bg-[#0b1329]/95 hover:bg-cyan-500/20 border border-cyan-500/30 hover:border-cyan-500/60 flex items-center gap-2.5 text-slate-200 hover:text-cyan-300 shadow-2xl backdrop-blur-2xl transition-all group duration-300 cursor-pointer"
        title="Open Workspace Studio"
      >
        {isOpen ? (
          <X size={16} className="text-cyan-400" />
        ) : (
          <LayoutTemplate
            size={16}
            className="text-cyan-400 group-hover:rotate-12 transition-transform"
          />
        )}
        <span className="text-xs font-bold font-display tracking-wide flex items-center gap-1.5">
          Workspace Studio
          {windows.length > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-cyan-500 text-black text-[10px] font-black">
              {windows.length}
            </span>
          )}
        </span>
      </button>

      {/* Expanded Studio Control Center */}
      {isOpen && (
        <div className="bg-[#090d1f]/95 backdrop-blur-2xl border border-cyan-500/30 rounded-3xl p-5 shadow-2xl flex flex-col gap-4 w-96 animate-fade-in text-white max-h-[85vh] overflow-y-auto scrollbar-thin">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center text-white shadow-md">
                <Sparkles size={15} />
              </div>
              <div>
                <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                  Workspace Studio Pro
                </h3>
                <p className="text-[10px] text-slate-400">
                  {windows.length} Floating Windows Active
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={() => navigate("/workspace")}
                title="Full Page Studio"
                className="p-1 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-white/5 transition-all cursor-pointer"
              >
                <ExternalLink size={14} />
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 text-xs cursor-pointer"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Quick Auto-Tile Toolbar */}
          <div className="glass p-2.5 rounded-2xl border border-white/10 space-y-2">
            <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              <span>Smart Window Layouts</span>
              <span className="text-cyan-400 font-mono">Auto-Dock</span>
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              <button
                onClick={tile2x2}
                title="Tile 2x2 Grid"
                className="p-2 rounded-xl bg-white/5 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-300 text-[10px] font-bold flex flex-col items-center gap-1 transition-all cursor-pointer border border-transparent hover:border-cyan-500/30"
              >
                <Grid size={13} />
                <span>2x2 Grid</span>
              </button>

              <button
                onClick={tileSplitHorizontal}
                title="Split Columns (Left/Right)"
                className="p-2 rounded-xl bg-white/5 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-300 text-[10px] font-bold flex flex-col items-center gap-1 transition-all cursor-pointer border border-transparent hover:border-cyan-500/30"
              >
                <Columns size={13} />
                <span>Columns</span>
              </button>

              <button
                onClick={tileSplitVertical}
                title="Split Rows (Top/Bottom)"
                className="p-2 rounded-xl bg-white/5 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-300 text-[10px] font-bold flex flex-col items-center gap-1 transition-all cursor-pointer border border-transparent hover:border-cyan-500/30"
              >
                <Layers size={13} />
                <span>Rows</span>
              </button>

              <button
                onClick={cascadeWindows}
                title="Cascade Windows"
                className="p-2 rounded-xl bg-white/5 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-300 text-[10px] font-bold flex flex-col items-center gap-1 transition-all cursor-pointer border border-transparent hover:border-cyan-500/30"
              >
                <Maximize2 size={13} />
                <span>Cascade</span>
              </button>
            </div>

            {/* Quick Bulk Actions */}
            <div className="flex items-center justify-between pt-1 border-t border-white/5 text-[10px] text-slate-400">
              <button
                onClick={minimizeAll}
                className="hover:text-white flex items-center gap-1 cursor-pointer"
              >
                <Minimize2 size={11} /> Minimize All
              </button>
              <button
                onClick={restoreAll}
                className="hover:text-white flex items-center gap-1 cursor-pointer"
              >
                <RotateCcw size={11} /> Restore All
              </button>
              <button
                onClick={() => {
                  closeAllWindows?.();
                  setFeedback("Closed all windows!");
                  setTimeout(() => setFeedback(""), 2000);
                }}
                className="text-red-400 hover:text-red-300 flex items-center gap-1 cursor-pointer"
              >
                <Trash2 size={11} /> Close All
              </button>
            </div>
          </div>

          {/* Navigation View Switcher */}
          <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/5 text-xs font-semibold">
            <button
              onClick={() => setViewMode("presets")}
              className={`flex-1 py-1 rounded-lg text-center transition-all cursor-pointer ${
                viewMode === "presets"
                  ? "bg-cyan-500 text-black font-bold shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Presets
            </button>
            <button
              onClick={() => setViewMode("tools")}
              className={`flex-1 py-1 rounded-lg text-center transition-all cursor-pointer ${
                viewMode === "tools"
                  ? "bg-cyan-500 text-black font-bold shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Tools (106)
            </button>
            <button
              onClick={() => setViewMode("custom")}
              className={`flex-1 py-1 rounded-lg text-center transition-all cursor-pointer ${
                viewMode === "custom"
                  ? "bg-cyan-500 text-black font-bold shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Saved ({savedLayouts.length})
            </button>
          </div>

          {/* Mode 1: Presets */}
          {viewMode === "presets" && (
            <div className="space-y-3">
              {/* Category Pills */}
              <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none text-[10px]">
                {["all", "ai", "gpu", "svd", "paper", "mlops"].map((c) => (
                  <button
                    key={c}
                    onClick={() => setActiveCategory(c)}
                    className={`px-2.5 py-1 rounded-full uppercase font-bold transition-all cursor-pointer ${
                      activeCategory === c
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                        : "text-slate-400 hover:text-white hover:bg-white/5"
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>

              {/* Presets Grid */}
              <div className="grid grid-cols-1 gap-2">
                {filteredPresets.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => {
                      p.action();
                      setFeedback(`Applied "${p.title}"!`);
                      setTimeout(() => setFeedback(""), 2000);
                    }}
                    className="glass-hover p-2.5 rounded-2xl border border-white/5 hover:border-cyan-500/40 text-left transition-all group cursor-pointer bg-white/[0.01]"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2 text-xs font-bold text-cyan-300 group-hover:text-cyan-200">
                        {p.icon}
                        <span>{p.title}</span>
                      </div>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-slate-400 font-mono">
                        {p.cat.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-400 group-hover:text-slate-300 line-clamp-1">
                      {p.desc}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Mode 2: Tools Launcher Search */}
          {viewMode === "tools" && (
            <div className="space-y-2.5">
              <div className="relative">
                <Search
                  size={13}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                />
                <input
                  type="text"
                  placeholder="Search among 106 tools..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-black/50 border border-white/10 rounded-xl pl-8 pr-3 py-1.5 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <div className="grid grid-cols-2 gap-1.5 max-h-56 overflow-y-auto scrollbar-thin">
                {filteredTools.map((tool) => (
                  <button
                    key={tool.type}
                    onClick={() => {
                      openWindow(tool.type, { title: tool.title });
                      setFeedback(`Opened ${tool.title}!`);
                      setTimeout(() => setFeedback(""), 1500);
                    }}
                    className="glass-hover p-2 rounded-xl border border-white/5 hover:border-cyan-500/30 text-left transition-all flex items-center gap-2 group cursor-pointer"
                  >
                    <span className="text-sm">{tool.icon}</span>
                    <span className="text-xs text-slate-300 group-hover:text-white truncate font-medium">
                      {tool.title}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Mode 3: Custom Layouts & Persistence */}
          {viewMode === "custom" && (
            <div className="space-y-3">
              <div className="flex gap-1.5">
                <input
                  type="text"
                  placeholder="Layout name..."
                  className="flex-1 bg-black/50 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500/50"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSave()}
                />
                <button
                  onClick={handleSave}
                  title="Save Current Arrangement"
                  className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold rounded-xl text-xs flex items-center gap-1 shadow-md shadow-cyan-500/20 cursor-pointer"
                >
                  <Save size={13} /> Save
                </button>
              </div>

              {/* Export / Import Toolbar */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportJSON}
                  className="flex-1 glass-hover border border-white/10 text-slate-300 px-2 py-1.5 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 cursor-pointer"
                >
                  <Download size={11} className="text-cyan-400" /> Export JSON
                </button>

                <label className="flex-1 glass-hover border border-white/10 text-slate-300 px-2 py-1.5 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 cursor-pointer">
                  <Upload size={11} className="text-purple-400" /> Import JSON
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleImportJSON}
                  />
                </label>
              </div>

              {/* Saved list */}
              {savedLayouts.length === 0 ? (
                <p className="text-[11px] text-slate-500 text-center py-4">
                  No saved custom layouts yet. Arrange your windows and click Save!
                </p>
              ) : (
                <div className="flex flex-col gap-1 max-h-48 overflow-y-auto scrollbar-thin">
                  {savedLayouts.map((layout) => (
                    <div
                      key={layout}
                      className="flex items-center justify-between group px-3 py-2 hover:bg-white/5 rounded-xl text-xs text-slate-200 cursor-pointer border border-transparent hover:border-white/10 transition-all"
                      onClick={() => handleLoad(layout)}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <FolderOpen size={13} className="text-cyan-400 flex-shrink-0" />
                        <span className="truncate font-medium">{layout}</span>
                      </div>
                      <button
                        onClick={(e) => handleDelete(layout, e)}
                        className="text-slate-500 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-all cursor-pointer"
                        title="Delete layout"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Feedback Toast */}
          {feedback && (
            <div className="text-[11px] text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 p-2 rounded-xl font-medium text-center animate-fade-in flex items-center justify-center gap-1.5">
              <Check size={13} />
              {feedback}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

