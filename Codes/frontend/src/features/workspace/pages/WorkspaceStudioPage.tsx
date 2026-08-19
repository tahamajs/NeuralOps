// src/features/workspace/pages/WorkspaceStudioPage.tsx
import React, { useState, useMemo } from "react";
import { useWindowStore, WindowItem } from "../../../store/windowStore";
import { AVAILABLE_WINDOWS } from "../../../components/WindowLauncher";
import {
  LayoutTemplate,
  Grid,
  Columns,
  Maximize2,
  Minimize2,
  RotateCcw,
  Trash2,
  Save,
  FolderOpen,
  Download,
  Upload,
  Sparkles,
  Bot,
  Zap,
  BarChart2,
  FileText,
  Layers,
  Cpu,
  ShieldCheck,
  Search,
  Check,
  ExternalLink,
  Plus,
  Play,
  Terminal,
  Activity,
  Boxes,
  Compass,
  Database,
  ArrowRight,
  Monitor,
  Flame,
} from "lucide-react";

export default function WorkspaceStudioPage() {
  const {
    windows,
    openWindow,
    saveLayout,
    loadLayout,
    closeAllWindows,
    updateWindow,
    bringToFront,
    removeWindow,
  } = useWindowStore();

  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [layoutName, setLayoutName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [selectedPresetCat, setSelectedPresetCat] = useState<string>("all");

  const savedLayouts = useMemo(() => {
    return Object.keys(localStorage)
      .filter((k) => k.startsWith("workspace-"))
      .map((k) => k.replace("workspace-", ""));
  }, [feedback]);

  const handleSave = () => {
    if (!layoutName.trim()) return;
    saveLayout(layoutName.trim());
    setFeedback(`Workspace layout "${layoutName}" saved!`);
    setLayoutName("");
    setTimeout(() => setFeedback(""), 2500);
  };

  const handleLoad = (name: string) => {
    loadLayout(name);
    setFeedback(`Loaded workspace "${name}"!`);
    setTimeout(() => setFeedback(""), 2000);
  };

  const handleDelete = (name: string) => {
    localStorage.removeItem(`workspace-${name}`);
    setFeedback(`Deleted workspace "${name}"`);
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
    const a = document.createElement("a");
    a.setAttribute("href", dataStr);
    a.setAttribute("download", `neuralops-workspace-${Date.now()}.json`);
    document.body.appendChild(a);
    a.click();
    a.remove();
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

  // Smart Auto-Tiling
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

  // 12 Curated Presets
  const presets = [
    {
      id: "coscientist",
      title: "Autonomous Co-Scientist Suite",
      desc: "Interactive terminal shell, hypothesis generator, research notes editor, and live citation graph map.",
      cat: "ai",
      tools: ["Terminal", "AI Co-Scientist Chat", "Notes", "Citation Graph"],
      icon: <Bot size={20} className="text-cyan-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("terminal", { title: "Terminal Shell", x: 50, y: 80, width: 620, height: 420 });
          openWindow("chat", { title: "AI Co-Scientist Chat", x: 700, y: 80, width: 540, height: 600 });
          openWindow("notes", { title: "Research Notes", x: 50, y: 520, width: 620, height: 350 });
          openWindow("citation-graph", { title: "Citation Graph", x: 700, y: 520, width: 540, height: 350 });
        }, 100);
      },
    },
    {
      id: "gpu_market",
      title: "GPU Supercompute Trading Terminal",
      desc: "Live GPU compute index, high-frequency rental marketplace, and multi-node SSH cluster console.",
      cat: "gpu",
      tools: ["GPU Marketplace", "Stock Trading", "Cluster SSH Shell"],
      icon: <Zap size={20} className="text-purple-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("marketplace", { title: "GPU Marketplace", x: 60, y: 80, width: 750, height: 550 });
          openWindow("gpumarket", { title: "GPU Stock Market", x: 830, y: 80, width: 520, height: 550 });
          openWindow("terminal", { title: "Cluster SSH Console", x: 60, y: 650, width: 1290, height: 280 });
        }, 100);
      },
    },
    {
      id: "svd_lab",
      title: "SVD Compression & HPO Lab",
      desc: "Layer-wise singular value entropy analyzer, training accelerator, and hyperparameter sweep matrix.",
      cat: "svd",
      tools: ["Research Analysis", "Accelerator", "HPO Sweeps"],
      icon: <BarChart2 size={20} className="text-emerald-400" />,
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
      title: "Academic Paper & Monograph Publisher",
      desc: "LaTeX typeset compiler, ArXiv literature explorer, citation network viewer, and monograph downloader.",
      cat: "paper",
      tools: ["LaTeX Editor", "Citation Graph", "Literature Map"],
      icon: <FileText size={20} className="text-amber-400" />,
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
      desc: "Side-by-side prompt execution arena, benchmark evaluation runner, and prompt templates catalog.",
      cat: "ai",
      tools: ["LLM Playground", "Evaluation Suite", "Prompt Library"],
      icon: <Sparkles size={20} className="text-rose-400" />,
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
      title: "Distributed Cluster & MLOps Operations",
      desc: "GPU hardware cluster node topology, data drift monitor, and real-time health telemetry.",
      cat: "mlops",
      tools: ["Cluster Overview", "Drift Monitor", "Health Telemetry"],
      icon: <Cpu size={20} className="text-blue-400" />,
      action: () => {
        closeAllWindows?.();
        setTimeout(() => {
          openWindow("cluster", { title: "Cluster Overview", x: 60, y: 80, width: 760, height: 550 });
          openWindow("drift", { title: "MLOps Drift Monitor", x: 840, y: 80, width: 500, height: 550 });
          openWindow("health", { title: "Health Telemetry Dashboard", x: 60, y: 650, width: 1280, height: 270 });
        }, 100);
      },
    },
  ];

  const filteredPresets =
    selectedPresetCat === "all"
      ? presets
      : presets.filter((p) => p.cat === selectedPresetCat);

  const categories = [
    { id: "all", label: "All Tools" },
    { id: "core", label: "Core & Shell" },
    { id: "market", label: "Marketplace & GPU" },
    { id: "experiments", label: "Experiments & HPO" },
    { id: "ai", label: "AI Co-Scientist" },
    { id: "literature", label: "Literature" },
    { id: "llm", label: "LLM Hub" },
    { id: "mlops", label: "MLOps & Security" },
  ];

  const filteredTools = useMemo(() => {
    return AVAILABLE_WINDOWS.filter((w) => {
      const matchSearch =
        !searchQuery.trim() ||
        w.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        w.type.toLowerCase().includes(searchQuery.toLowerCase());
      return matchSearch;
    });
  }, [searchQuery, activeCategory]);

  return (
    <div className="space-y-8 animate-fade-in max-w-7xl mx-auto p-4 select-text">
      {/* ────── Studio Hero Header ────── */}
      <div className="glass rounded-3xl border border-cyan-500/30 p-6 shadow-2xl relative overflow-hidden bg-gradient-to-r from-slate-900/90 via-cyan-950/30 to-slate-900/90">
        <div className="flex flex-wrap items-start justify-between gap-4 relative z-10">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-xl shadow-cyan-500/20">
                <LayoutTemplate size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-display font-black text-white tracking-wide flex items-center gap-2">
                  Workspace Studio Pro
                  <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-xs font-mono border border-cyan-500/30">
                    Mission Control
                  </span>
                </h1>
                <p className="text-xs text-slate-400">
                  Compose, dock, auto-tile, and orchestrate floating research tools and multi-window experiments.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono text-slate-300 pt-2 flex-wrap">
              <span className="flex items-center gap-1.5 text-cyan-400 font-bold">
                <Monitor size={14} /> Active Windows: {windows.length}
              </span>
              <span>·</span>
              <span className="flex items-center gap-1.5 text-purple-400 font-bold">
                <Boxes size={14} /> Total Tools: {AVAILABLE_WINDOWS.length}
              </span>
              <span>·</span>
              <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
                <Activity size={14} /> Auto-Dock: Enabled
              </span>
            </div>
          </div>

          {/* Quick Auto-Tile Control Group */}
          <div className="glass p-3 rounded-2xl border border-white/10 flex items-center gap-2 flex-wrap">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mr-1">
              Auto-Dock:
            </span>
            <button
              onClick={tile2x2}
              className="glass-hover px-3 py-1.5 rounded-xl border border-white/10 text-xs font-bold text-slate-200 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Grid size={13} className="text-cyan-400" /> 2x2 Grid
            </button>
            <button
              onClick={tileSplitHorizontal}
              className="glass-hover px-3 py-1.5 rounded-xl border border-white/10 text-xs font-bold text-slate-200 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Columns size={13} className="text-purple-400" /> Columns
            </button>
            <button
              onClick={tileSplitVertical}
              className="glass-hover px-3 py-1.5 rounded-xl border border-white/10 text-xs font-bold text-slate-200 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Layers size={13} className="text-emerald-400" /> Rows
            </button>
            <button
              onClick={cascadeWindows}
              className="glass-hover px-3 py-1.5 rounded-xl border border-white/10 text-xs font-bold text-slate-200 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Maximize2 size={13} className="text-yellow-400" /> Cascade
            </button>
            <button
              onClick={() => {
                closeAllWindows?.();
                setFeedback("Closed all open windows!");
                setTimeout(() => setFeedback(""), 2000);
              }}
              className="bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Trash2 size={13} /> Close All
            </button>
          </div>
        </div>
      </div>

      {feedback && (
        <div className="text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 p-3 rounded-2xl font-medium text-center animate-fade-in flex items-center justify-center gap-2">
          <Check size={15} />
          {feedback}
        </div>
      )}

      {/* ────── Live Interactive Desktop Canvas Preview ────── */}
      <div className="glass rounded-3xl border border-white/10 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Monitor size={16} className="text-cyan-400" />
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Live Desktop Layout Miniature Canvas
            </h3>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <button onClick={minimizeAll} className="hover:text-white flex items-center gap-1 cursor-pointer">
              <Minimize2 size={12} /> Minimize All
            </button>
            <button onClick={restoreAll} className="hover:text-white flex items-center gap-1 cursor-pointer">
              <RotateCcw size={12} /> Restore All
            </button>
          </div>
        </div>

        {/* 16:9 Miniature Monitor Screen */}
        <div className="w-full h-64 bg-[#050814] rounded-2xl border border-white/10 relative overflow-hidden p-2 shadow-inner">
          {windows.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-slate-500 space-y-2">
              <Boxes size={32} className="opacity-40" />
              <p className="text-xs font-medium">Desktop Workspace is Currently Empty</p>
              <p className="text-[10px] text-slate-600">
                Launch a research preset below or click any tool to add it to your live canvas.
              </p>
            </div>
          ) : (
            windows.map((win) => {
              // Scale down coordinates to miniature canvas
              const scaleX = 100 / window.innerWidth;
              const scaleY = 100 / window.innerHeight;
              const left = Math.max(2, Math.min(85, win.x * scaleX));
              const top = Math.max(2, Math.min(85, win.y * scaleY));
              const width = Math.max(10, Math.min(45, win.width * scaleX));
              const height = Math.max(12, Math.min(50, win.height * scaleY));

              return (
                <div
                  key={win.id}
                  onClick={() => bringToFront(win.id)}
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                    zIndex: win.zIndex,
                  }}
                  className={`absolute rounded-lg border p-1 text-[9px] font-mono flex flex-col justify-between transition-all cursor-pointer shadow-lg ${
                    win.minimized
                      ? "bg-slate-900/60 border-white/10 opacity-40"
                      : "bg-cyan-950/80 border-cyan-500/50 text-cyan-200 hover:border-cyan-400"
                  }`}
                >
                  <div className="flex items-center justify-between border-b border-white/10 pb-0.5">
                    <span className="truncate font-bold">{win.title}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeWindow(win.id);
                      }}
                      className="text-slate-400 hover:text-red-400"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="text-[8px] text-slate-400 truncate">
                    {win.width}x{win.height}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* ────── Pro Curated Research Presets ────── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sparkles size={18} className="text-cyan-400" />
              Pro Curated Research Presets
            </h3>
            <p className="text-xs text-slate-400">
              One-click multi-window orchestrations tailored for end-to-end AI workflows.
            </p>
          </div>

          <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/5 text-xs font-semibold">
            {["all", "ai", "gpu", "svd", "paper", "mlops"].map((c) => (
              <button
                key={c}
                onClick={() => setSelectedPresetCat(c)}
                className={`px-3 py-1 rounded-lg uppercase transition-all cursor-pointer ${
                  selectedPresetCat === c
                    ? "bg-cyan-500 text-black font-bold shadow"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPresets.map((preset) => (
            <div
              key={preset.id}
              className="glass p-5 rounded-3xl border border-white/10 hover:border-cyan-500/40 transition-all group flex flex-col justify-between space-y-4 bg-white/[0.01] hover:bg-cyan-950/10 shadow-xl"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center group-hover:scale-105 transition-transform">
                    {preset.icon}
                  </div>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-slate-400 font-mono uppercase">
                    {preset.cat}
                  </span>
                </div>

                <h4 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                  {preset.title}
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {preset.desc}
                </p>

                <div className="flex flex-wrap gap-1 pt-1">
                  {preset.tools.map((t, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded-lg bg-black/40 text-slate-300 text-[10px] font-mono border border-white/5"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              <button
                onClick={() => {
                  preset.action();
                  setFeedback(`Dispatched "${preset.title}" workspace!`);
                  setTimeout(() => setFeedback(""), 2500);
                }}
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
              >
                <Play size={13} fill="currentColor" /> Launch Workspace
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* ────── Floating Tools Directory (106 Tools) ────── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Boxes size={18} className="text-purple-400" />
              Floating Tools Directory ({AVAILABLE_WINDOWS.length})
            </h3>
            <p className="text-xs text-slate-400">
              Click any tool to launch as an interactive floating window.
            </p>
          </div>

          <div className="relative w-72">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <input
              type="text"
              placeholder="Search tools by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-black/50 border border-white/10 rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2.5">
          {filteredTools.map((tool) => (
            <button
              key={tool.type}
              onClick={() => {
                openWindow(tool.type, { title: tool.title });
                setFeedback(`Launched "${tool.title}" window!`);
                setTimeout(() => setFeedback(""), 1500);
              }}
              className="glass-hover p-3 rounded-2xl border border-white/5 hover:border-cyan-500/40 text-left transition-all group flex flex-col justify-between gap-2 bg-white/[0.01] hover:bg-cyan-950/20 cursor-pointer shadow-md"
            >
              <span className="text-xl group-hover:scale-110 transition-transform">
                {tool.icon}
              </span>
              <span className="text-xs font-semibold text-slate-200 group-hover:text-white truncate">
                {tool.title}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* ────── Custom Saved Workspaces & JSON Portability ────── */}
      <div className="glass rounded-3xl border border-white/10 p-6 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <FolderOpen size={16} className="text-emerald-400" />
              Custom Saved Workspaces & Portability
            </h3>
            <p className="text-xs text-slate-400">
              Save current arrangement, import configurations, or export shareable JSON templates.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportJSON}
              className="glass-hover border border-white/10 text-slate-200 px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 cursor-pointer"
            >
              <Download size={13} className="text-cyan-400" /> Export JSON
            </button>
            <label className="glass-hover border border-white/10 text-slate-200 px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 cursor-pointer">
              <Upload size={13} className="text-purple-400" /> Import JSON
              <input
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImportJSON}
              />
            </label>
          </div>
        </div>

        <div className="flex gap-2 max-w-md">
          <input
            type="text"
            placeholder="Save active layout as..."
            value={layoutName}
            onChange={(e) => setLayoutName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
            className="flex-1 bg-black/50 border border-white/10 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/50"
          />
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold rounded-xl text-xs flex items-center gap-1.5 shadow-md cursor-pointer"
          >
            <Save size={14} /> Save Layout
          </button>
        </div>

        {savedLayouts.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 pt-2">
            {savedLayouts.map((name) => (
              <div
                key={name}
                className="glass p-3 rounded-2xl border border-white/5 hover:border-cyan-500/30 flex items-center justify-between group cursor-pointer transition-all"
                onClick={() => handleLoad(name)}
              >
                <div className="flex items-center gap-2 truncate">
                  <FolderOpen size={14} className="text-cyan-400 flex-shrink-0" />
                  <span className="text-xs text-slate-200 font-medium truncate">
                    {name}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(name);
                    }}
                    className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
