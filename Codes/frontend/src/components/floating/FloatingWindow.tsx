import { Suspense, lazy, useRef, useEffect } from "react";
import { Rnd } from "react-rnd";
import { useFloatingWindowStore } from "./FloatingWindowManager";

// Lazy load content panels
const TerminalPanel = lazy(
  () => import("../../features/terminal/components/TerminalPanel"),
);
const PaperEditorPanel = lazy(
  () => import("../../features/paper/components/PaperEditorPanel"),
);
const NotesEditorPanel = lazy(
  () => import("../../features/notes/components/FloatingNotesEditor"),
);
const SystemMonitorPanel = lazy(
  () => import("../../features/hardware/components/SystemMonitorPanel"),
);
const ChatPanel = lazy(
  () => import("../../features/chat/components/ChatPanel"),
);
const AcceleratorPanel = lazy(
  () => import("../../features/accelerator/pages/AcceleratorDashboard"),
);
const HPOPanel = lazy(
  () => import("../../features/hpo/pages/HPODashboard"),
);
const AnomalyPanel = lazy(
  () => import("../../features/anomaly/pages/AnomalyListPage"),
);
const ForensicsPanel = lazy(
  () => import("../../features/forensics/pages/FailureAnalysisPage"),
);
const SupportPanel = lazy(
  () => import("../../features/support/pages/SupportHubPage"),
);
const ResearchAnalysisPanel = lazy(
  () => import("../../features/analysis/pages/ResearchAnalysisPage"),
);
const ExperimentsPanel = lazy(
  () => import("../../features/experiments/pages/ExperimentListPage"),
);

const CONTENT_RENDERERS: Record<string, React.ComponentType<any>> = {
  terminal: TerminalPanel,
  chat: ChatPanel,
  "paper-editor": PaperEditorPanel,
  "notes-editor": NotesEditorPanel,
  notes: NotesEditorPanel,
  "system-monitor": SystemMonitorPanel,
  accelerator: AcceleratorPanel,
  hpo: HPOPanel,
  sweeps: HPOPanel,
  anomaly: AnomalyPanel,
  forensics: ForensicsPanel,
  support: SupportPanel,
  analysis: ResearchAnalysisPanel,
  experiments: ExperimentsPanel,
};

export default function FloatingWindowRenderer() {
  const {
    windows,
    removeWindow,
    toggleMinimize,
    toggleMaximize,
    updatePosition,
    updateSize,
    focusWindow,
    bringToFront,
    focusedId,
  } = useFloatingWindowStore();

  // Ref for dragging to avoid accidental clicks
  const dragRef = useRef<boolean>(false);

  const minimizedWindows = windows.filter((w) => w.minimized);

  return (
    <>
      {windows.map((win) => {
        if (win.minimized) {
          // Minimized representation (taskbar icon aligned horizontally from right)
          const minIdx = minimizedWindows.findIndex((w) => w.id === win.id);
          const offsetRight = 16 + minIdx * 150; // 150px spacing

          return (
            <div
              key={win.id}
              onClick={(e) => {
                e.stopPropagation();
                bringToFront(win.id);
                toggleMinimize(win.id);
              }}
              style={{ right: `${offsetRight}px` }}
              className={`fixed bottom-4 z-[200] glass px-3 py-2 rounded-xl cursor-pointer flex items-center justify-between gap-2 text-xs font-bold text-white transition-all hover:scale-105 w-[140px] border ${
                focusedId === win.id
                  ? "border-neon-blue/60 shadow-neon-blue/20 shadow-md"
                  : "border-white/10"
              }`}
            >
              <div className="flex items-center gap-1.5 truncate">
                <span>{win.icon || "▣"}</span>
                <span className="truncate">{win.title}</span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeWindow(win.id);
                }}
                className="w-4 h-4 rounded-full flex items-center justify-center hover:bg-red-500/30 text-slate-400 hover:text-red-400 text-[10px]"
                title="Close"
              >
                ✕
              </button>
            </div>
          );
        }

        const ContentRenderer = CONTENT_RENDERERS[win.content as string];

        // Determine if window is maximized
        const isMaximized = win.maximized === true;
        const rndProps: any = isMaximized
          ? {
              position: { x: 0, y: 0 },
              size: { width: "100%", height: "100%" },
              disableDragging: true,
              enableResizing: false,
              style: {
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                zIndex: win.zIndex ?? 100,
              },
            }
          : {
              position: { x: win.x ?? 100, y: win.y ?? 100 },
              size: { width: win.width ?? 700, height: win.height ?? 500 },
              disableDragging: false,
              enableResizing: true,
              minWidth: 300,
              minHeight: 200,
              style: { position: "fixed", zIndex: win.zIndex ?? 100 },
            };

        return (
          <Rnd
            key={win.id}
            {...rndProps}
            onDragStart={() => {
              dragRef.current = true;
              bringToFront(win.id);
            }}
            onDragStop={(e, d) => {
              dragRef.current = false;
              updatePosition(win.id, d.x, d.y);
            }}
            onResizeStop={(e, dir, ref, delta, position) => {
              updatePosition(win.id, position.x, position.y);
              updateSize(win.id, ref.offsetWidth, ref.offsetHeight);
            }}
            className={`glass rounded-2xl shadow-2xl transition-shadow ${
              focusedId === win.id
                ? "border-neon-blue/50 shadow-neon-blue/20"
                : "border-white/10"
            }`}
            onMouseDown={() => {
              if (!dragRef.current) {
                focusWindow(win.id);
                bringToFront(win.id);
              }
            }}
          >
            {/* Title Bar */}
            <div
              className="h-9 bg-cyber-800 border-b border-white/5 flex items-center justify-between px-3 cursor-grab active:cursor-grabbing select-none"
              onMouseDown={() => {
                // Prevent focus loss when clicking on title bar
                focusWindow(win.id);
                bringToFront(win.id);
              }}
            >
              <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
                <span>{win.icon || "▣"}</span>
                <span className="uppercase tracking-wider">{win.title}</span>
              </div>
              <div className="flex gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleMinimize(win.id);
                  }}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 text-slate-400 hover:text-white text-xs"
                  title="Minimize"
                >
                  —
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleMaximize(win.id);
                  }}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 text-slate-400 hover:text-white text-xs"
                  title={isMaximized ? "Restore" : "Maximize"}
                >
                  □
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeWindow(win.id);
                  }}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-red-500/20 text-slate-400 hover:text-red-400 text-xs"
                  title="Close"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="h-[calc(100%-36px)] overflow-auto">
              <Suspense
                fallback={
                  <div className="flex items-center justify-center h-full text-slate-400 text-xs">
                    Loading...
                  </div>
                }
              >
                {ContentRenderer ? (
                  <ContentRenderer onClose={() => removeWindow(win.id)} />
                ) : (
                  <div className="p-4 text-slate-400">Unknown window type</div>
                )}
              </Suspense>
            </div>
          </Rnd>
        );
      })}
    </>
  );
}
