// src/features/research/components/ResearchTree.tsx
import {
  ChevronRight,
  ChevronDown,
  CheckCircle,
  Loader2,
  Clock,
} from "lucide-react";

interface TaskNode {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "error";
  children?: TaskNode[];
  duration?: number;
}

export function ResearchTree({ tasks, expanded, onToggle }: any) {
  const getIcon = (status: string) => {
    if (status === "done")
      return <CheckCircle size={16} className="text-green-400" />;
    if (status === "running")
      return <Loader2 size={16} className="animate-spin text-neon-blue" />;
    if (status === "error")
      return <XCircle size={16} className="text-red-400" />;
    return <Clock size={16} className="text-slate-500" />;
  };

  return (
    <div className="space-y-1">
      {tasks.map((task) => (
        <div key={task.id}>
          <div className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-white/5">
            <button
              onClick={() => onToggle(task.id)}
              className="text-slate-400"
            >
              {expanded.has(task.id) ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
            </button>
            {getIcon(task.status)}
            <span className="text-sm text-white">{task.name}</span>
            {task.duration && (
              <span className="text-xs text-slate-500 ml-auto">
                {task.duration}s
              </span>
            )}
          </div>
          {expanded.has(task.id) && task.children && (
            <div className="ml-6 border-l border-white/10 pl-2">
              <ResearchTree
                tasks={task.children}
                expanded={expanded}
                onToggle={onToggle}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
