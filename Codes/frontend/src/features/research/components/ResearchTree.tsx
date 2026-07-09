// src/features/research/components/ResearchTree.tsx
import React from "react";
import { CheckCircle2, Circle, Clock, ChevronDown, ChevronRight, Play } from "lucide-react";
import { cn } from "../../../lib/utils";

export interface ResearchTask {
  id: string;
  label: string;
  status: "queued" | "running" | "completed";
  duration?: string;
  children?: ResearchTask[];
}

interface ResearchTreeProps {
  tasks: ResearchTask[];
  expanded: Record<string, boolean>;
  onToggle: (setter: (prev: Record<string, boolean>) => Record<string, boolean>) => void;
}

export function ResearchTree({ tasks = [], expanded = {}, onToggle }: ResearchTreeProps) {
  // If tasks is empty, use a premium default nested research plan
  const defaultTasks: ResearchTask[] = tasks.length > 0 ? tasks : [
    {
      id: "1",
      label: "Hypothesis Formulation",
      status: "completed",
      duration: "1.2s",
      children: [
        { id: "1.1", label: "Analyze attention bottlenecks", status: "completed", duration: "0.5s" },
        { id: "1.2", label: "Define low-rank projection rank size", status: "completed", duration: "0.7s" },
      ],
    },
    {
      id: "2",
      label: "Literature Graph Synthesis",
      status: "completed",
      duration: "4.8s",
      children: [
        { id: "2.1", label: "Search publications for SVD linear attention", status: "completed", duration: "2.3s" },
        { id: "2.2", label: "Extract citation connections", status: "completed", duration: "2.5s" },
      ],
    },
    {
      id: "3",
      label: "Mathematical Proof Verification",
      status: "running",
      children: [
        { id: "3.1", label: "Formulate spectral norm bounds", status: "completed", duration: "3.1s" },
        { id: "3.2", label: "Verify stability limits on long sequences", status: "running" },
      ],
    },
    {
      id: "4",
      label: "Model Sweep Simulation",
      status: "queued",
      children: [
        { id: "4.1", label: "Generate Optuna configuration script", status: "queued" },
        { id: "4.2", label: "Analyze validation loss convergence", status: "queued" },
      ],
    },
  ];

  const handleToggle = (id: string) => {
    if (onToggle) {
      onToggle((prev) => ({
        ...prev,
        [id]: !prev[id],
      }));
    }
  };

  const renderNode = (node: ResearchTask, level = 0) => {
    const isExpanded = !!expanded[node.id];
    const hasChildren = node.children && node.children.length > 0;
    
    let StatusIcon = Clock;
    let iconClass = "text-slate-500";
    
    if (node.status === "completed") {
      StatusIcon = CheckCircle2;
      iconClass = "text-emerald-400";
    } else if (node.status === "running") {
      StatusIcon = Play;
      iconClass = "text-neon-blue animate-pulse";
    }

    return (
      <div key={node.id} className="select-none text-xs">
        <div
          onClick={() => handleToggle(node.id)}
          className={cn(
            "flex items-center justify-between py-2 px-2.5 rounded-xl cursor-pointer hover:bg-white/5 transition-colors border border-transparent",
            node.status === "running" && "bg-neon-blue/5 border-neon-blue/10",
            level > 0 && "ml-4 border-l border-white/5 pl-4"
          )}
        >
          <div className="flex items-center gap-2 min-w-0">
            {hasChildren ? (
              isExpanded ? (
                <ChevronDown size={14} className="text-slate-400 shrink-0" />
              ) : (
                <ChevronRight size={14} className="text-slate-400 shrink-0" />
              )
            ) : (
              <span className="w-3.5 shrink-0" />
            )}
            <StatusIcon size={14} className={iconClass} />
            <span className={cn(
              "truncate font-medium",
              node.status === "completed" && "text-slate-400 line-through decoration-slate-600/50",
              node.status === "running" && "text-white font-bold"
            )}>
              {node.label}
            </span>
          </div>
          {node.duration && (
            <span className="text-[9px] text-slate-500 font-mono shrink-0">{node.duration}</span>
          )}
        </div>
        {hasChildren && isExpanded && (
          <div className="mt-1 space-y-1">
            {node.children!.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-1.5 font-sans">
      {defaultTasks.map((task) => renderNode(task))}
    </div>
  );
}
