import React from "react";
import { Card } from "../../../components/ui/Card";
import { Activity, Beaker, CheckCircle2 } from "lucide-react";

export function DistillationStep({ step }: any) {
  const isCompleted = step?.status === "completed";
  const isRunning = step?.status === "running";

  return (
    <Card
      className={`p-4 border ${isCompleted ? "border-green-500/30 bg-green-500/5" : isRunning ? "border-neon-blue/30 bg-neon-blue/5 animate-pulse-slow" : "border-white/10 bg-white/5"} transition-all`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`p-2 rounded-lg ${isCompleted ? "bg-green-500/20 text-green-400" : isRunning ? "bg-neon-blue/20 text-neon-blue" : "bg-slate-800 text-slate-400"}`}
          >
            {isCompleted ? <CheckCircle2 size={18} /> : <Beaker size={18} />}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white">
              {step?.name || "Distillation Process"}
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              {step?.description || "Extracting knowledge from teacher model"}
            </p>
          </div>
        </div>

        {isRunning && (
          <div className="flex items-center gap-2">
            <Activity size={14} className="text-neon-blue animate-pulse" />
            <span className="text-xs font-medium text-neon-blue">
              Processing...
            </span>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="bg-black/30 rounded px-3 py-2">
          <p className="text-[10px] text-slate-500 uppercase">Teacher Size</p>
          <p className="text-xs text-white font-medium">
            {step?.teacherSize || "175B"}
          </p>
        </div>
        <div className="bg-black/30 rounded px-3 py-2">
          <p className="text-[10px] text-slate-500 uppercase">Student Size</p>
          <p className="text-xs text-white font-medium">
            {step?.studentSize || "7B"}
          </p>
        </div>
        <div className="bg-black/30 rounded px-3 py-2">
          <p className="text-[10px] text-slate-500 uppercase">Compression</p>
          <p className="text-xs text-green-400 font-medium">
            {step?.compressionRatio || "25x"}
          </p>
        </div>
      </div>
    </Card>
  );
}
