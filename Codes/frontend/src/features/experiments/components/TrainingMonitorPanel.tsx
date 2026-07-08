import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Switch } from "../../../components/ui/Switch";
import { Play, StopCircle, FileText, BookOpen, TrendingUp } from "lucide-react";

interface Props {
  runId: string;
  status: string;
}

export default function TrainingMonitorPanel({ runId, status }: Props) {
  const [autoStop, setAutoStop] = useState(true);
  const [noteGen, setNoteGen] = useState(true);
  const [checkInterval, setCheckInterval] = useState(0.25);
  const [monitorActive, setMonitorActive] = useState(false);

  const startMonitorMut = useMutation({
    mutationFn: async () => {
      const res = await api.post("/api/agents/monitor/start/", {
        run_id: runId,
        check_interval_ratio: checkInterval,
        auto_stop_enabled: autoStop,
        note_generation_enabled: noteGen,
      });
      return res.data;
    },
    onSuccess: () => setMonitorActive(true),
  });

  const recommendMut = useMutation({
    mutationFn: async () => {
      const res = await api.get(`/api/agents/papers/recommendations/${runId}/`);
      return res.data;
    },
    onSuccess: (data) => {
      alert(
        `Found ${data.recommendations?.length || 0} related papers. Check your notes!`,
      );
    },
  });

  const isRunning = status === "running" || status === "queued";

  return (
    <Card className="border border-neon-blue/20 bg-neon-blue/5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-display font-bold text-white flex items-center gap-2">
          <TrendingUp size={18} className="text-neon-blue" />
          Training Monitor & Auto‑Note
        </h3>
        {monitorActive && <Badge status="running">Active</Badge>}
      </div>

      <p className="text-xs text-slate-400 mb-4">
        The AI agent watches your training at every {checkInterval * 100}% of
        progress, detects issues (overfitting, plateau, divergence), and writes
        notes with charts.
      </p>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400">
            Auto‑Stop on Critical Issues
          </span>
          <Switch checked={autoStop} onChange={setAutoStop} />
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400">
            Auto‑Generate Notes with Charts
          </span>
          <Switch checked={noteGen} onChange={setNoteGen} />
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400">Check Interval</span>
          <select
            className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs text-white"
            value={checkInterval}
            onChange={(e) => setCheckInterval(parseFloat(e.target.value))}
          >
            <option value={0.25}>25% (4 checks)</option>
            <option value={0.1}>10% (10 checks)</option>
            <option value={0.5}>50% (2 checks)</option>
          </select>
        </div>
      </div>

      <div className="flex gap-2 mt-4">
        {!monitorActive && isRunning && (
          <Button
            size="sm"
            onClick={() => startMonitorMut.mutate()}
            disabled={startMonitorMut.isPending}
          >
            <Play size={14} className="mr-1" />
            {startMonitorMut.isPending ? "Starting..." : "Start Monitor"}
          </Button>
        )}
        {monitorActive && (
          <Button size="sm" variant="secondary">
            <StopCircle size={14} className="mr-1" /> Stop Monitor
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => recommendMut.mutate()}
          disabled={recommendMut.isPending}
        >
          <BookOpen size={14} className="mr-1" /> Find Related Papers
        </Button>
      </div>
    </Card>
  );
}
