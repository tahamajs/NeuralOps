import { useMutation, useQueryClient } from "@tanstack/react-query";
import { acceleratorAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import {
  Lightbulb,
  AlertTriangle,
  TrendingUp,
  Zap,
  X,
  Check,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { ExperimentInsight } from "../types";

interface InsightCardProps {
  insight: ExperimentInsight;
  onDismiss?: () => void;
}

export function InsightCard({ insight, onDismiss }: InsightCardProps) {
  const queryClient = useQueryClient();
  const typeConfig = {
    opportunity: {
      icon: TrendingUp,
      color: "text-green-400",
      bg: "bg-green-500/10",
      label: "Opportunity",
    },
    warning: {
      icon: AlertTriangle,
      color: "text-red-400",
      bg: "bg-red-500/10",
      label: "Warning",
    },
    recommendation: {
      icon: Lightbulb,
      color: "text-yellow-400",
      bg: "bg-yellow-500/10",
      label: "Recommendation",
    },
    anomaly: {
      icon: Zap,
      color: "text-purple-400",
      bg: "bg-purple-500/10",
      label: "Anomaly",
    },
  };
  const config = typeConfig[insight.type];

  const markRead = useMutation({
    mutationFn: () => acceleratorAPI.markInsightRead(insight.id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["accelerator-insights"] }),
  });

  const applySuggestion = useMutation({
    mutationFn: () => acceleratorAPI.applyIntervention(insight.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["accelerator-insights"] });
      markRead.mutate();
    },
  });

  return (
    <TerminalCard className="p-4">
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${config.bg}`}>
          {<config.icon size={18} className={config.color} />}
        </div>
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="info">{config.label}</Badge>
            <span className="text-xs text-slate-500">
              {formatDistanceToNow(new Date(insight.created_at), {
                addSuffix: true,
              })}
            </span>
            {insight.experiment_name && (
              <span className="text-xs text-slate-500">
                on {insight.experiment_name}
              </span>
            )}
          </div>
          <h3 className="text-white font-bold mt-1">{insight.title}</h3>
          <p className="text-sm text-slate-300 mt-1">{insight.description}</p>
          <div className="flex items-center gap-3 mt-3">
            <div className="flex items-center gap-1">
              <span className="text-xs text-slate-400">Impact score:</span>
              <span className="text-sm font-bold text-neon-mint">
                {insight.impact_score}/100
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-xs text-slate-400">Confidence:</span>
              <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-neon-blue rounded-full"
                  style={{ width: `${insight.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-slate-400">
                {Math.round(insight.confidence * 100)}%
              </span>
            </div>
          </div>
        </div>
      </div>
      <div className="flex justify-end gap-2 mt-3 pt-2 border-t border-white/10">
        {insight.suggested_action && insight.status !== "applied" && (
          <Button
            size="sm"
            variant="primary"
            onClick={() => applySuggestion.mutate()}
            disabled={applySuggestion.isPending}
          >
            <Zap size={12} className="mr-1" /> Apply
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => markRead.mutate()}>
          <Check size={12} className="mr-1" /> Mark read
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            onDismiss?.();
            markRead.mutate();
          }}
        >
          <X size={12} className="mr-1" /> Dismiss
        </Button>
      </div>
    </TerminalCard>
  );
}
