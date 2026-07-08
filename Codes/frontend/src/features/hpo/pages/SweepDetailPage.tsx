import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { hpoAPI } from "../services/api";
import { TrialTable } from "../components/TrialTable";
import { HyperparameterImportance } from "../components/HyperparameterImportance";
import { ParallelCoordinates } from "../components/ParallelCoordinates";
import { ContourPlot } from "../components/ContourPlot";
import { OptimizationHistory } from "../components/OptimizationHistory";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { ArrowLeft, Play, Square } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function SweepDetailPage({ onClose }: { onClose?: () => void }) {
  const { sweepId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: sweep, isLoading: sweepLoading } = useQuery({
    queryKey: ["sweep", sweepId],
    queryFn: () => hpoAPI.getSweep(sweepId!),
    enabled: !!sweepId,
  });

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["sweep-stats", sweepId],
    queryFn: () => hpoAPI.getSweepStats(sweepId!),
    enabled: !!sweepId,
  });

  const stopMutation = useMutation({
    mutationFn: () => hpoAPI.stopSweep(sweepId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sweep", sweepId] });
      success("Sweep stopped");
    },
  });

  if (sweepLoading || statsLoading) return <Spinner />;
  if (!sweep || !stats)
    return <div className="p-4 text-red-400">Sweep not found</div>;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate("/hpo")}>
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="text-xl font-bold text-white">{sweep.name}</h1>
            <p className="text-xs text-slate-400">
              {sweep.algorithm} · {sweep.objective}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {sweep.status === "running" && (
            <Button
              size="sm"
              variant="danger"
              onClick={() => stopMutation.mutate()}
            >
              <Square size={14} className="mr-1" /> Stop
            </Button>
          )}
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="p-4 space-y-4 overflow-auto">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="Trials" value={stats.total_trials} />
          <StatCard label="Completed" value={stats.completed_trials} />
          <StatCard label="Pruned" value={stats.pruned_trials} />
          <StatCard label="Failed" value={stats.failed_trials} />
          <StatCard
            label="Best Value"
            value={stats.best_value?.toFixed(4) || "-"}
            highlight
          />
        </div>

        <TerminalCard className="p-3">
          <div className="text-sm font-bold text-white mb-2">
            Hyperparameters
          </div>
          <div className="flex flex-wrap gap-2">
            {sweep.hyperparameters.map((hp) => (
              <Badge key={hp.name} variant="secondary">
                {hp.name}: {hp.type}
              </Badge>
            ))}
          </div>
        </TerminalCard>

        <Tabs defaultValue="trials">
          <TabList>
            <Tab value="trials">Trials</Tab>
            <Tab value="importance">Importance</Tab>
            <Tab value="parallel">Parallel Coords</Tab>
            <Tab value="contour">Contour</Tab>
            <Tab value="history">History</Tab>
          </TabList>
          <TabPanel value="trials">
            <TrialTable sweepId={sweepId!} />
          </TabPanel>
          <TabPanel value="importance">
            <HyperparameterImportance sweepId={sweepId!} />
          </TabPanel>
          <TabPanel value="parallel">
            <ParallelCoordinates sweepId={sweepId!} />
          </TabPanel>
          <TabPanel value="contour">
            <ContourPlot sweepId={sweepId!} />
          </TabPanel>
          <TabPanel value="history">
            <OptimizationHistory sweepId={sweepId!} />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}

const StatCard = ({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) => (
  <TerminalCard className="text-center p-3">
    <div
      className={`text-2xl font-bold ${highlight ? "text-neon-mint" : "text-white"}`}
    >
      {value}
    </div>
    <div className="text-xs text-slate-500 uppercase tracking-wider">
      {label}
    </div>
  </TerminalCard>
);
