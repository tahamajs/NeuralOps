import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { streaksAPI } from "../services/api";
import { WeeklySummaryCard } from "../components/WeeklySummaryCard";
import { WeeklyActivityChart } from "../components/WeeklyActivityChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Calendar, RefreshCw, TrendingUp } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function WeeklySummaryViewer({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedWeek, setSelectedWeek] = useState<string>("latest");
  const { success, error } = useToast();

  const {
    data: summaries,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["weekly-summaries", selectedWeek],
    queryFn: () => streaksAPI.getWeeklySummaries(undefined, 10),
  });

  const { data: latest } = useQuery({
    queryKey: ["latest-weekly-summary"],
    queryFn: () => streaksAPI.getMyWeeklySummary(),
  });

  const generateMutation = useMutation({
    mutationFn: () => streaksAPI.generateWeeklySummary(),
    onSuccess: () => {
      refetch();
      success("Weekly summary generated");
    },
    onError: () => error("Failed to generate summary"),
  });

  const weekOptions =
    summaries?.map((s, idx) => ({
      value: s.week_start,
      label: `${s.week_start} - ${s.week_end}`,
    })) || [];
  const displaySummary =
    selectedWeek === "latest"
      ? latest
      : summaries?.find((s) => s.week_start === selectedWeek);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Calendar size={20} className="text-neon-mint" /> Weekly Summaries
          </h1>
          <p className="text-xs text-slate-400">
            AI‑generated insights based on your activity
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={[
              { value: "latest", label: "Latest Week" },
              ...weekOptions,
            ]}
            value={selectedWeek}
            onChange={(e) => setSelectedWeek(e.target.value)}
            className="w-48"
          />
          <Button
            size="sm"
            variant="ghost"
            onClick={() => generateMutation.mutate()}
          >
            <RefreshCw size={14} className="mr-1" /> Generate
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-6">
        {displaySummary ? (
          <>
            <WeeklySummaryCard summary={displaySummary} />
            <WeeklyActivityChart summary={displaySummary} />
          </>
        ) : (
          <TerminalCard className="p-8 text-center">
            <TrendingUp
              size={48}
              className="mx-auto mb-4 text-slate-500 opacity-30"
            />
            <h3 className="text-lg font-medium text-white mb-1">
              No summary available
            </h3>
            <p className="text-slate-400 text-sm">
              Complete activities this week and generate your first summary.
            </p>
            <Button className="mt-4" onClick={() => generateMutation.mutate()}>
              Generate Now
            </Button>
          </TerminalCard>
        )}
      </div>
    </div>
  );
}
