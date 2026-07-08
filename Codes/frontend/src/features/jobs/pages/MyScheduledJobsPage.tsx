import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { EmptyState } from "../../../components/ui/EmptyState";

export default function MyScheduledJobsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["myJobs"],
    queryFn: async () => (await api.get("/api/jobs/my-jobs/")).data,
    refetchInterval: 5000,
  });

  if (isLoading) return <Spinner />;
  if (!data?.pending?.length && !data?.running?.length)
    return <EmptyState title="No scheduled jobs" />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        MY <span className="text-neon-blue">SCHEDULED JOBS</span>
      </h1>
      {data.running?.length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Running</h3>
          {data.running.map((j: any) => (
            <div
              key={j.job_id}
              className="flex justify-between items-center py-2 border-b border-white/5"
            >
              <span className="font-mono text-neon-blue">
                {j.job_id.slice(0, 8)}
              </span>
              <Badge status="running">Running</Badge>
              <span className="text-xs text-slate-400">GPUs: {j.gpu_ids}</span>
            </div>
          ))}
        </Card>
      )}
      {data.pending?.length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Pending</h3>
          {data.pending.map((j: any) => (
            <div
              key={j.job_id}
              className="flex justify-between items-center py-2 border-b border-white/5"
            >
              <span className="font-mono text-neon-blue">
                {j.job_id.slice(0, 8)}
              </span>
              <Badge status="queued">{j.state}</Badge>
              <span className="text-xs text-slate-400">
                Priority {j.priority} · {j.num_gpus} GPU
              </span>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}
