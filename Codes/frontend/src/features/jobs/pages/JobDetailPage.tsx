import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Input } from "../../../components/ui/Input";

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const queryClient = useQueryClient();

  const { data: job, isLoading } = useQuery({
    queryKey: ["job", jobId],
    queryFn: async () => (await api.get(`/jobs/${jobId}/`)).data,
  });

  const pauseMutation = useMutation({
    mutationFn: () => api.post(`/jobs/${jobId}/pause/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });
  const resumeMutation = useMutation({
    mutationFn: () => api.post(`/jobs/${jobId}/resume/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });
  const stopMutation = useMutation({
    mutationFn: () => api.post(`/jobs/${jobId}/stop/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });
  const retryMutation = useMutation({
    mutationFn: () => api.post(`/jobs/${jobId}/retry/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  const [notes, setNotes] = useState("");
  useEffect(() => {
    if (job) setNotes(job.notes || "");
  }, [job]);

  const saveNotes = useMutation({
    mutationFn: (notes: string) =>
      api.post(`/jobs/${jobId}/notes/`, { notes }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-black text-white">
          JOB <span className="text-neon-blue">{job.id.slice(0, 8)}</span>
        </h1>
        <Badge status={job.status}>{job.status}</Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 space-y-4">
          <h3 className="text-sm font-bold text-white">Configuration</h3>
          <pre className="text-xs text-slate-400 bg-cyber-800 p-3 rounded-lg">
            {JSON.stringify(
              {
                model: job.model_name,
                task: job.task_name,
                attention: job.attention_type,
                method: job.method,
                epochs: job.epochs,
                batch_size: job.batch_size,
                lr: job.learning_rate,
                compression_rank: job.compression_rank,
              },
              null,
              2,
            )}
          </pre>
          <h3 className="text-sm font-bold text-white mt-4">Notes</h3>
          <textarea
            className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={5}
          />
          <Button
            size="sm"
            onClick={() => saveNotes.mutate(notes)}
            disabled={saveNotes.isPending}
          >
            Save Notes
          </Button>
        </Card>

        <Card className="space-y-4">
          <h3 className="text-sm font-bold text-white">Actions</h3>
          <div className="flex flex-col gap-2">
            {job.status === "running" && (
              <Button
                variant="secondary"
                onClick={() => pauseMutation.mutate()}
              >
                Pause
              </Button>
            )}
            {job.status === "paused" && (
              <Button
                variant="secondary"
                onClick={() => resumeMutation.mutate()}
              >
                Resume
              </Button>
            )}
            {(job.status === "running" || job.status === "queued") && (
              <Button variant="danger" onClick={() => stopMutation.mutate()}>
                Stop
              </Button>
            )}
            {(job.status === "failed" || job.status === "stopped") && (
              <Button variant="primary" onClick={() => retryMutation.mutate()}>
                Retry
              </Button>
            )}
          </div>
          <div className="text-xs text-slate-400">
            <p>Priority: {job.priority}</p>
            <p>
              GPU: {job.gpu_requirements} / RAM: {job.memory_requirements} GB
            </p>
            <p>Created: {new Date(job.created_at).toLocaleString()}</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
