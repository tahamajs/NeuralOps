import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";

export default function JobSchedulingPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const queryClient = useQueryClient();
  const { data: job, isLoading } = useQuery({
    queryKey: ["job", jobId],
    queryFn: async () => (await api.get(`/api/jobs/${jobId}/`)).data,
  });

  const [form, setForm] = useState({
    priority: 5,
    estimated_duration: "",
    memory_requirements: "",
  });
  useEffect(() => {
    if (job) {
      setForm({
        priority: job.priority,
        estimated_duration: job.estimated_duration || "",
        memory_requirements: job.memory_requirements || "",
      });
    }
  }, [job]);

  const updateMutation = useMutation({
    mutationFn: async (payload: any) =>
      api.patch(`/api/jobs/${jobId}/scheduling/`, payload),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        JOB <span className="text-neon-blue">SCHEDULING</span>
      </h1>
      <Card>
        <div className="space-y-4">
          <Input
            label="Priority (1-10)"
            type="number"
            value={form.priority}
            onChange={(e) =>
              setForm({ ...form, priority: parseInt(e.target.value) || 1 })
            }
            min={1}
            max={10}
          />
          <Input
            label="Estimated Duration (seconds)"
            type="number"
            value={form.estimated_duration}
            onChange={(e) =>
              setForm({ ...form, estimated_duration: e.target.value })
            }
          />
          <Input
            label="Memory Requirements (GB)"
            type="number"
            value={form.memory_requirements}
            onChange={(e) =>
              setForm({ ...form, memory_requirements: e.target.value })
            }
          />
          <Button
            onClick={() => updateMutation.mutate(form)}
            disabled={updateMutation.isPending}
          >
            Update Schedule
          </Button>
        </div>
      </Card>
    </div>
  );
}
