import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";

export default function StopAllJobsPage() {
  const mutation = useMutation({
    mutationFn: async () => api.post("/runs/stop_all/"),
    onSuccess: () => alert("All jobs stopped"),
  });

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="text-center space-y-4 max-w-md w-full">
        <h1 className="text-2xl font-black text-white">
          STOP ALL <span className="text-red-500">JOBS</span>
        </h1>
        <p className="text-slate-400 text-sm">
          This will stop all currently running jobs.
        </p>
        <Button
          variant="danger"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Stopping..." : "Confirm Stop All"}
        </Button>
      </Card>
    </div>
  );
}
