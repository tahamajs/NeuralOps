import { Badge } from "../../../components/ui/Badge";

export function PipelineStep({ step }: any) {
  return (
    <div className="flex justify-between items-center p-2 rounded bg-white/5">
      <span className="text-sm text-white">{step.name}</span>
      <Badge
        variant={
          step.status === "completed"
            ? "success"
            : step.status === "running"
              ? "warning"
              : step.status === "failed"
                ? "failed"
                : "secondary"
        }
      >
        {step.status}
      </Badge>
    </div>
  );
}
