// src/features/annotation/pages/AIPreLabelingPanel.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { annotationAPI } from "../services/api";
import { useAnnotationWebSocket } from "../hooks/useAnnotationWebSocket";
import { ConfidenceSlider } from "../components/ConfidenceSlider";
import { PreLabelProgress } from "../components/PreLabelProgress";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Sparkles, Rocket, Settings, CheckCircle, Clock } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AIPreLabelingPanel({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { projectId } = useParams();
  const [model, setModel] = useState("yolov8");
  const [threshold, setThreshold] = useState(0.7);
  const [autoApply, setAutoApply] = useState(false);
  const { preLabelProgress, isConnected } = useAnnotationWebSocket(
    projectId || null,
  );
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: project } = useQuery({
    queryKey: ["annotation-project", projectId],
    queryFn: () => annotationAPI.getProject(projectId!),
    enabled: !!projectId,
  });

  const { data: preLabelStatus, refetch } = useQuery({
    queryKey: ["prelabel-status", projectId],
    queryFn: () => annotationAPI.getPreLabelStatus(projectId!),
    enabled: !!projectId,
  });

  const triggerPreLabel = useMutation({
    mutationFn: () => annotationAPI.triggerPreLabeling(projectId!, model),
    onSuccess: () => {
      success("Pre‑labeling started");
      refetch();
    },
  });

  const applyPreLabel = useMutation({
    mutationFn: (itemId: string) => annotationAPI.applyPreLabel(itemId, "auto"),
    onSuccess: () => success("Pre‑label applied"),
  });

  if (!project) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Sparkles size={20} className="text-neon-mint" /> AI Pre‑labeling
          </h1>
          <p className="text-xs text-slate-400">
            Accelerate annotation with model predictions
          </p>
        </div>
        <div className="flex gap-2">
          <div
            className={`px-2 py-1 rounded text-xs ${isConnected ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}
          >
            {isConnected ? "● Live" : "○ Offline"}
          </div>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <Select
                label="Model"
                options={[
                  { value: "yolov8", label: "YOLOv8 (Object Detection)" },
                  { value: "sam", label: "Segment Anything (SAM)" },
                  { value: "bert", label: "BERT (NER)" },
                ]}
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
              <ConfidenceSlider value={threshold} onChange={setThreshold} />
            </div>
            <div>
              <div className="text-sm text-slate-400 mb-1">Auto‑apply</div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoApply}
                  onChange={(e) => setAutoApply(e.target.checked)}
                  className="accent-neon-blue"
                />
                <span className="text-sm text-white">
                  Apply predictions automatically when confidence &gt; threshold
                </span>
              </label>
              <Button
                className="mt-4 w-full"
                onClick={() => triggerPreLabel.mutate()}
                disabled={triggerPreLabel.isPending}
              >
                <Rocket size={14} className="mr-1" /> Run Pre‑labeling
              </Button>
            </div>
          </div>
        </TerminalCard>
        <PreLabelProgress progress={preLabelProgress} status={preLabelStatus} />
        {preLabelStatus?.items && (
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">
              Pre‑labeled Items
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {preLabelStatus.items.slice(0, 10).map((item: any) => (
                <div
                  key={item.id}
                  className="flex justify-between items-center p-2 rounded bg-white/5"
                >
                  <div>
                    <span className="text-sm text-white">
                      {item.id.slice(0, 8)}
                    </span>
                    <div className="flex gap-1 mt-1">
                      {item.pre_labels.slice(0, 2).map((p: any) => (
                        <Badge key={p.id} variant="info">
                          {(p.confidence * 100).toFixed(0)}%
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => applyPreLabel.mutate(item.id)}
                  >
                    <CheckCircle size={12} /> Apply
                  </Button>
                </div>
              ))}
            </div>
          </TerminalCard>
        )}
      </div>
    </div>
  );
}
