// src/features/annotation/pages/ReviewQueue.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { annotationAPI } from "../services/api";
import { ReviewItemCard } from "../components/ReviewItemCard";
import { QualityScoreBadge } from "../components/QualityScoreBadge";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import {
  ClipboardList,
  CheckCircle,
  XCircle,
  RotateCcw,
  Filter,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ReviewQueue({ onClose }: { onClose?: () => void }) {
  const { projectId } = useParams();
  const [filter, setFilter] = useState("all");
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: queue, isLoading } = useQuery({
    queryKey: ["review-queue", projectId],
    queryFn: () => annotationAPI.getReviewQueue(projectId!),
    enabled: !!projectId,
  });

  const submitReview = useMutation({
    mutationFn: (result: any) => annotationAPI.submitReview(projectId!, result),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["review-queue", projectId] });
      setSelectedItem(null);
      success("Review submitted");
    },
  });

  const queueData = (queue as any)?.data || queue;

  const filteredQueue =
    queueData?.filter((q: any) => filter === "all" || q.priority === filter) || [];

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <ClipboardList size={20} className="text-yellow-400" /> Review Queue
          </h1>
          <p className="text-xs text-slate-400">
            Quality assurance for submitted annotations
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={[
              { value: "all", label: "All Items" },
              { value: "high", label: "High Priority" },
              { value: "medium", label: "Medium Priority" },
              { value: "low", label: "Low Priority" },
            ]}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-36"
          />
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4">
          {filteredQueue.map((item) => (
            <ReviewItemCard
              key={item.item.id}
              item={item}
              onSelect={() => setSelectedItem(item)}
              selected={selectedItem?.item.id === item.item.id}
            />
          ))}
        </div>
        {selectedItem && (
          <div className="mt-4">
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">
                Review: {selectedItem.item.id.slice(0, 8)}
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-slate-400">Annotations</div>
                  <div className="space-y-1 mt-1">
                    {selectedItem.item.annotations.map((a: any) => (
                      <div key={a.id} className="flex justify-between">
                        <span>{a.label_name}</span>
                        <QualityScoreBadge score={a.confidence} />
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Pre‑labels (AI)</div>
                  <div className="space-y-1 mt-1">
                    {selectedItem.item.pre_labels?.map((p: any) => (
                      <div key={p.id} className="flex justify-between">
                        <span>{p.label_name}</span>
                        <QualityScoreBadge score={p.confidence} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex gap-2 mt-4">
                <Button
                  variant="primary"
                  onClick={() =>
                    submitReview.mutate({
                      item_id: selectedItem.item.id,
                      action: "approve",
                      score: 100,
                    })
                  }
                >
                  <CheckCircle size={14} className="mr-1" /> Approve
                </Button>
                <Button
                  variant="danger"
                  onClick={() =>
                    submitReview.mutate({
                      item_id: selectedItem.item.id,
                      action: "reject",
                    })
                  }
                >
                  <XCircle size={14} className="mr-1" /> Reject
                </Button>
                <Button
                  variant="ghost"
                  onClick={() =>
                    submitReview.mutate({
                      item_id: selectedItem.item.id,
                      action: "requeue",
                    })
                  }
                >
                  <RotateCcw size={14} className="mr-1" /> Requeue
                </Button>
              </div>
            </TerminalCard>
          </div>
        )}
      </div>
    </div>
  );
}
