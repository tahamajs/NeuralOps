import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Textarea } from "../../../components/ui/Input";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { MessageSquare, CheckCircle, XCircle } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export function DisputeCard({ dispute, onRefresh }: any) {
  const [showRespond, setShowRespond] = useState(false);
  const [response, setResponse] = useState("");
  const [showResolve, setShowResolve] = useState(false);
  const [resolution, setResolution] = useState("");
  const [refundAmount, setRefundAmount] = useState(0);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const respond = useMutation({
    mutationFn: () => marketplaceAPI.respondToDispute(dispute.id, response),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["disputes"] });
      setShowRespond(false);
      success("Response sent");
    },
  });

  const resolve = useMutation({
    mutationFn: () =>
      marketplaceAPI.resolveDispute(dispute.id, resolution, refundAmount),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["disputes"] });
      setShowResolve(false);
      success("Dispute resolved");
    },
  });

  const statusConfig = {
    open: "danger",
    in_review: "warning",
    resolved: "success",
    closed: "secondary",
  };

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs">{dispute.id.slice(0, 8)}</span>
            <Badge variant={statusConfig[dispute.status]}>
              {dispute.status}
            </Badge>
            <span className="text-xs text-slate-400">
              Job: {dispute.job_id.slice(0, 8)}
            </span>
          </div>
          <div className="text-sm font-bold text-white mt-1">
            {dispute.reason.replace("_", " ")}
          </div>
          <p className="text-sm text-slate-300 mt-1">{dispute.description}</p>
          <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
            <span>
              Raised by: {dispute.raised_by} ({dispute.raised_by_role})
            </span>
            <span>{new Date(dispute.created_at).toLocaleString()}</span>
          </div>
          {dispute.messages?.length > 0 && (
            <div className="mt-2 border-t border-white/10 pt-2">
              <div className="flex items-center gap-1 text-xs text-slate-400">
                <MessageSquare size={12} /> Messages
              </div>
              <div className="max-h-32 overflow-y-auto mt-1 space-y-1">
                {dispute.messages.map((m) => (
                  <div key={m.id} className="text-sm">
                    <span className="font-medium text-white">{m.user}:</span>{" "}
                    {m.content}
                    <div className="text-[10px] text-slate-500">
                      {new Date(m.created_at).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          {dispute.status !== "resolved" && !showRespond && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowRespond(true)}
            >
              <MessageSquare size={14} /> Respond
            </Button>
          )}
          {dispute.status !== "resolved" &&
            dispute.raised_by_role === "provider" && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowResolve(true)}
              >
                <CheckCircle size={14} /> Resolve
              </Button>
            )}
        </div>
      </div>
      {showRespond && (
        <div className="mt-3 pt-2 border-t border-white/10">
          <Textarea
            rows={2}
            value={response}
            onChange={(e) => setResponse(e.target.value)}
            placeholder="Your response..."
          />
          <div className="flex gap-2 mt-2">
            <Button size="sm" onClick={() => respond.mutate()}>
              Send
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowRespond(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
      {showResolve && (
        <Modal
          open={showResolve}
          onClose={() => setShowResolve(false)}
          title="Resolve Dispute"
        >
          <div className="space-y-3">
            <Input
              label="Resolution Summary"
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
            />
            <Input
              label="Refund Amount (optional)"
              type="number"
              value={refundAmount}
              onChange={(e) => setRefundAmount(parseFloat(e.target.value))}
            />
            <Button onClick={() => resolve.mutate()}>Resolve</Button>
          </div>
        </Modal>
      )}
    </TerminalCard>
  );
}
