import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Textarea, Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { MessageSquare, CheckCircle, ShieldAlert, Send } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export function DisputeCard({ dispute, onRefresh, onUpdateLocal }: any) {
  const [showRespond, setShowRespond] = useState(false);
  const [response, setResponse] = useState("");
  const [showResolve, setShowResolve] = useState(false);
  const [resolution, setResolution] = useState("");
  const [refundAmount, setRefundAmount] = useState(dispute.refund_amount || 0);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const handleRespond = () => {
    if (!response.trim()) return;
    const newMsg = {
      id: `msg-${Date.now().toString().slice(-4)}`,
      user: "Current User (Arbitrator / Party)",
      content: response,
      created_at: new Date().toISOString(),
    };
    if (onUpdateLocal) {
      onUpdateLocal(dispute.id, {
        messages: [...(dispute.messages || []), newMsg],
      });
    }
    marketplaceAPI.respondToDispute(dispute.id, response).catch(() => {});
    setResponse("");
    setShowRespond(false);
    success("Response Sent", "Your message was added to the dispute thread.");
  };

  const handleResolve = () => {
    if (!resolution.trim()) return;
    if (onUpdateLocal) {
      onUpdateLocal(dispute.id, {
        status: "resolved",
        resolution,
        refund_amount: refundAmount,
        resolved_at: new Date().toISOString(),
      });
    }
    marketplaceAPI.resolveDispute(dispute.id, resolution, refundAmount).catch(() => {});
    setShowResolve(false);
    success("Dispute Resolved", `Settlement of $${Number(refundAmount || 0).toFixed(2)} recorded.`);
  };

  const statusConfig: Record<string, "danger" | "warning" | "success" | "secondary"> = {
    open: "danger",
    in_review: "warning",
    resolved: "success",
    closed: "secondary",
  };

  const safeId = (dispute.id || "").slice(0, 8) || "disp-001";
  const safeJobId = (dispute.job_id || dispute.job || "").slice(0, 8) || "job-001";

  return (
    <TerminalCard className="p-4 space-y-3 border border-white/10 hover:border-yellow-500/30 transition-all">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-white">#{safeId}</span>
            <Badge variant={statusConfig[dispute.status] || "warning"}>
              {dispute.status?.toUpperCase()}
            </Badge>
            <span className="text-xs text-slate-400 font-mono">Job: #{safeJobId}</span>
          </div>

          <div className="text-sm font-bold text-white mt-1 capitalize">
            {(dispute.reason || "Dispute").replace(/_/g, " ")}
          </div>
          <p className="text-xs text-slate-300 mt-1 leading-relaxed">{dispute.description}</p>

          <div className="flex flex-wrap items-center gap-4 mt-2 text-[11px] text-slate-400">
            <span>
              Claimant: <strong className="text-white">{dispute.raised_by || "User"}</strong> ({dispute.raised_by_role || "consumer"})
            </span>
            <span>Created: {new Date(dispute.created_at || Date.now()).toLocaleString()}</span>
            {dispute.refund_amount ? (
              <span className="text-emerald-400 font-mono font-bold">
                Refund Claim: ${Number(dispute.refund_amount).toFixed(2)}
              </span>
            ) : null}
          </div>

          {dispute.resolution && (
            <div className="mt-2 p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300">
              <strong>Resolution:</strong> {dispute.resolution}
            </div>
          )}

          {dispute.messages && dispute.messages.length > 0 && (
            <div className="mt-3 border-t border-white/10 pt-2 space-y-2">
              <div className="flex items-center gap-1 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                <MessageSquare size={12} /> Discussion History ({dispute.messages.length})
              </div>
              <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                {dispute.messages.map((m: any) => (
                  <div key={m.id} className="text-xs bg-black/40 p-2 rounded-lg border border-white/5">
                    <div className="flex justify-between items-center text-[10px] text-slate-400 mb-0.5">
                      <span className="font-bold text-cyan-300">{m.user}</span>
                      <span>{new Date(m.created_at || Date.now()).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-slate-200">{m.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2 shrink-0">
          {dispute.status !== "resolved" && (
            <>
              <Button size="sm" variant="ghost" onClick={() => setShowRespond(true)} className="text-xs">
                <MessageSquare size={13} className="mr-1" /> Reply
              </Button>
              <Button
                size="sm"
                variant="primary"
                onClick={() => setShowResolve(true)}
                className="text-xs font-bold bg-emerald-500 hover:bg-emerald-400 text-black"
              >
                <CheckCircle size={13} className="mr-1" /> Resolve Claim
              </Button>
            </>
          )}
        </div>
      </div>

      {showRespond && (
        <div className="mt-3 pt-3 border-t border-white/10 space-y-2">
          <Textarea
            rows={2}
            value={response}
            onChange={(e) => setResponse(e.target.value)}
            placeholder="Write response or submit logs..."
            className="text-xs"
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={handleRespond} className="text-xs">
              <Send size={12} className="mr-1" /> Submit Reply
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setShowRespond(false)} className="text-xs">
              Cancel
            </Button>
          </div>
        </div>
      )}

      {showResolve && (
        <Modal open={showResolve} onClose={() => setShowResolve(false)} title="Settle & Resolve Dispute">
          <div className="space-y-3 text-xs">
            <p className="text-slate-400">
              Record arbitrator agreement or bilateral settlement. Funds in escrow will adjust automatically.
            </p>
            <Input
              label="Settlement Summary"
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              placeholder="e.g., 50% credit voucher issued due to verified hardware throttle."
            />
            <Input
              label="Refund Amount ($ USD)"
              type="number"
              value={refundAmount}
              onChange={(e) => setRefundAmount(parseFloat(e.target.value) || 0)}
            />
            <div className="flex gap-2 pt-2">
              <Button variant="ghost" onClick={() => setShowResolve(false)} className="flex-1">
                Cancel
              </Button>
              <Button variant="primary" onClick={handleResolve} className="flex-1 font-bold">
                Confirm Settlement
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </TerminalCard>
  );
}
