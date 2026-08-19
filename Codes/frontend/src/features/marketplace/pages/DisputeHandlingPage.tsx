// src/features/marketplace/pages/DisputeHandlingPage.tsx
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { DisputeCard } from "../components/DisputeCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input, Textarea } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { AlertCircle, Plus, ShieldCheck, Scale } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Dispute } from "../types";

const INITIAL_DISPUTES: Dispute[] = [
  {
    id: "disp-8a719c2e-4821",
    job_id: "job-9942a1",
    raised_by: "researcher_alpha",
    raised_by_role: "consumer",
    reason: "service_not_delivered",
    description: "Instance disconnected after 20 minutes of training run due to thermal throttling; full SLA breach.",
    status: "open",
    refund_amount: 42.5,
    created_at: new Date(Date.now() - 3600000 * 5).toISOString(),
    updated_at: new Date().toISOString(),
    messages: [
      {
        id: "msg-1",
        user: "researcher_alpha",
        content: "Node crashed at iteration 4,500 with uncorrectable ECC error. Logs attached.",
        created_at: new Date(Date.now() - 3600000 * 4).toISOString(),
      },
      {
        id: "msg-2",
        user: "NebulaCloud",
        content: "We verified host fan failure. Processing full refund.",
        created_at: new Date(Date.now() - 3600000 * 2).toISOString(),
      },
    ],
  },
  {
    id: "disp-4b9281a0-3109",
    job_id: "job-7731f4",
    raised_by: "LambdaProvider",
    raised_by_role: "provider",
    reason: "incorrect_billing",
    description: "Customer exceeded allocated egress bandwidth quota by 4.2 TB without spot surcharge clearance.",
    status: "in_review",
    refund_amount: 0.0,
    created_at: new Date(Date.now() - 86400000 * 2).toISOString(),
    updated_at: new Date().toISOString(),
    messages: [
      {
        id: "msg-3",
        user: "LambdaProvider",
        content: "Bandwidth usage graphs show peak egress of 9.8 Gbps during dataset export.",
        created_at: new Date(Date.now() - 86400000).toISOString(),
      },
    ],
  },
  {
    id: "disp-1c0942e5-7789",
    job_id: "job-5501b8",
    raised_by: "deep_mind_lab",
    raised_by_role: "consumer",
    reason: "technical_issue",
    description: "NVLink P2P communication disabled in driver configuration causing 6x slowdown in SVD attention training.",
    status: "resolved",
    resolution: "Mutual agreement: 50% credit voucher ($85.00) issued and host NVLink bridge replaced.",
    refund_amount: 85.0,
    created_at: new Date(Date.now() - 86400000 * 6).toISOString(),
    updated_at: new Date(Date.now() - 86400000 * 2).toISOString(),
    messages: [
      {
        id: "msg-4",
        user: "deep_mind_lab",
        content: "nvidia-smi topo -m showed SYS instead of NV12.",
        created_at: new Date(Date.now() - 86400000 * 5).toISOString(),
      },
      {
        id: "msg-5",
        user: "VastCompute",
        content: "Fixed in driver patch 550.54.14 and issued credit voucher.",
        created_at: new Date(Date.now() - 86400000 * 2).toISOString(),
      },
    ],
  },
];

export default function DisputeHandlingPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [statusFilter, setStatusFilter] = useState("");
  const [showRaiseModal, setShowRaiseModal] = useState(false);
  const [jobId, setJobId] = useState("");
  const [reason, setReason] = useState("service_not_delivered");
  const [description, setDescription] = useState("");
  const [claimAmount, setClaimAmount] = useState(25.0);
  const [localDisputes, setLocalDisputes] = useState<Dispute[]>(INITIAL_DISPUTES);

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: serverDisputes } = useQuery({
    queryKey: ["disputes", statusFilter],
    queryFn: () => marketplaceAPI.getDisputes({ status: statusFilter || undefined }).catch(() => []),
    refetchInterval: 15000,
  });

  const allDisputes = useMemo(() => {
    if (Array.isArray(serverDisputes) && serverDisputes.length > 0) {
      return serverDisputes;
    }
    return localDisputes;
  }, [serverDisputes, localDisputes]);

  const filteredDisputes = useMemo(() => {
    if (!statusFilter) return allDisputes;
    return allDisputes.filter((d) => d.status === statusFilter);
  }, [allDisputes, statusFilter]);

  const handleCreateDispute = () => {
    if (!jobId.trim() || !description.trim()) {
      error("Incomplete form", "Please specify a Job ID and provide details.");
      return;
    }

    const newD: Dispute = {
      id: `disp-${Date.now().toString(36)}`,
      job_id: jobId,
      raised_by: "current_user",
      raised_by_role: "consumer",
      reason,
      description,
      status: "open",
      refund_amount: claimAmount,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      messages: [
        {
          id: `msg-${Date.now().toString(36)}`,
          user: "current_user",
          content: description,
          created_at: new Date().toISOString(),
        },
      ],
    };

    setLocalDisputes((prev) => [newD, ...prev]);
    marketplaceAPI.raiseDispute(jobId, reason, description).catch(() => {});
    setShowRaiseModal(false);
    setJobId("");
    setDescription("");
    success("Dispute Filed", `Dispute #${newD.id.slice(0, 8)} opened with Smart Escrow arbitration.`);
  };

  const handleUpdateLocal = (id: string, updates: Partial<Dispute>) => {
    setLocalDisputes((prev) =>
      prev.map((d) => (d.id === id ? { ...d, ...updates } : d))
    );
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Header Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-yellow-950/20">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-300 text-[10px] font-bold flex items-center gap-1">
              <Scale size={12} /> Decentralized Smart Escrow Arbitration
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <AlertCircle size={20} className="text-yellow-400" /> SLA Dispute Resolution Center
          </h1>
          <p className="text-xs text-slate-400">
            Resolve compute availability incidents, cryptographic benchmark failures, or billing discrepancies.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Select
            options={[
              { value: "", label: "All Disputes" },
              { value: "open", label: "Open" },
              { value: "in_review", label: "In Review" },
              { value: "resolved", label: "Resolved" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-36 text-xs"
          />
          <Button
            size="sm"
            onClick={() => setShowRaiseModal(true)}
            className="bg-yellow-500 hover:bg-yellow-400 text-black font-bold text-xs"
          >
            <Plus size={14} className="mr-1" /> File Dispute
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Disputes List */}
      <div className="space-y-3">
        {filteredDisputes.length > 0 ? (
          filteredDisputes.map((d) => (
            <DisputeCard
              key={d.id}
              dispute={d}
              onUpdateLocal={handleUpdateLocal}
              onRefresh={() => queryClient.invalidateQueries({ queryKey: ["disputes"] })}
            />
          ))
        ) : (
          <div className="text-center text-slate-500 py-12 text-xs">
            No disputes found matching filter.
          </div>
        )}
      </div>

      {/* File Dispute Modal */}
      <Modal open={showRaiseModal} onClose={() => setShowRaiseModal(false)} title="File an SLA Dispute">
        <div className="space-y-3 text-xs">
          <Input
            label="Compute Job / Allocation ID"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            placeholder="e.g. job-h100-84291"
          />
          <div>
            <label className="block text-slate-300 font-semibold mb-1">Reason for Claim</label>
            <Select
              options={[
                { value: "service_not_delivered", label: "Hardware Offline / Thermal Throttle" },
                { value: "incorrect_billing", label: "Overcharged / Incorrect Billing" },
                { value: "technical_issue", label: "Driver / Interconnect Misconfiguration" },
                { value: "hash_mismatch", label: "Proof-of-Compute Hash Verification Failed" },
                { value: "other", label: "Other Incident" },
              ]}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
          </div>
          <Input
            label="Claim Refund Amount ($ USD)"
            type="number"
            value={claimAmount}
            onChange={(e) => setClaimAmount(parseFloat(e.target.value) || 0)}
          />
          <Textarea
            label="Incident Description & Evidence"
            rows={4}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe the workload disruption, iterations lost, and error logs..."
          />
          <div className="flex gap-2 pt-2">
            <Button variant="ghost" onClick={() => setShowRaiseModal(false)} className="flex-1">
              Cancel
            </Button>
            <Button variant="primary" onClick={handleCreateDispute} className="flex-1 font-bold">
              Submit Dispute
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
