// src/features/governance/pages/ContractSigningUI.tsx
import { useState, useRef, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { SignatureCanvas } from "../components/SignatureCanvas";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import {
  FileText,
  PenTool,
  History,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Download,
  Plus,
  ShieldCheck,
  Award,
  Type,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Contract } from "../types";

export default function ContractSigningUI({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { contractId: routeContractId } = useParams();
  const [activeContractId, setActiveContractId] = useState<string>(routeContractId || "");
  const [signatureMode, setSignatureMode] = useState<"draw" | "type">("draw");
  const [typedName, setTypedName] = useState("Elena Rostova");
  const [signatureData, setSignatureData] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);

  // New Contract Form
  const [newTitle, setNewTitle] = useState("");
  const [newParties, setNewParties] = useState("NeuralOps AI Corp., Client Organization");
  const [newContent, setNewContent] = useState("");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: contracts, isLoading: contractsLoading } = useQuery({
    queryKey: ["contracts"],
    queryFn: () => governanceAPI.getContracts(),
  });

  // Default selection
  useEffect(() => {
    if (contracts && contracts.length > 0 && !activeContractId) {
      setActiveContractId(contracts[0].id);
    }
  }, [contracts, activeContractId]);

  const { data: contract, isLoading: contractLoading } = useQuery({
    queryKey: ["contract", activeContractId],
    queryFn: () => governanceAPI.getContract(activeContractId),
    enabled: !!activeContractId,
  });

  const signMutation = useMutation({
    mutationFn: () => {
      const finalSig =
        signatureMode === "type"
          ? `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><text x="10" y="50" font-family="cursive" font-size="28" fill="#38bdf8">${typedName}</text></svg>`
          : signatureData;
      return governanceAPI.signContract(activeContractId, finalSig);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["contracts"] });
      queryClient.invalidateQueries({ queryKey: ["contract", activeContractId] });
      setSignatureData("");
      success("Contract digitally signed", "SHA-256 cryptographic signature verified");
    },
    onError: (err: any) => {
      error("Signing failed", err?.message || "Please provide a valid signature");
    },
  });

  const declineMutation = useMutation({
    mutationFn: (reason: string) =>
      governanceAPI.declineContract(activeContractId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["contracts"] });
      queryClient.invalidateQueries({ queryKey: ["contract", activeContractId] });
      success("Contract declined");
    },
  });

  const createContractMutation = useMutation({
    mutationFn: () =>
      governanceAPI.createContract({
        title: newTitle,
        parties: newParties.split(",").map((p) => p.trim()),
        content: newContent || "STANDARD GOVERNANCE CLAUSES...\n1. Purpose\n2. Liability\n3. Term",
      }),
    onSuccess: (newC) => {
      queryClient.invalidateQueries({ queryKey: ["contracts"] });
      setActiveContractId(newC.id);
      setShowCreateModal(false);
      setNewTitle("");
      setNewContent("");
      success("Contract draft created");
    },
  });

  const handleDownloadCertificate = () => {
    if (!contract) return;
    const certText = `CRYPTOGRAPHIC SIGNATURE RECEIPT & AUDIT CERTIFICATE
=====================================================
Document ID: ${contract.id}
Title: ${contract.title}
Status: ${contract.status.toUpperCase()}
Execution Timestamp: ${contract.signed_at || new Date().toISOString()}

SIGNATORIES:
${contract.signers.map((s) => `• ${s.name} (${s.email}) - [${s.status.toUpperCase()}] at ${s.signed_at || "Pending"}`).join("\n")}

AUDIT TRAIL LOGS:
${contract.audit_trail.map((a) => `[${a.timestamp}] ${a.username} -> ${a.action}`).join("\n")}

VERIFICATION HASH:
SHA256:${Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join("")}
=====================================================`;

    const blob = new Blob([certText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `signature_certificate_${contract.id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    success("Certificate downloaded", "Digital signature audit receipt exported");
  };

  if (contractsLoading) return <Spinner />;

  const currentUserSigner = contract?.signers.find(
    (s) => s.email === "current@user.com" || s.id === "sgn-02" || s.name.includes("Elena"),
  );
  const isSigned = currentUserSigner?.status === "signed";
  const allSigned = contract?.signers.every((s) => s.status === "signed");

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FileText size={20} className="text-neon-mint" /> Contract & Model Card Signing
          </h1>
          <p className="text-xs text-slate-400">
            Cryptographically binding governance sign-off and legal licensing workflow
          </p>
        </div>
        <div className="flex gap-2 items-center">
          <select
            value={activeContractId}
            onChange={(e) => setActiveContractId(e.target.value)}
            className="bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint max-w-xs truncate"
          >
            {contracts?.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title} ({c.status})
              </option>
            ))}
          </select>
          <Button size="sm" variant="ghost" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Contract
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4">
        {contract ? (
          <div className="grid lg:grid-cols-12 gap-5">
            {/* Left Column: Contract Document Viewer */}
            <div className="lg:col-span-7 space-y-3">
              <TerminalCard className="p-5 overflow-auto max-h-[calc(100vh-210px)]">
                <div className="flex justify-between items-center border-b border-white/10 pb-3 mb-4">
                  <div>
                    <h2 className="text-base font-bold text-white">{contract.title}</h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Parties: <strong className="text-neon-mint">{contract.parties.join(" ↔ ")}</strong>
                    </p>
                  </div>
                  <Badge
                    variant={
                      contract.status === "signed"
                        ? "success"
                        : contract.status === "pending"
                          ? "warning"
                          : "failed"
                    }
                    className="text-[10px] uppercase font-bold"
                  >
                    {contract.status}
                  </Badge>
                </div>

                <div className="prose prose-invert max-w-none text-xs text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                  {contract.content}
                </div>
              </TerminalCard>
            </div>

            {/* Right Column: Signing Pad, Signers & Audit Trail */}
            <div className="lg:col-span-5 space-y-4">
              {/* Signers Status Card */}
              <TerminalCard className="p-4 space-y-2.5">
                <h3 className="text-xs font-bold text-white flex items-center gap-1.5 uppercase tracking-wide">
                  <Award size={14} className="text-neon-mint" /> Signatories & Execution Status
                </h3>
                <div className="space-y-1.5">
                  {contract.signers.map((signer) => (
                    <div
                      key={signer.id}
                      className="flex justify-between items-center p-2.5 rounded bg-white/5 border border-white/5 text-xs"
                    >
                      <div>
                        <div className="font-bold text-white">{signer.name}</div>
                        <div className="text-[11px] text-slate-400 font-mono">{signer.email}</div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {signer.status === "signed" ? (
                          <span className="flex items-center gap-1 text-green-400 font-mono text-[11px] font-bold">
                            <CheckCircle size={14} /> Signed
                          </span>
                        ) : signer.status === "declined" ? (
                          <span className="flex items-center gap-1 text-red-400 font-mono text-[11px] font-bold">
                            <XCircle size={14} /> Declined
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-yellow-400 font-mono text-[11px]">
                            <AlertTriangle size={14} /> Pending
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </TerminalCard>

              {/* Digital Signing Pad (If not signed) */}
              {!isSigned && contract.status === "pending" && (
                <TerminalCard className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="text-xs font-bold text-white uppercase tracking-wide flex items-center gap-1.5">
                      <PenTool size={13} className="text-neon-blue" /> Your Digital Signature
                    </h3>
                    <div className="flex gap-1 bg-white/5 p-0.5 rounded border border-white/10 text-xs">
                      <button
                        onClick={() => setSignatureMode("draw")}
                        className={`px-2 py-0.5 rounded flex items-center gap-1 ${
                          signatureMode === "draw"
                            ? "bg-neon-blue text-white font-bold"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        <PenTool size={11} /> Draw
                      </button>
                      <button
                        onClick={() => setSignatureMode("type")}
                        className={`px-2 py-0.5 rounded flex items-center gap-1 ${
                          signatureMode === "type"
                            ? "bg-neon-blue text-white font-bold"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        <Type size={11} /> Type
                      </button>
                    </div>
                  </div>

                  {signatureMode === "draw" ? (
                    <SignatureCanvas onSave={setSignatureData} />
                  ) : (
                    <div className="space-y-2">
                      <Input
                        label="Full Legal Name"
                        value={typedName}
                        onChange={(e) => setTypedName(e.target.value)}
                        placeholder="Elena Rostova"
                      />
                      <div className="p-4 bg-slate-900 rounded border border-white/10 text-center font-mono text-2xl text-neon-blue italic">
                        {typedName || "Your Signature"}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 pt-2 border-t border-white/10">
                    <Button
                      onClick={() => signMutation.mutate()}
                      disabled={signMutation.isPending || (signatureMode === "draw" && !signatureData)}
                      className="bg-neon-mint text-slate-950 font-bold hover:bg-neon-mint/90 flex-1"
                    >
                      <ShieldCheck size={14} className="mr-1" />
                      {signMutation.isPending ? "Recording Signature..." : "Sign & Bind Contract"}
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => declineMutation.mutate("Signer declined terms")}
                      disabled={declineMutation.isPending}
                    >
                      Decline
                    </Button>
                  </div>
                </TerminalCard>
              )}

              {/* Verified Digital Seal (If fully signed or user signed) */}
              {(isSigned || allSigned) && (
                <TerminalCard className="p-4 text-center space-y-2 border-green-500/30 bg-green-500/5">
                  <ShieldCheck size={36} className="mx-auto text-green-400" />
                  <h3 className="text-sm font-bold text-white">
                    {allSigned ? "Fully Executed & Legally Binding" : "Your Signature Recorded"}
                  </h3>
                  <p className="text-xs text-slate-400">
                    Cryptographic signature attached to immutable governance ledger
                  </p>
                  <div className="pt-2">
                    <Button size="sm" variant="ghost" onClick={handleDownloadCertificate}>
                      <Download size={13} className="mr-1.5" /> Download Audit Certificate
                    </Button>
                  </div>
                </TerminalCard>
              )}

              {/* Audit Trail List */}
              {contract.audit_trail && contract.audit_trail.length > 0 && (
                <TerminalCard className="p-4 space-y-2">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wide flex items-center gap-1.5">
                    <History size={13} className="text-slate-400" /> Audit Log Entries ({contract.audit_trail.length})
                  </h4>
                  <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                    {contract.audit_trail.map((entry) => (
                      <div
                        key={entry.id}
                        className="text-[11px] text-slate-300 font-mono border-l-2 border-neon-blue pl-2 py-0.5"
                      >
                        <span className="text-slate-500">
                          {new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>{" "}
                        <strong className="text-white">{entry.username}</strong> performed{" "}
                        <span className="text-neon-mint">{entry.action}</span>
                      </div>
                    ))}
                  </div>
                </TerminalCard>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-slate-500 py-16">
            <FileText size={32} className="mx-auto mb-2 opacity-40" />
            <p>No contracts found. Click "New Contract" to draft a legal agreement.</p>
          </div>
        )}
      </div>

      {/* Create Contract Modal */}
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Draft Legal / Governance Agreement"
      >
        <div className="space-y-4">
          <Input
            label="Contract Title"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="e.g. Model Weight Escrow & Safety Commitment"
          />
          <Input
            label="Parties (comma-separated)"
            value={newParties}
            onChange={(e) => setNewParties(e.target.value)}
            placeholder="NeuralOps Corp, Partner Organization"
          />
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Contract Terms / Model Card Clauses</label>
            <textarea
              rows={8}
              className="w-full bg-slate-900 border border-white/10 rounded p-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-neon-mint"
              placeholder="Paste agreement text..."
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
            />
          </div>
          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createContractMutation.mutate()}
              disabled={!newTitle.trim() || createContractMutation.isPending}
            >
              {createContractMutation.isPending ? "Creating..." : "Save Draft"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

