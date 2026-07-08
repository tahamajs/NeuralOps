// src/features/governance/pages/ContractSigningUI.tsx
import { useState, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { SignatureCanvas } from "../components/SignatureCanvas";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import {
  FileText,
  PenTool,
  History,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ContractSigningUI({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { contractId } = useParams();
  const navigate = useNavigate();
  const [signatureData, setSignatureData] = useState("");
  const [signaturePadRef, setSignaturePadRef] = useState<any>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: contract, isLoading } = useQuery({
    queryKey: ["contract", contractId],
    queryFn: () => governanceAPI.getContract(contractId!),
    enabled: !!contractId,
  });

  const signMutation = useMutation({
    mutationFn: () => governanceAPI.signContract(contractId!, signatureData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["contract", contractId] });
      success("Contract signed", "Your signature has been recorded");
      navigate("/contracts");
    },
  });

  const declineMutation = useMutation({
    mutationFn: (reason: string) =>
      governanceAPI.declineContract(contractId!, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["contract", contractId] });
      success("Contract declined");
      navigate("/contracts");
    },
  });

  if (isLoading) return <Spinner />;
  if (!contract)
    return <div className="p-4 text-red-400">Contract not found</div>;

  const currentUserSigner = contract.signers.find(
    (s) => s.email === "current@user.com",
  ); // Replace with actual auth
  const isSigned = currentUserSigner?.status === "signed";
  const allSigned = contract.signers.every((s) => s.status === "signed");

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FileText size={20} className="text-neon-mint" /> Contract Signing
          </h1>
          <p className="text-xs text-slate-400">
            {contract.title} · {contract.parties.join(" → ")}
          </p>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-2 gap-6">
          <TerminalCard className="p-4 overflow-auto">
            <div className="prose prose-invert max-w-none whitespace-pre-wrap">
              {contract.content}
            </div>
          </TerminalCard>
          <div className="space-y-4">
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">Signers</h3>
              <div className="space-y-2">
                {contract.signers.map((signer) => (
                  <div
                    key={signer.id}
                    className="flex justify-between items-center p-2 rounded bg-white/5"
                  >
                    <div>
                      <div className="font-medium text-white">
                        {signer.name}
                      </div>
                      <div className="text-xs text-slate-400">
                        {signer.email}
                      </div>
                    </div>
                    <div>
                      {signer.status === "signed" ? (
                        <CheckCircle size={16} className="text-green-400" />
                      ) : signer.status === "declined" ? (
                        <XCircle size={16} className="text-red-400" />
                      ) : (
                        <AlertTriangle size={16} className="text-yellow-400" />
                      )}
                      <span className="text-xs ml-1">{signer.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </TerminalCard>
            {!isSigned && !allSigned && (
              <TerminalCard className="p-4">
                <h3 className="text-sm font-bold text-white mb-2">
                  Your Signature
                </h3>
                <SignatureCanvas
                  ref={setSignaturePadRef}
                  onSave={setSignatureData}
                />
                <div className="flex gap-2 mt-4">
                  <Button
                    onClick={() => signMutation.mutate()}
                    disabled={!signatureData}
                  >
                    Sign Contract
                  </Button>
                  <Button
                    variant="danger"
                    onClick={() => declineMutation.mutate("User declined")}
                  >
                    Decline
                  </Button>
                </div>
              </TerminalCard>
            )}
            {allSigned && (
              <TerminalCard className="p-4 text-center">
                <CheckCircle
                  size={40}
                  className="mx-auto text-green-400 mb-2"
                />
                <h3 className="text-lg font-bold text-white">
                  All parties have signed
                </h3>
                <p className="text-slate-400">
                  This contract is now legally binding
                </p>
              </TerminalCard>
            )}
            {contract.audit_trail.length > 0 && (
              <Tabs defaultValue="audit">
                <TabList>
                  <Tab value="audit">Audit Trail</Tab>
                </TabList>
                <TabPanel value="audit">
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {contract.audit_trail.map((entry) => (
                      <div
                        key={entry.id}
                        className="text-xs text-slate-400 border-l-2 border-neon-blue pl-2 py-1"
                      >
                        {new Date(entry.timestamp).toLocaleString()} ·{" "}
                        {entry.username} {entry.action}
                      </div>
                    ))}
                  </div>
                </TabPanel>
              </Tabs>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
