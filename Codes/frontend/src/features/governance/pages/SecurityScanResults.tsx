// src/features/governance/pages/SecurityScanResults.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { VulnerabilityCard } from "../components/VulnerabilityCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Shield, AlertTriangle, CheckCircle, Play, Filter } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function SecurityScanResults({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedScan, setSelectedScan] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState("");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: scans, isLoading: scansLoading } = useQuery({
    queryKey: ["security-scans"],
    queryFn: () => governanceAPI.getSecurityScans(),
  });

  const { data: scan, isLoading: scanLoading } = useQuery({
    queryKey: ["security-scan", selectedScan],
    queryFn: () => governanceAPI.getScan(selectedScan!),
    enabled: !!selectedScan,
  });

  const triggerScan = useMutation({
    mutationFn: ({
      targetId,
      targetType,
    }: {
      targetId: string;
      targetType: string;
    }) => governanceAPI.triggerScan(targetId, targetType),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["security-scans"] }),
  });

  if (scansLoading) return <Spinner />;

  const filteredVulns =
    scan?.vulnerabilities.filter(
      (v) => !severityFilter || v.severity === severityFilter,
    ) || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-red-400" /> Security Scan Results
          </h1>
          <p className="text-xs text-slate-400">
            Vulnerability assessments and remediation
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={() =>
              triggerScan.mutate({ targetId: "model-123", targetType: "model" })
            }
          >
            <Play size={14} className="mr-1" /> New Scan
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-3 gap-4">
        <div className="space-y-3">
          {scans?.map((s) => (
            <TerminalCard
              key={s.id}
              className={`p-3 cursor-pointer ${selectedScan === s.id ? "border-neon-blue" : ""}`}
              onClick={() => setSelectedScan(s.id)}
            >
              <div className="flex justify-between">
                <div>
                  <div className="font-bold text-white">{s.target}</div>
                  <div className="text-xs text-slate-400">
                    {s.target_type} · {new Date(s.started_at).toLocaleString()}
                  </div>
                </div>
                <Badge
                  variant={
                    s.status === "completed"
                      ? "success"
                      : s.status === "in_progress"
                        ? "warning"
                        : "failed"
                  }
                >
                  {s.status}
                </Badge>
              </div>
              <div className="flex gap-2 mt-2 text-xs">
                <span className="text-red-400">C:{s.summary.critical}</span>
                <span className="text-orange-400">H:{s.summary.high}</span>
                <span className="text-yellow-400">M:{s.summary.medium}</span>
                <span className="text-blue-400">L:{s.summary.low}</span>
              </div>
            </TerminalCard>
          ))}
        </div>
        <div className="lg:col-span-2 space-y-4">
          {scan ? (
            <>
              <TerminalCard className="p-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-sm font-bold text-white">
                      Scan Summary
                    </h3>
                    <p className="text-xs text-slate-400">
                      Completed: {new Date(scan.completed_at!).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSeverityFilter("")}
                    >
                      All
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSeverityFilter("critical")}
                    >
                      Critical
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSeverityFilter("high")}
                    >
                      High
                    </Button>
                  </div>
                </div>
              </TerminalCard>
              <div className="space-y-3">
                {filteredVulns.map((v) => (
                  <VulnerabilityCard key={v.id} vulnerability={v} />
                ))}
              </div>
            </>
          ) : (
            <div className="text-center text-slate-500 py-12">
              Select a scan to view vulnerabilities
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
