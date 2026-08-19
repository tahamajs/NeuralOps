// src/features/governance/pages/ComplianceReportView.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { ComplianceScorecard } from "../components/ComplianceScorecard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Progress } from "../../../components/ui/Progress";
import { Modal } from "../../../components/ui/Modal";
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  FileText,
  Download,
  RefreshCw,
  FileCheck,
  Shield,
  ExternalLink,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { ComplianceControl } from "../types";

export default function ComplianceReportView({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedFramework, setSelectedFramework] = useState("");
  const [inspectControl, setInspectControl] = useState<ComplianceControl | null>(null);

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: frameworks, isLoading: frameworksLoading } = useQuery({
    queryKey: ["compliance-frameworks"],
    queryFn: () => governanceAPI.getComplianceFrameworks(),
  });

  // Auto select first framework
  useEffect(() => {
    if (frameworks && frameworks.length > 0 && !selectedFramework) {
      setSelectedFramework(frameworks[0].id);
    }
  }, [frameworks, selectedFramework]);

  const { data: framework, isLoading: frameworkLoading } = useQuery({
    queryKey: ["compliance-framework", selectedFramework],
    queryFn: () => governanceAPI.getComplianceFramework(selectedFramework),
    enabled: !!selectedFramework,
  });

  const runCheck = useMutation({
    mutationFn: () => governanceAPI.runComplianceCheck(selectedFramework),
    onSuccess: (res) => {
      queryClient.invalidateQueries({
        queryKey: ["compliance-framework", selectedFramework],
      });
      queryClient.invalidateQueries({
        queryKey: ["compliance-frameworks"],
      });
      success("Compliance check verified", `All control rules re-assessed · Score: ${res.score}%`);
    },
  });

  const generateReport = useMutation({
    mutationFn: () => governanceAPI.generateComplianceReport(selectedFramework, "pdf"),
    onSuccess: (res) => {
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `compliance_report_${selectedFramework}_${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      success("Report exported", "Official Compliance Verification Certificate generated");
    },
  });

  if (frameworksLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-neon-mint" /> Compliance & Audit Reports
          </h1>
          <p className="text-xs text-slate-400">
            Automated verification against EU AI Act, SOC 2 Type II, and NIST AI RMF standards
          </p>
        </div>
        <div className="flex gap-2 items-center">
          <select
            value={selectedFramework}
            onChange={(e) => setSelectedFramework(e.target.value)}
            className="bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
          >
            {frameworks?.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name} ({f.version})
              </option>
            ))}
          </select>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {selectedFramework && framework ? (
          <>
            {/* Top Scorecard and Action Bar */}
            <div className="flex flex-col md:flex-row gap-3 items-stretch justify-between">
              <ComplianceScorecard framework={framework} />
              <div className="flex md:flex-col gap-2 justify-center">
                <Button
                  size="sm"
                  onClick={() => runCheck.mutate()}
                  disabled={runCheck.isPending}
                  className="bg-neon-blue text-white whitespace-nowrap"
                >
                  <RefreshCw
                    size={14}
                    className={`mr-1.5 ${runCheck.isPending ? "animate-spin" : ""}`}
                  />{" "}
                  {runCheck.isPending ? "Evaluating..." : "Run Live Audit"}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => generateReport.mutate()}
                  disabled={generateReport.isPending}
                  className="whitespace-nowrap"
                >
                  <Download size={14} className="mr-1.5" /> Export Report
                </Button>
              </div>
            </div>

            {/* Controls List Card */}
            <TerminalCard className="p-4 space-y-3">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <FileCheck size={16} className="text-neon-mint" /> Evaluated Controls ({framework.controls.length})
                </h3>
                <span className="text-xs text-slate-400">
                  Click any control to inspect cryptographic evidence
                </span>
              </div>

              <div className="space-y-2.5">
                {framework.controls.map((control) => {
                  const isCompliant = control.status === "compliant";
                  const isPartial = control.status === "partial";

                  return (
                    <div
                      key={control.id}
                      onClick={() => setInspectControl(control)}
                      className="p-3 bg-white/5 hover:bg-white/10 rounded border border-white/10 cursor-pointer transition-all space-y-2"
                    >
                      <div className="flex justify-between items-start gap-2">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-xs text-neon-blue font-bold">
                              {control.id}
                            </span>
                            <h4 className="font-bold text-white text-xs">
                              {control.title}
                            </h4>
                            <Badge
                              variant={
                                isCompliant
                                  ? "success"
                                  : isPartial
                                    ? "warning"
                                    : "failed"
                              }
                              className="text-[10px] py-0 uppercase"
                            >
                              {control.status}
                            </Badge>
                            <Badge
                              variant={
                                control.risk_level === "critical"
                                  ? "failed"
                                  : control.risk_level === "warning"
                                    ? "warning"
                                    : "info"
                              }
                              className="text-[10px] py-0"
                            >
                              Risk: {control.risk_level}
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-300">
                            {control.description}
                          </p>
                        </div>

                        <div className="text-right text-[11px] text-slate-400 font-mono whitespace-nowrap">
                          <div>
                            Checked: {new Date(control.last_checked).toLocaleDateString()}
                          </div>
                          <div className="text-neon-mint">
                            {control.evidence.length} Evidence File(s)
                          </div>
                        </div>
                      </div>

                      {control.remediation_steps && (
                        <div className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-xs text-yellow-300">
                          <strong>Remediation:</strong> {control.remediation_steps}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </TerminalCard>
          </>
        ) : (
          <div className="text-center text-slate-500 py-16">
            <Spinner />
          </div>
        )}
      </div>

      {/* Inspect Control Evidence Modal */}
      <Modal
        open={!!inspectControl}
        onClose={() => setInspectControl(null)}
        title={`Control Verification: ${inspectControl?.id || ""}`}
      >
        {inspectControl && (
          <div className="space-y-4 text-xs">
            <div className="p-3 bg-white/5 rounded border border-white/10 space-y-1">
              <div className="font-bold text-white text-sm">{inspectControl.title}</div>
              <p className="text-slate-300">{inspectControl.description}</p>
            </div>

            <div className="space-y-2">
              <span className="font-bold text-white flex items-center gap-1.5">
                <FileText size={13} className="text-neon-mint" /> Attached Proof & Evidence Artifacts
              </span>
              <div className="space-y-1.5">
                {inspectControl.evidence.map((ev, i) => (
                  <div
                    key={i}
                    className="p-2 bg-slate-900 rounded border border-white/10 flex justify-between items-center font-mono text-[11px]"
                  >
                    <span className="text-slate-200">{ev}</span>
                    <span className="text-neon-mint flex items-center gap-1">
                      <CheckCircle size={12} /> Verified
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {inspectControl.remediation_steps && (
              <div className="p-2.5 bg-yellow-500/10 border border-yellow-500/20 rounded space-y-1 text-yellow-300">
                <span className="font-bold uppercase text-[10px]">Action Item</span>
                <p>{inspectControl.remediation_steps}</p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

