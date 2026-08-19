// src/features/governance/pages/SecurityScanResults.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { VulnerabilityCard } from "../components/VulnerabilityCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Play,
  Filter,
  Download,
  Search,
  CheckCircle2,
  Lock,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { SecurityScan } from "../types";

export default function SecurityScanResults({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [showNewScanModal, setShowNewScanModal] = useState(false);

  // New Scan Form
  const [targetId, setTargetId] = useState("model-svd-transformer-v2.safetensors");
  const [targetType, setTargetType] = useState<"model" | "dataset" | "code" | "container">("model");
  const [scanProfile, setScanProfile] = useState("full_cve_and_injection");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: scans, isLoading: scansLoading } = useQuery({
    queryKey: ["security-scans"],
    queryFn: () => governanceAPI.getSecurityScans(),
  });

  // Auto select first scan
  useEffect(() => {
    if (scans && scans.length > 0 && !selectedScanId) {
      setSelectedScanId(scans[0].id);
    }
  }, [scans, selectedScanId]);

  const { data: scan, isLoading: scanLoading } = useQuery({
    queryKey: ["security-scan", selectedScanId],
    queryFn: () => governanceAPI.getScan(selectedScanId!),
    enabled: !!selectedScanId,
  });

  const triggerScan = useMutation({
    mutationFn: () =>
      governanceAPI.triggerScan(targetId, targetType),
    onSuccess: (newScan) => {
      queryClient.invalidateQueries({ queryKey: ["security-scans"] });
      setSelectedScanId(newScan.id);
      setShowNewScanModal(false);
      success("Security scan completed", `Assessed ${targetId} for CVE vulnerabilities and prompt leakage`);
    },
    onError: (err: any) => {
      error("Scan failed", err?.message || "Unknown scanner error");
    },
  });

  const handleExportFindings = () => {
    if (!scan) return;
    const jsonStr = JSON.stringify(scan, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `security_assessment_${scan.target}_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    success("Findings exported", "Security assessment JSON report downloaded");
  };

  if (scansLoading) return <Spinner />;

  const filteredVulns =
    scan?.vulnerabilities.filter((v) => {
      const matchSeverity = severityFilter === "all" || v.severity === severityFilter;
      const matchSearch =
        v.title.toLowerCase().includes(search.toLowerCase()) ||
        v.description.toLowerCase().includes(search.toLowerCase()) ||
        (v.cve_id && v.cve_id.toLowerCase().includes(search.toLowerCase())) ||
        v.affected_component.toLowerCase().includes(search.toLowerCase());
      return matchSeverity && matchSearch;
    }) || [];

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-red-400" /> Security Scan Results
          </h1>
          <p className="text-xs text-slate-400">
            Automated SAST, CVE vulnerability detection, weights tampering & prompt jailbreak assessment
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {scan && (
            <Button size="sm" variant="ghost" onClick={handleExportFindings}>
              <Download size={14} className="mr-1" /> Export Findings
            </Button>
          )}
          <Button
            size="sm"
            onClick={() => setShowNewScanModal(true)}
            className="bg-red-500 hover:bg-red-600 text-white font-bold"
          >
            <Play size={14} className="mr-1" /> Trigger Scan
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-12 gap-4">
        {/* Left Column: Scans History List */}
        <div className="lg:col-span-4 space-y-2.5 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
          {scans?.map((s) => {
            const isSelected = selectedScanId === s.id;
            return (
              <TerminalCard
                key={s.id}
                className={`p-3.5 cursor-pointer transition-all ${
                  isSelected
                    ? "border-neon-blue bg-neon-blue/10 shadow-lg"
                    : "hover:border-white/20 hover:bg-white/5"
                }`}
                onClick={() => setSelectedScanId(s.id)}
              >
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <div className="font-bold text-white text-xs truncate max-w-[200px]" title={s.target}>
                      {s.target}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono">
                      Type: <span className="text-neon-mint uppercase">{s.target_type}</span>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {new Date(s.started_at).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
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
                    className="text-[10px] uppercase py-0"
                  >
                    {s.status}
                  </Badge>
                </div>

                <div className="flex gap-2 mt-2 pt-2 border-t border-white/5 text-[11px] font-mono justify-between">
                  <span className="text-red-400 font-bold">Crit:{s.summary.critical}</span>
                  <span className="text-orange-400 font-bold">High:{s.summary.high}</span>
                  <span className="text-yellow-400">Med:{s.summary.medium}</span>
                  <span className="text-blue-400">Low:{s.summary.low}</span>
                </div>
              </TerminalCard>
            );
          })}
        </div>

        {/* Right Column: Selected Scan Findings Details */}
        <div className="lg:col-span-8 space-y-3.5">
          {scan ? (
            <>
              {/* Scan Summary Top Card */}
              <TerminalCard className="p-4 space-y-3">
                <div className="flex justify-between items-center flex-wrap gap-2">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                      <Lock size={15} className="text-neon-mint" /> Target: {scan.target}
                    </h3>
                    <p className="text-xs text-slate-400 font-mono mt-0.5">
                      Scanned at: {new Date(scan.started_at).toLocaleString()} · Duration: ~15s
                    </p>
                  </div>

                  {/* Filter Toolbar */}
                  <div className="flex items-center gap-2">
                    <div className="relative w-44">
                      <Input
                        placeholder="Search CVE / titles..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full text-xs pl-7"
                      />
                      <Search
                        size={12}
                        className="absolute left-2 top-2.5 text-slate-500 pointer-events-none"
                      />
                    </div>
                    <select
                      value={severityFilter}
                      onChange={(e) => setSeverityFilter(e.target.value)}
                      className="bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-neon-mint"
                    >
                      <option value="all">All Severities</option>
                      <option value="critical">Critical</option>
                      <option value="high">High</option>
                      <option value="medium">Medium</option>
                      <option value="low">Low</option>
                    </select>
                  </div>
                </div>

                {/* Summary Badges Bar */}
                <div className="grid grid-cols-4 gap-2 pt-2 border-t border-white/5 text-center text-xs">
                  <div className="p-2 bg-red-500/10 border border-red-500/20 rounded">
                    <span className="text-slate-400 text-[10px] block">CRITICAL</span>
                    <strong className="text-red-400 font-mono text-base">{scan.summary.critical}</strong>
                  </div>
                  <div className="p-2 bg-orange-500/10 border border-orange-500/20 rounded">
                    <span className="text-slate-400 text-[10px] block">HIGH</span>
                    <strong className="text-orange-400 font-mono text-base">{scan.summary.high}</strong>
                  </div>
                  <div className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded">
                    <span className="text-slate-400 text-[10px] block">MEDIUM</span>
                    <strong className="text-yellow-400 font-mono text-base">{scan.summary.medium}</strong>
                  </div>
                  <div className="p-2 bg-blue-500/10 border border-blue-500/20 rounded">
                    <span className="text-slate-400 text-[10px] block">LOW</span>
                    <strong className="text-blue-400 font-mono text-base">{scan.summary.low}</strong>
                  </div>
                </div>
              </TerminalCard>

              {/* Vulnerabilities List */}
              <div className="space-y-3">
                {filteredVulns.map((v) => (
                  <VulnerabilityCard key={v.id} vulnerability={v} />
                ))}

                {filteredVulns.length === 0 && (
                  <div className="text-center text-slate-500 py-12">
                    <CheckCircle2 size={32} className="mx-auto mb-2 opacity-40 text-green-400" />
                    <p>No vulnerabilities found matching current filters.</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="text-center text-slate-500 py-16">
              <Shield size={32} className="mx-auto mb-2 opacity-40" />
              <p>Select a scan to inspect security findings.</p>
            </div>
          )}
        </div>
      </div>

      {/* Trigger Security Scan Modal */}
      <Modal
        open={showNewScanModal}
        onClose={() => setShowNewScanModal(false)}
        title="Trigger Security Assessment Scan"
      >
        <div className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Target Asset Type</label>
            <select
              className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
              value={targetType}
              onChange={(e) => setTargetType(e.target.value as any)}
            >
              <option value="model">Model Checkpoint / Safetensors</option>
              <option value="dataset">Dataset / Feature Store Table</option>
              <option value="container">Docker / Triton Serving Image</option>
              <option value="code">Source Code / CUDA Kernels</option>
            </select>
          </div>

          <Input
            label="Target Identifier / Artifact Path"
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
            placeholder="model-svd-transformer-v2.safetensors"
          />

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Scan Profile</label>
            <select
              className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
              value={scanProfile}
              onChange={(e) => setScanProfile(e.target.value)}
            >
              <option value="full_cve_and_injection">Comprehensive (CVE, Prompt Jailbreak & Weight Safety)</option>
              <option value="sast_only">Static Analysis (SAST & CUDA Bounds Checking)</option>
              <option value="pii_leakage">Data Privacy & PII Leakage Check</option>
            </select>
          </div>

          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowNewScanModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => triggerScan.mutate()}
              disabled={!targetId.trim() || triggerScan.isPending}
              className="bg-red-500 hover:bg-red-600 text-white font-bold"
            >
              {triggerScan.isPending ? "Scanning Target..." : "Start Security Scan"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

