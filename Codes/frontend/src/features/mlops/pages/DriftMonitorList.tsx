// src/features/mlops/pages/DriftMonitorList.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsAPI } from "../services/api";
import { DriftChart } from "../components/DriftChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import {
  Activity,
  Plus,
  Trash2,
  Play,
  Download,
  Search,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Info,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { DriftMonitor } from "../types";

export default function DriftMonitorList({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedMonitor, setSelectedMonitor] = useState<DriftMonitor | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  // Create Form State
  const [newMonitorName, setNewMonitorName] = useState("");
  const [newGroupId, setNewGroupId] = useState("");
  const [newFeatureName, setNewFeatureName] = useState("");
  const [newMethod, setNewMethod] = useState<"ks" | "psi" | "emd" | "js">("ks");
  const [baselineWindow, setBaselineWindow] = useState("7d");
  const [threshold, setThreshold] = useState(0.05);

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: monitors, isLoading: monitorsLoading } = useQuery({
    queryKey: ["drift-monitors"],
    queryFn: () => mlopsAPI.getDriftMonitors(),
  });

  const { data: groups } = useQuery({
    queryKey: ["feature-groups"],
    queryFn: () => mlopsAPI.getFeatureGroups(),
  });

  // Set default selection
  useEffect(() => {
    if (monitors && monitors.length > 0 && !selectedMonitor) {
      setSelectedMonitor(monitors[0]);
    }
  }, [monitors, selectedMonitor]);

  // Set default group in create modal
  useEffect(() => {
    if (groups && groups.length > 0 && !newGroupId) {
      setNewGroupId(groups[0].id);
      if (groups[0].features.length > 0) {
        setNewFeatureName(groups[0].features[0].name);
      }
    }
  }, [groups, newGroupId]);

  const activeGroupId = newGroupId || groups?.[0]?.id || "";
  const currentSelectedGroup = groups?.find((g) => g.id === activeGroupId);

  const { data: checks, isLoading: checksLoading } = useQuery({
    queryKey: ["drift-checks", selectedMonitor?.id],
    queryFn: () => mlopsAPI.getDriftChecks(selectedMonitor?.id || "", 30),
    enabled: !!selectedMonitor?.id,
  });

  const createMonitor = useMutation({
    mutationFn: () =>
      mlopsAPI.createDriftMonitor({
        name: newMonitorName,
        feature_name: newFeatureName,
        feature_group_id: newGroupId,
        method: newMethod,
        baseline_window: baselineWindow,
        alert_threshold: threshold,
      }),
    onSuccess: (newMon) => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setSelectedMonitor(newMon);
      setShowCreateModal(false);
      setNewMonitorName("");
      success("Monitor created", `Continuous drift monitor configured for ${newFeatureName}`);
    },
    onError: (err: any) => {
      error("Failed to create monitor", err?.message || "Unknown error");
    },
  });

  const deleteMonitor = useMutation({
    mutationFn: (id: string) => mlopsAPI.deleteDriftMonitor(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setSelectedMonitor(null);
      success("Monitor removed");
    },
  });

  const runCheck = useMutation({
    mutationFn: (id: string) => mlopsAPI.runDriftCheck(id),
    onSuccess: (newCheck) => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      queryClient.invalidateQueries({
        queryKey: ["drift-checks", selectedMonitor?.id],
      });
      success(
        "Drift check complete",
        `P-Value: ${newCheck.p_value} · Divergence: ${newCheck.divergence} (${newCheck.status.toUpperCase()})`,
      );
    },
  });

  const handleExportDriftReport = () => {
    if (!checks || checks.length === 0 || !selectedMonitor) return;
    const csvHeader = "Timestamp,P_Value,Divergence,Status,SampleSize\n";
    const csvRows = checks
      .map(
        (c) =>
          `"${c.timestamp}",${c.p_value},${c.divergence},"${c.status}",${c.sample_size}`,
      )
      .join("\n");
    const blob = new Blob([csvHeader + csvRows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `drift_report_${selectedMonitor.name.toLowerCase().replace(/\s+/g, "_")}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    success("Report exported", "CSV drift report downloaded");
  };

  const filteredMonitors =
    monitors?.filter((m) => {
      const matchSearch =
        m.name.toLowerCase().includes(search.toLowerCase()) ||
        m.feature_name.toLowerCase().includes(search.toLowerCase());
      const matchStatus =
        statusFilter === "all" || m.status === statusFilter;
      return matchSearch && matchStatus;
    }) || [];

  if (monitorsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-yellow-400" /> Drift Monitors
          </h1>
          <p className="text-xs text-slate-400">
            Statistical data & concept drift detection using KS, PSI, Wasserstein, & JS metrics
          </p>
        </div>
        <div className="flex gap-2">
          {selectedMonitor && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleExportDriftReport}
              title="Export CSV History"
            >
              <Download size={14} className="mr-1" /> Export Report
            </Button>
          )}
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Monitor
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Grid */}
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-12 gap-4">
        {/* Left Column: Monitors List */}
        <div className="lg:col-span-5 space-y-3">
          {/* Filters */}
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                placeholder="Search monitors..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-8 text-xs"
              />
              <Search
                size={13}
                className="absolute left-2.5 top-3 text-slate-500 pointer-events-none"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-2 text-xs text-slate-300 focus:outline-none focus:border-neon-blue"
            >
              <option value="all">All Status</option>
              <option value="normal">Normal</option>
              <option value="warning">Warning</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          {/* Monitors list cards */}
          <div className="space-y-2 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
            {filteredMonitors.map((m) => {
              const isSelected = selectedMonitor?.id === m.id;
              const statusVariant =
                m.status === "normal"
                  ? "success"
                  : m.status === "warning"
                    ? "warning"
                    : "failed";

              return (
                <TerminalCard
                  key={m.id}
                  className={`p-3 cursor-pointer transition-all ${
                    isSelected
                      ? "border-neon-blue bg-neon-blue/10 shadow-lg"
                      : "hover:border-white/20 hover:bg-white/5"
                  }`}
                  onClick={() => setSelectedMonitor(m)}
                >
                  <div className="flex justify-between items-start">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <h3 className="font-bold text-white text-sm">
                          {m.name}
                        </h3>
                        <Badge variant={statusVariant} className="text-[10px] py-0">
                          {m.status}
                        </Badge>
                      </div>
                      <div className="text-xs text-slate-400 font-mono">
                        Feature: <span className="text-neon-mint">{m.feature_name}</span>
                      </div>
                      <div className="text-[11px] text-slate-500 flex gap-3">
                        <span>Method: <strong>{m.method?.toUpperCase() || "KS"}</strong></span>
                        <span>Threshold: <strong>{m.alert_threshold}</strong></span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          runCheck.mutate(m.id);
                        }}
                        disabled={runCheck.isPending}
                        title="Run check now"
                        className="text-xs h-7 px-2 hover:text-neon-mint"
                      >
                        <Play size={12} className="mr-1" /> Check
                      </Button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMonitor.mutate(m.id);
                        }}
                        className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                        title="Delete Monitor"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                </TerminalCard>
              );
            })}

            {filteredMonitors.length === 0 && (
              <div className="text-center text-slate-500 py-10 text-xs">
                No drift monitors match the search filter.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Selected Monitor Inspection & Chart */}
        <div className="lg:col-span-7 space-y-4">
          {selectedMonitor ? (
            <>
              {/* Monitor Details Top Banner */}
              <TerminalCard className="p-4 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-base font-bold text-white">
                        {selectedMonitor.name}
                      </h2>
                      <Badge
                        variant={
                          selectedMonitor.status === "normal"
                            ? "success"
                            : selectedMonitor.status === "warning"
                              ? "warning"
                              : "failed"
                        }
                      >
                        {selectedMonitor.status.toUpperCase()}
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Target feature:{" "}
                      <code className="font-mono text-neon-mint">
                        {selectedMonitor.feature_name}
                      </code>{" "}
                      · Baseline window: {selectedMonitor.baseline_window}
                    </p>
                  </div>

                  <Button
                    size="sm"
                    onClick={() => runCheck.mutate(selectedMonitor.id)}
                    disabled={runCheck.isPending}
                    className="bg-neon-blue text-white"
                  >
                    <Play size={13} className="mr-1" />{" "}
                    {runCheck.isPending ? "Calculating..." : "Run Drift Check"}
                  </Button>
                </div>

                {/* Metric Summary Chips */}
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5 text-center text-xs">
                  <div className="p-2 bg-white/5 rounded">
                    <span className="text-slate-400">Current P-Value</span>
                    <div className="font-bold text-sm text-neon-blue font-mono mt-0.5">
                      {selectedMonitor.current_p_value ?? 0.35}
                    </div>
                  </div>
                  <div className="p-2 bg-white/5 rounded">
                    <span className="text-slate-400">Divergence Score</span>
                    <div className="font-bold text-sm text-yellow-400 font-mono mt-0.5">
                      {selectedMonitor.current_divergence ?? 0.02}
                    </div>
                  </div>
                  <div className="p-2 bg-white/5 rounded">
                    <span className="text-slate-400">Alert Threshold</span>
                    <div className="font-bold text-sm text-red-400 font-mono mt-0.5">
                      {selectedMonitor.alert_threshold}
                    </div>
                  </div>
                </div>
              </TerminalCard>

              {/* Drift Interactive Chart */}
              <TerminalCard className="p-4 space-y-3">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Activity size={15} className="text-neon-mint" /> Drift History & Distribution
                  </h3>
                  <span className="text-xs text-slate-400">
                    Last check:{" "}
                    {new Date(selectedMonitor.last_checked_at).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>

                {checksLoading ? (
                  <div className="py-12 text-center">
                    <Spinner />
                  </div>
                ) : checks && checks.length > 0 ? (
                  <DriftChart
                    checks={checks}
                    threshold={selectedMonitor.alert_threshold}
                  />
                ) : (
                  <div className="text-center text-slate-500 py-12 text-xs">
                    No drift checks recorded yet. Click "Run Drift Check" to perform initial test.
                  </div>
                )}
              </TerminalCard>
            </>
          ) : (
            <div className="text-center text-slate-500 py-20">
              <Activity size={32} className="mx-auto mb-2 opacity-40" />
              <p>Select a monitor to view drift timeline and distribution comparison.</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Drift Monitor Modal */}
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Drift Monitor"
      >
        <div className="space-y-4">
          <Input
            label="Monitor Name"
            value={newMonitorName}
            onChange={(e) => setNewMonitorName(e.target.value)}
            placeholder="e.g. Prompt Token Entropy Monitor"
          />

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">
              Feature Group
            </label>
            <select
              className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
              value={activeGroupId}
              onChange={(e) => {
                setNewGroupId(e.target.value);
                const g = groups?.find((grp) => grp.id === e.target.value);
                if (g && g.features.length > 0) {
                  setNewFeatureName(g.features[0].name);
                }
              }}
            >
              {groups?.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.name} ({g.entity_key})
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">
              Target Feature
            </label>
            <select
              className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
              value={newFeatureName}
              onChange={(e) => setNewFeatureName(e.target.value)}
            >
              {currentSelectedGroup?.features.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} ({f.type})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">
                Drift Test Method
              </label>
              <select
                className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
                value={newMethod}
                onChange={(e) => setNewMethod(e.target.value as any)}
              >
                <option value="ks">Kolmogorov-Smirnov (KS-Test)</option>
                <option value="psi">Population Stability Index (PSI)</option>
                <option value="emd">Wasserstein (Earth Mover's Distance)</option>
                <option value="js">Jensen-Shannon Divergence</option>
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">
                Baseline Window
              </label>
              <select
                className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
                value={baselineWindow}
                onChange={(e) => setBaselineWindow(e.target.value)}
              >
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days (Recommended)</option>
                <option value="14d">Last 14 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
            </div>
          </div>

          <Input
            label="Alert Threshold (p-value / divergence cut-off)"
            type="number"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />

          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMonitor.mutate()}
              disabled={!newMonitorName.trim() || !newFeatureName || createMonitor.isPending}
            >
              {createMonitor.isPending ? "Creating..." : "Create Drift Monitor"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

