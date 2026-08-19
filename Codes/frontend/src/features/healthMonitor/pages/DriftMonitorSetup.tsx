// src/features/healthMonitor/pages/DriftMonitorSetup.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { healthMonitorAPI } from "../services/api";
import { DriftChart } from "../components/DriftChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import {
  Activity,
  Plus,
  Trash2,
  Play,
  Save,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Settings2,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { DriftMonitor } from "../types";

export default function DriftMonitorSetup({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedMonitorId, setSelectedMonitorId] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  // Edit form state (prevents mutation on every keystroke)
  const [editName, setEditName] = useState("");
  const [editThreshold, setEditThreshold] = useState(0.1);
  const [editMethod, setEditMethod] = useState<"psi" | "ks" | "emd" | "js">("psi");
  const [editEnabled, setEditEnabled] = useState(true);

  // New monitor form state
  const [newForm, setNewForm] = useState({
    name: "",
    model_id: "model-svd-transformer-v2",
    model_name: "SVD Linear Attention Transformer v2",
    feature_name: "attention_entropy",
    method: "psi" as const,
    alert_threshold: 0.15,
    enabled: true,
  });

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: monitors, isLoading: monitorsLoading } = useQuery({
    queryKey: ["drift-monitors"],
    queryFn: () => healthMonitorAPI.getDriftMonitors(),
  });

  // Auto-select first monitor
  useEffect(() => {
    if (monitors && monitors.length > 0 && !selectedMonitorId) {
      setSelectedMonitorId(monitors[0].id);
    }
  }, [monitors, selectedMonitorId]);

  const selectedMonitor = monitors?.find((m) => m.id === selectedMonitorId);

  // Sync local edit state when selected monitor changes
  useEffect(() => {
    if (selectedMonitor) {
      setEditName(selectedMonitor.name);
      setEditThreshold(selectedMonitor.alert_threshold);
      setEditMethod(selectedMonitor.method);
      setEditEnabled(selectedMonitor.enabled);
    }
  }, [selectedMonitor?.id]);

  const { data: checks, isLoading: checksLoading } = useQuery({
    queryKey: ["drift-checks", selectedMonitorId],
    queryFn: () => healthMonitorAPI.getDriftChecks(selectedMonitorId!, 30),
    enabled: !!selectedMonitorId,
  });

  const createMonitor = useMutation({
    mutationFn: () => healthMonitorAPI.createDriftMonitor(newForm),
    onSuccess: (newMon) => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setSelectedMonitorId(newMon.id);
      setIsCreating(false);
      setNewForm({
        name: "",
        model_id: "model-svd-transformer-v2",
        model_name: "SVD Linear Attention Transformer v2",
        feature_name: "attention_entropy",
        method: "psi",
        alert_threshold: 0.15,
        enabled: true,
      });
      success("Monitor created", "Proactive detector activated for model feature");
    },
    onError: (err: any) => {
      error("Failed to create monitor", err?.message || "Unknown error");
    },
  });

  const updateMonitor = useMutation({
    mutationFn: () =>
      healthMonitorAPI.updateDriftMonitor(selectedMonitorId!, {
        name: editName,
        alert_threshold: editThreshold,
        method: editMethod,
        enabled: editEnabled,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      success("Settings saved", "Updated drift detection parameters");
    },
  });

  const deleteMonitor = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.deleteDriftMonitor(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setSelectedMonitorId(null);
      success("Monitor removed");
    },
  });

  const runCheck = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.runDriftCheck(id),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      queryClient.invalidateQueries({
        queryKey: ["drift-checks", selectedMonitorId],
      });
      success("Drift check complete", `Observed score: ${res.drift_score} (${res.status.toUpperCase()})`);
    },
  });

  if (monitorsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-yellow-400" /> Drift Monitor Setup & Calibration
          </h1>
          <p className="text-xs text-slate-400">
            Calibrate statistical divergence thresholds, baseline windows & automated trigger checks
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={() => {
              setIsCreating(true);
              setSelectedMonitorId(null);
            }}
          >
            <Plus size={14} className="mr-1" /> New Monitor
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
        <div className="grid lg:grid-cols-12 gap-4">
          {/* Left Column: Monitors list */}
          <div className="lg:col-span-4 space-y-2.5 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
            {monitors?.map((m) => {
              const isSelected = selectedMonitorId === m.id;
              const statusVariant =
                m.status === "normal"
                  ? "success"
                  : m.status === "warning"
                    ? "warning"
                    : "failed";

              return (
                <div
                  key={m.id}
                  className={`p-3.5 rounded-lg cursor-pointer border transition-all ${
                    isSelected
                      ? "border-neon-blue bg-neon-blue/10 shadow-lg"
                      : "border-white/10 hover:border-white/20 hover:bg-white/5"
                  }`}
                  onClick={() => {
                    setSelectedMonitorId(m.id);
                    setIsCreating(false);
                  }}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-bold text-white text-xs sm:text-sm truncate max-w-[180px]">
                      {m.name}
                    </span>
                    <Badge variant={statusVariant} className="text-[10px] py-0 uppercase font-bold">
                      {m.status}
                    </Badge>
                  </div>
                  <div className="text-xs text-slate-400 mt-1 font-mono">
                    Feature: <span className="text-neon-mint">{m.feature_name}</span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1 flex justify-between">
                    <span>Method: {m.method.toUpperCase()}</span>
                    <span>Threshold: {m.alert_threshold}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right Column: Edit Monitor / Create New */}
          <div className="lg:col-span-8 space-y-4">
            {selectedMonitor ? (
              <>
                {/* Configuration and Actions Bar */}
                <TerminalCard className="p-4 space-y-3">
                  <div className="flex justify-between items-center flex-wrap gap-2">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        {selectedMonitor.name}
                        <Badge
                          variant={
                            selectedMonitor.status === "normal"
                              ? "success"
                              : selectedMonitor.status === "warning"
                                ? "warning"
                                : "failed"
                          }
                          className="text-[10px]"
                        >
                          {selectedMonitor.status.toUpperCase()}
                        </Badge>
                      </h3>
                      <p className="text-xs text-slate-400 font-mono mt-0.5">
                        Feature: <span className="text-neon-mint">{selectedMonitor.feature_name}</span> · Model:{" "}
                        {selectedMonitor.model_name}
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        onClick={() => runCheck.mutate(selectedMonitor.id)}
                        disabled={runCheck.isPending}
                        className="bg-neon-blue text-white"
                      >
                        <Play size={13} className="mr-1" />
                        {runCheck.isPending ? "Calculating..." : "Run Test Now"}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => deleteMonitor.mutate(selectedMonitor.id)}
                        className="hover:text-red-400"
                        title="Delete Monitor"
                      >
                        <Trash2 size={14} />
                      </Button>
                    </div>
                  </div>

                  {/* Calibration Input Form */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-white/5">
                    <Input
                      label="Monitor Name"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="text-xs"
                    />

                    <div className="space-y-1">
                      <label className="text-xs font-semibold text-slate-300">Method</label>
                      <select
                        className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
                        value={editMethod}
                        onChange={(e) => setEditMethod(e.target.value as any)}
                      >
                        <option value="psi">Population Stability Index (PSI)</option>
                        <option value="ks">Kolmogorov-Smirnov (KS-Test)</option>
                        <option value="emd">Wasserstein (EMD)</option>
                        <option value="js">Jensen-Shannon (JS)</option>
                      </select>
                    </div>

                    <Input
                      label="Alert Threshold"
                      type="number"
                      step="0.01"
                      value={editThreshold}
                      onChange={(e) => setEditThreshold(parseFloat(e.target.value))}
                      className="text-xs font-mono"
                    />
                  </div>

                  <div className="flex justify-end pt-1">
                    <Button
                      size="sm"
                      onClick={() => updateMonitor.mutate()}
                      disabled={updateMonitor.isPending}
                      className="bg-neon-mint text-slate-950 font-bold hover:bg-neon-mint/90"
                    >
                      <Save size={13} className="mr-1" /> Save Calibration
                    </Button>
                  </div>
                </TerminalCard>

                {/* Drift Timeline and Distribution */}
                <TerminalCard className="p-4 space-y-2">
                  <div className="flex justify-between items-center">
                    <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                      <Activity size={15} className="text-neon-mint" /> Drift Timeline & Histogram
                    </h3>
                    <span className="text-xs text-slate-400 font-mono">
                      Last check: {new Date(selectedMonitor.last_check).toLocaleTimeString()}
                    </span>
                  </div>

                  {checksLoading ? (
                    <div className="py-12 text-center">
                      <Spinner />
                    </div>
                  ) : checks && checks.length > 0 ? (
                    <DriftChart checks={checks} threshold={selectedMonitor.alert_threshold} />
                  ) : (
                    <div className="text-center text-slate-500 py-12 text-xs">
                      No drift history yet. Click "Run Test Now" to record a data point.
                    </div>
                  )}
                </TerminalCard>
              </>
            ) : isCreating ? (
              <TerminalCard className="p-5 space-y-4">
                <div className="flex justify-between items-center border-b border-white/10 pb-2">
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Plus size={16} className="text-neon-mint" /> Configure New Drift Detector
                  </h3>
                  <Button size="sm" variant="ghost" onClick={() => setIsCreating(false)}>
                    ✕
                  </Button>
                </div>

                <Input
                  label="Monitor Name"
                  value={newForm.name}
                  onChange={(e) => setNewForm({ ...newForm, name: e.target.value })}
                  placeholder="e.g. SVD Attention Norm Shift"
                  className="text-xs"
                />

                <div className="grid grid-cols-2 gap-3">
                  <Input
                    label="Model Name"
                    value={newForm.model_name}
                    onChange={(e) => setNewForm({ ...newForm, model_name: e.target.value })}
                    placeholder="SVD Linear Attention Transformer v2"
                    className="text-xs"
                  />
                  <Input
                    label="Target Feature Name"
                    value={newForm.feature_name}
                    onChange={(e) => setNewForm({ ...newForm, feature_name: e.target.value })}
                    placeholder="attention_entropy"
                    className="text-xs"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-300">Detection Method</label>
                    <select
                      className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint"
                      value={newForm.method}
                      onChange={(e) => setNewForm({ ...newForm, method: e.target.value as any })}
                    >
                      <option value="psi">Population Stability Index (PSI)</option>
                      <option value="ks">Kolmogorov-Smirnov (KS-Test)</option>
                      <option value="emd">Wasserstein (Earth Mover's Distance)</option>
                      <option value="js">Jensen-Shannon Divergence</option>
                    </select>
                  </div>

                  <Input
                    label="Alert Threshold"
                    type="number"
                    step="0.01"
                    value={newForm.alert_threshold}
                    onChange={(e) =>
                      setNewForm({ ...newForm, alert_threshold: parseFloat(e.target.value) })
                    }
                    className="text-xs font-mono"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
                  <Button variant="ghost" onClick={() => setIsCreating(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={() => createMonitor.mutate()}
                    disabled={!newForm.name.trim() || !newForm.feature_name.trim() || createMonitor.isPending}
                    className="bg-neon-mint text-slate-950 font-bold hover:bg-neon-mint/90"
                  >
                    {createMonitor.isPending ? "Creating..." : "Create Monitor"}
                  </Button>
                </div>
              </TerminalCard>
            ) : (
              <div className="text-center text-slate-500 py-20">
                <Activity size={32} className="mx-auto mb-2 opacity-40" />
                <p>Select a monitor or click "New Monitor" to configure proactive drift detection.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

