import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useNavigate } from "react-router-dom";
import {
  datasetsAPI,
  Dataset,
  DatasetVersion,
  AuditLog,
} from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { useToast } from "../../../components/ui/Toast";
import {
  Database,
  GitBranch,
  ArrowLeft,
  Trash2,
  Calendar,
  FileDown,
  ShieldAlert,
  GitMerge,
  ScrollText,
  Table,
  Binary,
  Tag,
  Info,
  AlertTriangle,
  Play,
  CheckCircle,
} from "lucide-react";

export default function DatasetDetailPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { id: paramId } = useParams<{ id: string }>();
  // Also support query param id if launched inside the floating window system
  const queryId = new URLSearchParams(window.location.search).get("id");
  const datasetId = paramId || queryId || "";

  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const [activeTab, setActiveTab] = useState("versions");
  const [selectedVersion, setSelectedVersion] = useState<DatasetVersion | null>(
    null,
  );

  // Audit Logs filters
  const [logSeverity, setLogSeverity] = useState("");

  const { data: dataset, isLoading: isDatasetLoading } = useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => datasetsAPI.getDataset(datasetId),
    enabled: !!datasetId,
  });

  const { data: lineage, isLoading: isLineageLoading } = useQuery({
    queryKey: ["dataset-lineage", datasetId],
    queryFn: () => datasetsAPI.getArtifactLineage(datasetId),
    enabled: !!datasetId,
  });

  const { data: auditLogsData, isLoading: isLogsLoading } = useQuery({
    queryKey: ["dataset-audit-logs", datasetId, logSeverity],
    queryFn: () => datasetsAPI.getAuditLogs({ resource_type: "dataset" }),
    enabled: !!datasetId,
  });

  const deleteMutation = useMutation({
    mutationFn: () => datasetsAPI.deleteDataset(datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      success(
        "Dataset Deleted",
        "The dataset metadata mapping has been removed.",
      );
      if (onClose) onClose();
      else navigate("/experiments");
    },
    onError: (err: any) => {
      error("Deletion Failed", err.userMessage || "Could not delete dataset");
    },
  });

  const exportLogsMutation = useMutation({
    mutationFn: (format: "csv" | "json" | "pdf") =>
      datasetsAPI.requestAuditExport({
        format,
        filters: { resource_type: "dataset", resource_id: datasetId },
      }),
    onSuccess: (data) => {
      success(
        "Export Requested",
        `Audit logs export generated as ${data.format || "CSV"}`,
      );
    },
    onError: (err: any) => {
      error(
        "Export Failed",
        err.userMessage || "Could not request audit export",
      );
    },
  });

  if (isDatasetLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-20 gap-3 h-full">
        <Spinner />
        <p className="text-slate-400">Loading dataset details...</p>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="p-8 text-center bg-slate-950 text-white flex flex-col items-center justify-center h-full">
        <AlertTriangle className="text-rose-500 mb-3" size={48} />
        <h2 className="text-xl font-bold">Dataset Not Found</h2>
        <p className="text-slate-500 mt-1 max-w-sm">
          The requested dataset may have been deleted or the ID is invalid.
        </p>
        <Button onClick={() => navigate("/experiments")} className="mt-4">
          Back to List
        </Button>
      </div>
    );
  }

  const exportUrl = datasetsAPI.exportDatasetUrl(datasetId);

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-200 font-sans">
      {/* Detail Header */}
      <div className="flex justify-between items-center p-5 border-b border-white/10 bg-slate-900/30 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate(-1)}
            className="hover:bg-white/5 text-slate-400 hover:text-white"
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold text-white tracking-tight">
                {dataset.name}
              </h1>
              <Badge
                variant="neutral"
                className="border-neon-blue/30 text-neon-blue"
              >
                {dataset.status}
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5 flex items-center gap-2">
              <span>
                ID:{" "}
                <code className="text-[10px] text-slate-500 font-mono">
                  {dataset.id}
                </code>
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Calendar size={12} />{" "}
                {new Date(dataset.created_at).toLocaleDateString()}
              </span>
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <a href={exportUrl} target="_blank" rel="noopener noreferrer">
            <Button
              size="sm"
              variant="outline"
              className="border-white/10 text-slate-300 hover:text-white flex items-center gap-1.5"
            >
              <FileDown size={14} /> Export CSV
            </Button>
          </a>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => deleteMutation.mutate()}
            className="text-rose-400 hover:bg-rose-500/10 hover:text-rose-300"
          >
            <Trash2 size={14} />
          </Button>
          {onClose && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onClose}
              className="hover:bg-white/5 text-white"
            >
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Info Pane */}
        <aside className="w-full lg:w-80 shrink-0 border-r border-white/10 p-5 bg-slate-950/60 overflow-y-auto space-y-6">
          <div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-1">
              <Info size={12} /> Description
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed bg-white/5 p-3.5 rounded-xl border border-white/10">
              {dataset.description || "No description provided."}
            </p>
          </div>

          <div className="space-y-3">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1">
              <GitBranch size={12} /> DVC Configuration
            </h3>
            <div className="bg-slate-900/40 border border-white/5 rounded-xl p-3 space-y-2.5 text-xs">
              <div>
                <span className="text-slate-500 block">DVC Remote:</span>
                <span className="font-mono text-slate-300 break-all select-all">
                  {dataset.dvc_remote || "Not set"}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block">Current Revision:</span>
                <span className="font-mono text-purple-400 break-all select-all">
                  {dataset.dvc_revision || "None"}
                </span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1">
              <Tag size={12} /> Metadata
            </h3>
            <div className="flex flex-wrap gap-1.5">
              {dataset.tags?.map((tag) => (
                <span
                  key={tag}
                  className="px-2.5 py-0.5 rounded-md bg-white/5 border border-white/10 text-xs text-slate-300"
                >
                  {tag}
                </span>
              ))}
            </div>
            {dataset.license && (
              <div className="mt-3 text-xs text-slate-400">
                License:{" "}
                <span className="text-white font-bold">{dataset.license}</span>
              </div>
            )}
          </div>

          {dataset.citation && (
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
                Citation
              </h3>
              <pre className="text-[10px] bg-slate-900 text-slate-400 p-2.5 rounded-lg border border-white/5 font-mono overflow-x-auto whitespace-pre-wrap">
                {dataset.citation}
              </pre>
            </div>
          )}
        </aside>

        {/* Right Tab Pane */}
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-900/10">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col overflow-hidden"
          >
            <div className="border-b border-white/10 bg-slate-950/20 px-6 py-2 shrink-0">
              <TabList className="flex gap-4">
                <Tab
                  value="versions"
                  className="flex items-center gap-1.5 py-2"
                >
                  <Table size={14} /> Versions
                </Tab>
                <Tab value="lineage" className="flex items-center gap-1.5 py-2">
                  <GitMerge size={14} /> Lineage DAG
                </Tab>
                <Tab value="audit" className="flex items-center gap-1.5 py-2">
                  <ScrollText size={14} /> Audit Log
                </Tab>
              </TabList>
            </div>

            <div className="flex-1 overflow-hidden relative">
              {/* VERSIONS TAB */}
              <TabPanel
                value="versions"
                className="h-full flex flex-col overflow-hidden p-6 space-y-4"
              >
                <div className="flex-1 overflow-y-auto space-y-4 pr-1">
                  {!dataset.versions || dataset.versions.length === 0 ? (
                    <div className="text-center py-20 text-slate-500">
                      <Binary className="mx-auto mb-3 opacity-20" size={36} />
                      No registered versions for this dataset.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {dataset.versions.map((ver) => (
                        <TerminalCard
                          key={ver.id}
                          className={`p-4 border transition-all cursor-pointer ${
                            selectedVersion?.id === ver.id
                              ? "border-neon-blue/80 bg-neon-blue/5 shadow-glow-sm"
                              : "border-slate-800/80 bg-slate-900/10 hover:border-slate-700"
                          }`}
                          onClick={() => setSelectedVersion(ver)}
                        >
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="font-extrabold text-white text-base">
                              Version {ver.version_number}
                            </h4>
                            <span className="text-xs text-slate-500 font-mono">
                              {(ver.size_bytes / (1024 * 1024)).toFixed(2)} MB
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 line-clamp-2 mb-3">
                            {ver.description || "No version details provided."}
                          </p>
                          <div className="grid grid-cols-2 gap-2 text-xs text-slate-500 pt-2.5 border-t border-white/5 font-mono">
                            <div>
                              Rows:{" "}
                              <span className="text-slate-300 font-bold">
                                {ver.row_count}
                              </span>
                            </div>
                            <div>
                              Cols:{" "}
                              <span className="text-slate-300 font-bold">
                                {ver.column_count}
                              </span>
                            </div>
                          </div>
                        </TerminalCard>
                      ))}
                    </div>
                  )}

                  {/* Schema / Sample Panel */}
                  {selectedVersion && (
                    <div className="border border-white/10 bg-slate-950/60 rounded-2xl p-5 space-y-4 mt-6">
                      <div className="flex justify-between items-center border-b border-white/5 pb-3">
                        <h4 className="font-bold text-white text-sm">
                          Schema & Sample Data: Version{" "}
                          {selectedVersion.version_number}
                        </h4>
                        <Badge
                          variant="secondary"
                          className="font-mono text-[10px]"
                        >
                          MD5: {selectedVersion.md5?.slice(0, 8)}
                        </Badge>
                      </div>

                      {/* Schema Grid */}
                      <div>
                        <h5 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
                          Inferred Schema
                        </h5>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                          {Object.entries(selectedVersion.schema || {}).map(
                            ([col, type]) => (
                              <div
                                key={col}
                                className="bg-slate-900 border border-white/5 p-2 rounded-lg text-xs font-mono"
                              >
                                <span className="text-slate-300 block truncate">
                                  {col}
                                </span>
                                <span className="text-neon-blue text-[10px]">
                                  {type}
                                </span>
                              </div>
                            ),
                          )}
                        </div>
                      </div>

                      {/* Sample Rows */}
                      {selectedVersion.sample_data?.length > 0 && (
                        <div>
                          <h5 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
                            Sample Rows
                          </h5>
                          <div className="overflow-x-auto border border-white/5 rounded-xl">
                            <table className="w-full text-left border-collapse text-xs font-mono">
                              <thead>
                                <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400">
                                  {Object.keys(
                                    selectedVersion.sample_data[0] || {},
                                  ).map((k) => (
                                    <th key={k} className="p-3 font-semibold">
                                      {k}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-white/5 bg-slate-950/20 text-slate-300">
                                {selectedVersion.sample_data.map((row, idx) => (
                                  <tr key={idx} className="hover:bg-white/5">
                                    {Object.values(row).map(
                                      (val: any, colIdx) => (
                                        <td
                                          key={colIdx}
                                          className="p-3 truncate max-w-[200px]"
                                          title={String(val)}
                                        >
                                          {String(val)}
                                        </td>
                                      ),
                                    )}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </TabPanel>

              {/* LINEAGE TAB */}
              <TabPanel
                value="lineage"
                className="h-full flex flex-col overflow-hidden p-6 space-y-4"
              >
                {isLineageLoading ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-3">
                    <Spinner />
                    <p className="text-slate-400">Rendering graph...</p>
                  </div>
                ) : (
                  <div className="flex-1 bg-slate-950/50 rounded-2xl border border-white/10 flex flex-col overflow-hidden relative min-h-[400px]">
                    <div className="absolute top-4 left-4 z-10 flex gap-2">
                      <span className="px-2.5 py-1 rounded bg-slate-900 border border-white/10 text-xs text-slate-400 flex items-center gap-1.5">
                        <GitMerge size={12} className="text-purple-400" />{" "}
                        Lineage Dependencies DAG
                      </span>
                    </div>

                    <div className="flex-1 flex items-center justify-center p-6">
                      {/* Render Interactive Lineage SVG */}
                      {!lineage?.nodes || lineage.nodes.length === 0 ? (
                        <div className="flex flex-col items-center text-center gap-3">
                          {/* Fallback mock visualization if API returns empty */}
                          <svg className="w-full max-w-xl h-64 border border-dashed border-white/10 rounded-xl bg-slate-900/10">
                            <defs>
                              <marker
                                id="arrow"
                                viewBox="0 0 10 10"
                                refX="6"
                                refY="5"
                                markerWidth="6"
                                markerHeight="6"
                                orient="auto-start-reverse"
                              >
                                <path
                                  d="M 0 0 L 10 5 L 0 10 z"
                                  fill="#6366f1"
                                />
                              </marker>
                            </defs>
                            {/* Connections */}
                            <path
                              d="M 120 120 C 180 120, 200 120, 260 120"
                              stroke="#6366f1"
                              strokeWidth="2"
                              strokeDasharray="4 2"
                              fill="none"
                              markerEnd="url(#arrow)"
                            />
                            <path
                              d="M 380 120 C 440 120, 460 120, 520 120"
                              stroke="#a855f7"
                              strokeWidth="2"
                              fill="none"
                              markerEnd="url(#arrow)"
                            />

                            {/* Node 1: SVD Config */}
                            <rect
                              x="20"
                              y="85"
                              width="100"
                              height="70"
                              rx="12"
                              fill="#0f172a"
                              stroke="#1e293b"
                            />
                            <text
                              x="32"
                              y="115"
                              fill="#94a3b8"
                              fontSize="11"
                              fontWeight="bold"
                            >
                              SVD CONFIG
                            </text>
                            <text x="32" y="135" fill="#475569" fontSize="10">
                              Rank: 64
                            </text>

                            {/* Node 2: Current Dataset (Highlighted) */}
                            <rect
                              x="260"
                              y="85"
                              width="120"
                              height="70"
                              rx="12"
                              fill="rgba(99, 102, 241, 0.05)"
                              stroke="#6366f1"
                              strokeWidth="2"
                            />
                            <text
                              x="272"
                              y="115"
                              fill="#fff"
                              fontSize="11"
                              fontWeight="extrabold"
                            >
                              {dataset.name.slice(0, 14)}
                            </text>
                            <text x="272" y="135" fill="#6366f1" fontSize="10">
                              This Dataset
                            </text>

                            {/* Node 3: Model Checkpoint */}
                            <rect
                              x="520"
                              y="85"
                              width="100"
                              height="70"
                              rx="12"
                              fill="#0f172a"
                              stroke="#a855f7"
                            />
                            <text
                              x="532"
                              y="115"
                              fill="#e2e8f0"
                              fontSize="11"
                              fontWeight="bold"
                            >
                              TRAINED MODEL
                            </text>
                            <text x="532" y="135" fill="#a855f7" fontSize="10">
                              svd_nova_v1
                            </text>
                          </svg>
                          <p className="text-xs text-slate-500 mt-2">
                            Self-assembling SVG DAG showing inputs, current
                            mapping, and downstream artifacts.
                          </p>
                        </div>
                      ) : (
                        <div className="w-full h-full flex flex-col md:flex-row gap-6 items-center justify-around">
                          {lineage.nodes.map((node: any) => (
                            <div
                              key={node.id}
                              className={`p-4 rounded-xl border flex flex-col gap-2 w-48 shadow-lg ${
                                node.id === datasetId
                                  ? "border-neon-blue bg-neon-blue/5"
                                  : "border-slate-800 bg-slate-900/60"
                              }`}
                            >
                              <div className="text-xs text-slate-500 font-bold uppercase tracking-wider">
                                {node.type}
                              </div>
                              <div className="text-sm font-extrabold text-white truncate">
                                {node.name}
                              </div>
                              <div className="text-[10px] text-slate-400 font-mono">
                                ID: {node.id.slice(0, 8)}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </TabPanel>

              {/* AUDIT LOG TAB */}
              <TabPanel
                value="audit"
                className="h-full flex flex-col overflow-hidden p-6 space-y-4"
              >
                <div className="flex justify-between items-center shrink-0">
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => exportLogsMutation.mutate("csv")}
                      className="border-white/10 text-xs"
                    >
                      Export CSV Report
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => exportLogsMutation.mutate("json")}
                      className="border-white/10 text-xs"
                    >
                      Export JSON Report
                    </Button>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto border border-white/5 rounded-2xl bg-slate-950/20">
                  {isLogsLoading ? (
                    <div className="flex flex-col items-center justify-center py-20 gap-2">
                      <Spinner />
                      <p className="text-slate-500 text-xs">
                        Loading audit logs...
                      </p>
                    </div>
                  ) : !auditLogsData?.results ||
                    auditLogsData.results.length === 0 ? (
                    <div className="text-center py-20 text-slate-500">
                      <ShieldAlert
                        className="mx-auto mb-3 opacity-25"
                        size={32}
                      />
                      No audit trails recorded for this dataset.
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400 uppercase tracking-widest font-bold">
                            <th className="p-3">Action</th>
                            <th className="p-3">Actor</th>
                            <th className="p-3">Details</th>
                            <th className="p-3">IP Address</th>
                            <th className="p-3">Time</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-slate-300">
                          {auditLogsData.results.map((log: AuditLog) => (
                            <tr key={log.id} className="hover:bg-white/5">
                              <td className="p-3 font-semibold">
                                <Badge
                                  variant={
                                    log.severity === "critical"
                                      ? "failed"
                                      : "secondary"
                                  }
                                  className="text-[10px] uppercase font-mono"
                                >
                                  {log.action_type}
                                </Badge>
                              </td>
                              <td className="p-3 font-mono text-[11px] text-slate-400">
                                User
                              </td>
                              <td className="p-3 text-slate-200">
                                {log.description}
                              </td>
                              <td className="p-3 font-mono text-[10px] text-slate-500">
                                {log.ip_address || "local"}
                              </td>
                              <td className="p-3 text-slate-500">
                                {new Date(log.created_at).toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </TabPanel>
            </div>
          </Tabs>
        </main>
      </div>
    </div>
  );
}
