import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import { useToast } from "../../../components/ui/Toast";
import { mlopsAPI } from "../services/api";
import {
  Database,
  Upload,
  Download,
  BarChart2,
  CheckCircle,
  Zap,
  Globe,
  HardDrive,
  Info,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { FeatureGroup, Feature } from "../types";

export function FeatureGroupCard({
  group,
  onRefresh,
}: {
  group: FeatureGroup;
  onRefresh: () => void;
}) {
  const { success, error } = useToast();
  const [showIngestModal, setShowIngestModal] = useState(false);
  const [showServeModal, setShowServeModal] = useState(false);
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(
    group.features[1] || group.features[0] || null,
  );

  // Ingest state
  const [ingestData, setIngestData] = useState(
    JSON.stringify(
      [
        {
          [group.entity_key]: `entity_${Math.floor(Math.random() * 10000)}`,
          ...(group.features[1] ? { [group.features[1].name]: 42.5 } : {}),
          ...(group.features[2] ? { [group.features[2].name]: 0.15 } : {}),
        },
      ],
      null,
      2,
    ),
  );
  const [isIngesting, setIsIngesting] = useState(false);

  // Serve state
  const [serveKeys, setServeKeys] = useState("entity_1001, entity_1002");
  const [serveResults, setServeResults] = useState<{
    features: Record<string, any>[];
    latency_ms: number;
  } | null>(null);
  const [isServing, setIsServing] = useState(false);

  const handleIngest = async () => {
    try {
      setIsIngesting(true);
      const parsed = JSON.parse(ingestData);
      const records = Array.isArray(parsed) ? parsed : [parsed];
      await mlopsAPI.ingestFeatures(group.id, records);
      success("Features ingested", `${records.length} record(s) written to online and offline store`);
      setShowIngestModal(false);
      onRefresh();
    } catch (e: any) {
      error("Ingestion failed", e?.message || "Invalid JSON payload format");
    } finally {
      setIsIngesting(false);
    }
  };

  const handleServe = async () => {
    try {
      setIsServing(true);
      const keys = serveKeys
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean);
      const res = await mlopsAPI.serveFeatures(group.id, keys);
      setServeResults(res);
      success("Features served", `Retrieved ${res.features.length} vector(s) in ${res.latency_ms}ms`);
    } catch (e: any) {
      error("Serving failed", e?.message || "Could not retrieve features");
    } finally {
      setIsServing(false);
    }
  };

  const typeColorMap: Record<string, "info" | "success" | "warning" | "secondary"> = {
    int: "info",
    float: "info",
    string: "success",
    boolean: "warning",
    datetime: "secondary",
  };

  // Prepare histogram data for selected feature
  const histogramData =
    selectedFeature?.stats?.histogram?.map((val, idx) => ({
      bin: selectedFeature.stats?.bins?.[idx] || `Bin ${idx + 1}`,
      count: val,
    })) || [];

  return (
    <>
      <TerminalCard className="p-4 flex flex-col justify-between hover:border-white/20 transition-colors">
        <div>
          <div className="flex justify-between items-start">
            <div className="flex items-center gap-2">
              <Database size={18} className="text-neon-mint flex-shrink-0" />
              <div>
                <h3 className="font-bold text-white text-base leading-tight">
                  {group.name}
                </h3>
                <span className="text-xs text-slate-400">
                  v{group.version || "1.0"} · Key:{" "}
                  <code className="font-mono text-neon-blue">{group.entity_key}</code>
                </span>
              </div>
            </div>
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="ghost"
                title="View Feature Statistics & Distribution"
                onClick={() => setShowStatsModal(true)}
                className="hover:text-neon-mint"
              >
                <BarChart2 size={15} />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                title="Ingest Features"
                onClick={() => setShowIngestModal(true)}
                className="hover:text-neon-blue"
              >
                <Upload size={15} />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                title="Serve / Query Online Features"
                onClick={() => setShowServeModal(true)}
                className="hover:text-yellow-400"
              >
                <Download size={15} />
              </Button>
            </div>
          </div>

          {group.description && (
            <p className="text-xs text-slate-300 mt-2 line-clamp-2">
              {group.description}
            </p>
          )}

          <div className="flex items-center gap-3 mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <Globe
                size={12}
                className={group.online_enabled ? "text-green-400" : "text-slate-600"}
              />
              Online: {group.online_enabled ? "Active" : "Disabled"}
            </span>
            <span className="flex items-center gap-1">
              <HardDrive
                size={12}
                className={group.offline_enabled ? "text-green-400" : "text-slate-600"}
              />
              Offline: {group.offline_enabled ? "Parquet" : "Disabled"}
            </span>
            <span>
              Records: <strong>{group.records_count?.toLocaleString() || "1,420,000"}</strong>
            </span>
          </div>

          <div className="mt-3 pt-3 border-t border-white/5">
            <div className="text-xs font-semibold text-slate-400 mb-1.5 flex justify-between">
              <span>Schema ({group.features.length} features)</span>
              <button
                onClick={() => setShowStatsModal(true)}
                className="text-xs text-neon-mint hover:underline"
              >
                Inspect stats →
              </button>
            </div>
            <div className="flex flex-wrap gap-1">
              {group.features.slice(0, 6).map((f) => (
                <span
                  key={f.name}
                  onClick={() => {
                    setSelectedFeature(f);
                    setShowStatsModal(true);
                  }}
                  className="cursor-pointer"
                  title={`${f.name} (${f.type}): ${f.description || "No description"}`}
                >
                  <Badge
                    variant={typeColorMap[f.type] || "secondary"}
                    className="font-mono text-xs py-0.5 hover:bg-white/20"
                  >
                    {f.name}
                    <span className="opacity-60 ml-1 text-[10px]">:{f.type}</span>
                  </Badge>
                </span>
              ))}
              {group.features.length > 6 && (
                <Badge variant="secondary" className="text-xs">
                  +{group.features.length - 6} more
                </Badge>
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center mt-3 pt-2 text-[11px] text-slate-500 border-t border-white/5">
          <span>Updated {new Date(group.updated_at).toLocaleDateString()}</span>
          <div className="flex gap-1">
            {group.tags?.map((t) => (
              <span
                key={t}
                className="px-1.5 py-0.5 rounded bg-white/5 text-slate-400 font-mono text-[10px]"
              >
                #{t}
              </span>
            ))}
          </div>
        </div>
      </TerminalCard>

      {/* Feature Statistics & Histogram Modal */}
      <Modal
        open={showStatsModal}
        onClose={() => setShowStatsModal(false)}
        title={`Feature Statistics: ${group.name}`}
      >
        <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-1">
          <div>
            <label className="text-xs text-slate-400 mb-1 block">
              Select Feature to inspect
            </label>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {group.features.map((f) => (
                <button
                  key={f.name}
                  onClick={() => setSelectedFeature(f)}
                  className={`px-2 py-1 text-xs rounded font-mono border transition-colors ${
                    selectedFeature?.name === f.name
                      ? "border-neon-mint bg-neon-mint/10 text-white font-bold"
                      : "border-white/10 bg-white/5 text-slate-400 hover:text-white"
                  }`}
                >
                  {f.name}
                </button>
              ))}
            </div>
          </div>

          {selectedFeature && (
            <div className="space-y-3">
              <div className="p-3 bg-white/5 rounded border border-white/10 flex justify-between items-start">
                <div>
                  <h4 className="font-bold text-white font-mono text-sm">
                    {selectedFeature.name}
                  </h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {selectedFeature.description || "No description provided"}
                  </p>
                </div>
                <Badge variant={typeColorMap[selectedFeature.type] || "info"}>
                  Type: {selectedFeature.type}
                </Badge>
              </div>

              {selectedFeature.stats ? (
                <>
                  <div className="grid grid-cols-4 gap-2 text-center text-xs">
                    <div className="p-2 bg-white/5 rounded">
                      <div className="text-slate-400">Min</div>
                      <div className="font-bold text-white text-sm font-mono mt-0.5">
                        {selectedFeature.stats.min}
                      </div>
                    </div>
                    <div className="p-2 bg-white/5 rounded">
                      <div className="text-slate-400">Max</div>
                      <div className="font-bold text-white text-sm font-mono mt-0.5">
                        {selectedFeature.stats.max}
                      </div>
                    </div>
                    <div className="p-2 bg-white/5 rounded">
                      <div className="text-slate-400">Mean</div>
                      <div className="font-bold text-neon-mint text-sm font-mono mt-0.5">
                        {selectedFeature.stats.mean}
                      </div>
                    </div>
                    <div className="p-2 bg-white/5 rounded">
                      <div className="text-slate-400">Std Dev</div>
                      <div className="font-bold text-white text-sm font-mono mt-0.5">
                        {selectedFeature.stats.std}
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-4 text-xs text-slate-400 px-1">
                    <span>
                      Null count: <strong>{selectedFeature.stats.null_count}</strong>
                    </span>
                    {selectedFeature.stats.unique_count !== undefined && (
                      <span>
                        Unique values:{" "}
                        <strong>
                          {selectedFeature.stats.unique_count.toLocaleString()}
                        </strong>
                      </span>
                    )}
                  </div>

                  {histogramData.length > 0 && (
                    <div className="p-3 bg-white/5 rounded border border-white/10">
                      <h5 className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1">
                        <BarChart2 size={13} className="text-neon-mint" /> Sample Distribution Histogram
                      </h5>
                      <ResponsiveContainer width="100%" height={160}>
                        <BarChart data={histogramData}>
                          <XAxis
                            dataKey="bin"
                            stroke="#64748b"
                            tick={{ fontSize: 10 }}
                          />
                          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0a0f1a",
                              borderColor: "#334155",
                              borderRadius: "4px",
                              fontSize: "11px",
                            }}
                          />
                          <Bar
                            dataKey="count"
                            fill="#10b981"
                            radius={[3, 3, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-6 text-slate-500 text-xs">
                  No statistical profile recorded yet for this column.
                </div>
              )}
            </div>
          )}
        </div>
      </Modal>

      {/* Ingest Features Modal */}
      <Modal
        open={showIngestModal}
        onClose={() => setShowIngestModal(false)}
        title={`Ingest Features: ${group.name}`}
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400">
            Submit JSON records to be written into the Feature Store online Redis
            and offline Parquet layers.
          </p>
          <div className="space-y-1">
            <label className="text-xs font-mono text-slate-300">
              JSON Records (Array or Object)
            </label>
            <textarea
              rows={8}
              value={ingestData}
              onChange={(e) => setIngestData(e.target.value)}
              className="w-full bg-slate-900 border border-white/10 rounded p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-neon-mint"
            />
          </div>
          <div className="flex justify-between items-center">
            <button
              onClick={() => {
                const sample = [
                  {
                    [group.entity_key]: `entity_${Math.floor(Math.random() * 90000 + 10000)}`,
                    ...(group.features[1] ? { [group.features[1].name]: parseFloat((Math.random() * 100).toFixed(2)) } : {}),
                    ...(group.features[2] ? { [group.features[2].name]: parseFloat((Math.random()).toFixed(3)) } : {}),
                  },
                ];
                setIngestData(JSON.stringify(sample, null, 2));
              }}
              className="text-xs text-neon-blue hover:underline"
            >
              Generate Random Sample
            </button>
            <div className="flex gap-2">
              <Button variant="ghost" onClick={() => setShowIngestModal(false)}>
                Cancel
              </Button>
              <Button onClick={handleIngest} disabled={isIngesting}>
                {isIngesting ? "Ingesting..." : "Ingest Records"}
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      {/* Serve Features / Online Query Modal */}
      <Modal
        open={showServeModal}
        onClose={() => setShowServeModal(false)}
        title={`Serve Features: ${group.name}`}
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400">
            Simulate low-latency point-in-time online feature retrieval by entity
            keys.
          </p>
          <div>
            <Input
              label={`Entity Keys (comma-separated ${group.entity_key})`}
              value={serveKeys}
              onChange={(e) => setServeKeys(e.target.value)}
              placeholder="user_101, user_102, user_103"
            />
          </div>
          <Button onClick={handleServe} disabled={isServing || !serveKeys.trim()}>
            {isServing ? "Retrieving..." : "Query Online Store"}
          </Button>

          {serveResults && (
            <div className="space-y-2 mt-3 pt-3 border-t border-white/10">
              <div className="flex justify-between items-center text-xs">
                <span className="text-neon-mint flex items-center gap-1 font-semibold">
                  <CheckCircle size={14} /> Retrieved {serveResults.features.length} vector(s)
                </span>
                <span className="text-slate-400 font-mono flex items-center gap-1">
                  <Zap size={12} className="text-yellow-400" /> {serveResults.latency_ms} ms
                </span>
              </div>
              <div className="max-h-48 overflow-auto bg-slate-950 p-2 rounded border border-white/10">
                <pre className="text-[11px] font-mono text-neon-blue">
                  {JSON.stringify(serveResults.features, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </Modal>
    </>
  );
}

