// src/features/mlops/pages/FeatureStoreBrowser.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsAPI } from "../services/api";
import { FeatureGroupCard } from "../components/FeatureGroupCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import {
  Database,
  Plus,
  Search,
  Download,
  Layers,
  Zap,
  Globe,
  Trash2,
  Table,
  CheckCircle,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Feature, FeatureType } from "../types";

export default function FeatureStoreBrowser({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [selectedTag, setSelectedTag] = useState("all");
  const [showCreateModal, setShowCreateModal] = useState(false);

  // New Group Form State
  const [newGroupName, setNewGroupName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newEntityKey, setNewEntityKey] = useState("user_id");
  const [newTags, setNewTags] = useState("production, v1");
  const [newFeatures, setNewFeatures] = useState<
    { name: string; type: FeatureType; description: string }[]
  >([
    { name: "user_id", type: "string", description: "Primary entity identifier" },
    { name: "feature_score", type: "float", description: "Normalized score" },
  ]);

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: groups, isLoading } = useQuery({
    queryKey: ["feature-groups"],
    queryFn: () => mlopsAPI.getFeatureGroups(),
  });

  const createGroup = useMutation({
    mutationFn: () =>
      mlopsAPI.createFeatureGroup({
        name: newGroupName.trim().toLowerCase().replace(/\s+/g, "_"),
        description: newDescription,
        entity_key: newEntityKey,
        online_enabled: true,
        offline_enabled: true,
        tags: newTags.split(",").map((t) => t.trim()).filter(Boolean),
        features: newFeatures.map((f) => ({
          name: f.name,
          type: f.type,
          description: f.description,
          stats: {
            min: 0,
            max: 1,
            mean: 0.5,
            std: 0.2,
            null_count: 0,
            histogram: [10, 25, 40, 20, 5],
          },
        })),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feature-groups"] });
      setShowCreateModal(false);
      setNewGroupName("");
      setNewDescription("");
      setNewEntityKey("user_id");
      setNewFeatures([
        { name: "user_id", type: "string", description: "Primary entity identifier" },
        { name: "feature_score", type: "float", description: "Normalized score" },
      ]);
      success("Feature group created", "Schema registered in Feast / Hopsworks registry");
    },
    onError: (err: any) => {
      error("Failed to create feature group", err?.message || "Unknown error");
    },
  });

  const handleExportSchema = () => {
    if (!groups || groups.length === 0) return;
    const schemaJson = JSON.stringify(groups, null, 2);
    const blob = new Blob([schemaJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `feature_store_schema_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    success("Schema exported", "Feature Store schema JSON downloaded");
  };

  const allTags = Array.from(
    new Set(groups?.flatMap((g) => g.tags || []) || []),
  );

  const filtered =
    groups?.filter((g) => {
      const matchSearch =
        g.name.toLowerCase().includes(search.toLowerCase()) ||
        g.features.some((f) => f.name.toLowerCase().includes(search.toLowerCase())) ||
        (g.description && g.description.toLowerCase().includes(search.toLowerCase()));
      const matchTag = selectedTag === "all" || g.tags?.includes(selectedTag);
      return matchSearch && matchTag;
    }) || [];

  const totalFeatures =
    groups?.reduce((acc, g) => acc + g.features.length, 0) || 0;
  const onlineCount = groups?.filter((g) => g.online_enabled).length || 0;
  const totalRecords =
    groups?.reduce((acc, g) => acc + (g.records_count || 0), 0) || 0;

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Database size={20} className="text-neon-mint" /> Feature Store Browser
          </h1>
          <p className="text-xs text-slate-400">
            Enterprise feature repository for online low-latency inference & offline training
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={handleExportSchema}
            title="Export Schema JSON"
          >
            <Download size={14} className="mr-1" /> Export Schema
          </Button>
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Group
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* KPI Stats Overview */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400 flex items-center justify-center gap-1">
              <Layers size={13} className="text-neon-blue" /> Feature Groups
            </div>
            <div className="text-2xl font-bold text-white mt-1">
              {groups?.length || 0}
            </div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400 flex items-center justify-center gap-1">
              <Table size={13} className="text-neon-mint" /> Total Features
            </div>
            <div className="text-2xl font-bold text-white mt-1">
              {totalFeatures}
            </div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400 flex items-center justify-center gap-1">
              <Globe size={13} className="text-green-400" /> Online Enabled
            </div>
            <div className="text-2xl font-bold text-green-400 mt-1">
              {onlineCount}
            </div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400 flex items-center justify-center gap-1">
              <Zap size={13} className="text-yellow-400" /> Total Ingested
            </div>
            <div className="text-2xl font-bold text-yellow-400 mt-1">
              {totalRecords > 1000000
                ? `${(totalRecords / 1000000).toFixed(1)}M`
                : totalRecords.toLocaleString()}
            </div>
          </TerminalCard>
        </div>

        {/* Search and Tag Filters */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="relative w-full max-w-sm">
            <Input
              placeholder="Search groups, features, entity keys..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-8"
            />
            <Search
              size={14}
              className="absolute left-2.5 top-3 text-slate-500 pointer-events-none"
            />
          </div>

          <div className="flex flex-wrap gap-1.5 items-center">
            <span className="text-xs text-slate-400 mr-1">Tags:</span>
            <button
              onClick={() => setSelectedTag("all")}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                selectedTag === "all"
                  ? "bg-neon-blue text-white font-bold"
                  : "bg-white/5 text-slate-400 hover:text-white"
              }`}
            >
              All ({groups?.length || 0})
            </button>
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setSelectedTag(tag)}
                className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
                  selectedTag === tag
                    ? "bg-neon-blue text-white font-bold"
                    : "bg-white/5 text-slate-400 hover:text-white"
                }`}
              >
                #{tag}
              </button>
            ))}
          </div>
        </div>

        {/* Feature Groups Grid */}
        {filtered.length > 0 ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((group) => (
              <FeatureGroupCard
                key={group.id}
                group={group}
                onRefresh={() =>
                  queryClient.invalidateQueries({ queryKey: ["feature-groups"] })
                }
              />
            ))}
          </div>
        ) : (
          <div className="text-center text-slate-500 py-16">
            <Database size={32} className="mx-auto mb-2 opacity-40" />
            <p>No feature groups found matching "{search}".</p>
          </div>
        )}
      </div>

      {/* Create Feature Group Modal */}
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Feature Group"
      >
        <div className="space-y-4 max-h-[75vh] overflow-y-auto pr-1">
          <Input
            label="Group Name (lowercase, snake_case)"
            value={newGroupName}
            onChange={(e) => setNewGroupName(e.target.value)}
            placeholder="e.g. user_session_features_v1"
          />

          <Input
            label="Description"
            value={newDescription}
            onChange={(e) => setNewDescription(e.target.value)}
            placeholder="Real-time sequence embeddings and engagement metrics"
          />

          <div className="grid grid-cols-2 gap-3">
            <Input
              label="Entity Key (Primary Join Key)"
              value={newEntityKey}
              onChange={(e) => setNewEntityKey(e.target.value)}
              placeholder="user_id"
            />
            <Input
              label="Tags (comma-separated)"
              value={newTags}
              onChange={(e) => setNewTags(e.target.value)}
              placeholder="production, v2, recommender"
            />
          </div>

          {/* Feature Schema Builder */}
          <div className="space-y-2 pt-2 border-t border-white/10">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-white">
                Feature Schema Definitions ({newFeatures.length})
              </label>
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  setNewFeatures((prev) => [
                    ...prev,
                    {
                      name: `feature_${prev.length + 1}`,
                      type: "float",
                      description: "",
                    },
                  ])
                }
              >
                <Plus size={12} className="mr-1" /> Add Column
              </Button>
            </div>

            <div className="space-y-2">
              {newFeatures.map((feat, idx) => (
                <div
                  key={idx}
                  className="p-2 bg-white/5 rounded border border-white/10 grid grid-cols-12 gap-2 items-center text-xs"
                >
                  <div className="col-span-4">
                    <input
                      className="w-full bg-slate-900 border border-white/10 rounded px-2 py-1 text-white font-mono text-xs focus:outline-none focus:border-neon-mint"
                      placeholder="Column name"
                      value={feat.name}
                      onChange={(e) => {
                        const updated = [...newFeatures];
                        updated[idx].name = e.target.value;
                        setNewFeatures(updated);
                      }}
                    />
                  </div>
                  <div className="col-span-3">
                    <select
                      className="w-full bg-slate-900 border border-white/10 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-neon-mint"
                      value={feat.type}
                      onChange={(e) => {
                        const updated = [...newFeatures];
                        updated[idx].type = e.target.value as FeatureType;
                        setNewFeatures(updated);
                      }}
                    >
                      <option value="int">int</option>
                      <option value="float">float</option>
                      <option value="string">string</option>
                      <option value="boolean">boolean</option>
                      <option value="datetime">datetime</option>
                    </select>
                  </div>
                  <div className="col-span-4">
                    <input
                      className="w-full bg-slate-900 border border-white/10 rounded px-2 py-1 text-slate-400 text-xs focus:outline-none focus:border-neon-mint"
                      placeholder="Description"
                      value={feat.description}
                      onChange={(e) => {
                        const updated = [...newFeatures];
                        updated[idx].description = e.target.value;
                        setNewFeatures(updated);
                      }}
                    />
                  </div>
                  <div className="col-span-1 text-right">
                    {newFeatures.length > 1 && (
                      <button
                        onClick={() =>
                          setNewFeatures(newFeatures.filter((_, i) => i !== idx))
                        }
                        className="text-slate-500 hover:text-red-400"
                        title="Remove Column"
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createGroup.mutate()}
              disabled={!newGroupName.trim() || !newEntityKey.trim() || createGroup.isPending}
            >
              {createGroup.isPending ? "Creating..." : "Create Feature Group"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

