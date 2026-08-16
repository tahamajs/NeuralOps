import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { datasetsAPI, Dataset } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Select } from "../../../components/ui/Select";
import {
  Database,
  Plus,
  Search,
  FileText,
  Globe,
  Lock,
  GitBranch,
  ArrowRight,
  ShieldCheck,
  Tag,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function DatasetListPage({ onClose }: { onClose?: () => void }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTag, setSelectedTag] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);

  // New Dataset Form States
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isPublic, setIsPublic] = useState(false);
  const [status, setStatus] = useState<
    "draft" | "published" | "deprecated" | "archived"
  >("draft");
  const [dvcRemote, setDvcRemote] = useState("");
  const [dvcRevision, setDvcRevision] = useState("");
  const [license, setLicense] = useState("MIT");
  const [tags, setTags] = useState("");

  const { data: datasetsData, isLoading } = useQuery({
    queryKey: ["datasets", searchQuery, selectedTag],
    queryFn: () =>
      datasetsAPI.getDatasets({ q: searchQuery, tag: selectedTag }),
  });

  const createDatasetMutation = useMutation({
    mutationFn: (data: Partial<Dataset>) => datasetsAPI.createDataset(data),
    onSuccess: (newDataset) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      success(
        "Dataset Created",
        `Successfully created dataset: ${newDataset.name}`,
      );
      setShowCreateModal(false);
      resetForm();
    },
    onError: (err: any) => {
      error("Creation Failed", err.userMessage || "Could not create dataset");
    },
  });

  const resetForm = () => {
    setName("");
    setDescription("");
    setIsPublic(false);
    setStatus("draft");
    setDvcRemote("");
    setDvcRevision("");
    setLicense("MIT");
    setTags("");
  };

  const handleCreateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    createDatasetMutation.mutate({
      name,
      description,
      is_public: isPublic,
      status,
      dvc_remote: dvcRemote,
      dvc_revision: dvcRevision,
      license,
      tags: tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
    });
  };

  const getStatusColor = (status: Dataset["status"]) => {
    switch (status) {
      case "published":
        return "success";
      case "draft":
        return "warning";
      case "deprecated":
        return "secondary";
      case "archived":
        return "failed";
      default:
        return "info";
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950/40 backdrop-blur-md rounded-2xl overflow-hidden border border-white/5 font-sans text-slate-200">
      {/* Header */}
      <div className="flex justify-between items-center p-6 border-b border-white/10 bg-slate-900/20">
        <div>
          <h1 className="text-2xl font-black text-white flex items-center gap-2 tracking-tight">
            <Database size={24} className="text-neon-blue animate-pulse" />
            Dataset Catalog
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Browse, search, and manage data lineage artifacts with DVC sync.
          </p>
        </div>
        <div className="flex gap-3">
          <Button
            size="sm"
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-r from-neon-blue to-purple-600 hover:from-neon-blue/90 hover:to-purple-600/90 text-white shadow-glow-sm"
          >
            <Plus size={16} className="mr-1" /> New Dataset
          </Button>
          {onClose && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onClose}
              className="hover:bg-white/5"
            >
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Toolbar / Filters */}
      <div className="p-4 border-b border-white/10 bg-slate-900/10 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="relative w-full md:max-w-md">
          <Search
            size={16}
            className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500"
          />
          <Input
            placeholder="Search datasets by name or description..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 w-full bg-white/5 border-white/10 hover:border-white/20 focus:border-neon-blue/50"
          />
        </div>

        <div className="flex gap-3 w-full md:w-auto shrink-0">
          <Input
            placeholder="Filter by tag..."
            value={selectedTag}
            onChange={(e) => setSelectedTag(e.target.value)}
            className="w-full md:w-48 bg-white/5 border-white/10"
          />
        </div>
      </div>

      {/* Main Grid */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <Spinner />
            <p className="text-sm text-slate-400">Loading datasets...</p>
          </div>
        ) : !datasetsData?.results || datasetsData.results.length === 0 ? (
          <TerminalCard className="p-12 text-center border-slate-900/60 bg-slate-900/10 backdrop-blur-xl">
            <Database
              size={48}
              className="mx-auto mb-3 opacity-20 text-slate-400"
            />
            <h3 className="text-lg font-bold text-white mb-1">
              No Datasets Found
            </h3>
            <p className="text-sm text-slate-500 max-w-sm mx-auto mb-4">
              Get started by creating a new dataset mapping or sync with DVC.
            </p>
            <Button size="sm" onClick={() => setShowCreateModal(true)}>
              Create First Dataset
            </Button>
          </TerminalCard>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {datasetsData.results.map((dataset) => (
              <TerminalCard
                key={dataset.id}
                className="p-5 hover:bg-slate-900/40 hover:border-neon-blue/30 transition-all duration-300 group cursor-pointer flex flex-col justify-between border-slate-800/80 bg-slate-900/20 backdrop-blur-lg"
                onClick={() =>
                  navigate(`/dashboard?win=dataset_detail&id=${dataset.id}`)
                }
              >
                <div>
                  <div className="flex justify-between items-start gap-2 mb-2">
                    <h3 className="font-extrabold text-lg text-white group-hover:text-neon-blue transition-colors">
                      {dataset.name}
                    </h3>
                    <div className="flex items-center gap-2">
                      <Badge variant={getStatusColor(dataset.status)}>
                        {dataset.status}
                      </Badge>
                      {dataset.is_public ? (
                        <span title="Public">
                          <Globe size={14} className="text-emerald-400" />
                        </span>
                      ) : (
                        <span title="Private">
                          <Lock size={14} className="text-slate-500" />
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-sm text-slate-400 line-clamp-2 mb-4 leading-relaxed">
                    {dataset.description || "No description provided."}
                  </p>

                  <div className="flex flex-wrap gap-1.5 mb-4">
                    {dataset.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-xs text-slate-300 flex items-center gap-1"
                      >
                        <Tag size={10} /> {tag}
                      </span>
                    ))}
                    {dataset.license && (
                      <span className="px-2 py-0.5 rounded bg-neon-blue/10 border border-neon-blue/20 text-xs text-neon-blue">
                        {dataset.license}
                      </span>
                    )}
                  </div>
                </div>

                <div className="pt-4 border-t border-white/5 flex items-center justify-between text-xs text-slate-500">
                  <div className="flex gap-4">
                    {dataset.dvc_revision && (
                      <span className="flex items-center gap-1 font-mono text-[11px] text-purple-400">
                        <GitBranch size={12} />{" "}
                        {dataset.dvc_revision.slice(0, 7)}
                      </span>
                    )}
                    <span>
                      Updated{" "}
                      {new Date(dataset.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                  <span className="text-neon-blue font-bold group-hover:translate-x-1 transition-transform flex items-center gap-1">
                    Details <ArrowRight size={12} />
                  </span>
                </div>
              </TerminalCard>
            ))}
          </div>
        )}
      </div>

      {/* Creation Modal */}
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Dataset Metadata Mapping"
      >
        <form onSubmit={handleCreateSubmit} className="space-y-4">
          <Input
            label="Dataset Name *"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. WikiText-103 Compressed"
            required
            className="bg-white/5 border-white/10"
          />

          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe dataset distribution, schema, and use cases..."
              rows={3}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                License
              </label>
              <Input
                value={license}
                onChange={(e) => setLicense(e.target.value)}
                placeholder="MIT / Apache 2.0 / CC-BY"
                className="bg-white/5 border-white/10"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                Status
              </label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value as any)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:border-neon-blue/50 transition-colors"
              >
                <option value="draft" className="bg-slate-900 text-white">
                  Draft
                </option>
                <option value="published" className="bg-slate-900 text-white">
                  Published
                </option>
                <option value="deprecated" className="bg-slate-900 text-white">
                  Deprecated
                </option>
                <option value="archived" className="bg-slate-900 text-white">
                  Archived
                </option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Input
              label="DVC Remote URL"
              value={dvcRemote}
              onChange={(e) => setDvcRemote(e.target.value)}
              placeholder="s3://my-bucket/dvc-store"
              className="bg-white/5 border-white/10"
            />
            <Input
              label="DVC Revision Hash"
              value={dvcRevision}
              onChange={(e) => setDvcRevision(e.target.value)}
              placeholder="revision_hash_or_branch"
              className="bg-white/5 border-white/10 font-mono text-xs"
            />
          </div>

          <Input
            label="Tags (comma-separated)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="nlp, transformer, text-generation"
            className="bg-white/5 border-white/10"
          />

          <div className="flex items-center gap-3 py-2">
            <input
              type="checkbox"
              id="isPublic"
              checked={isPublic}
              onChange={(e) => setIsPublic(e.target.checked)}
              className="w-4 h-4 accent-neon-blue bg-white/5 border-white/10 rounded"
            />
            <label
              htmlFor="isPublic"
              className="text-sm font-semibold text-slate-300 select-none cursor-pointer flex items-center gap-1.5"
            >
              <Globe size={14} className="text-slate-400" /> Make this dataset
              public to all cluster members
            </label>
          </div>

          <div className="flex gap-3 justify-end pt-4 border-t border-white/10">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setShowCreateModal(false)}
              className="hover:bg-white/5"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createDatasetMutation.isPending}
              className="bg-neon-blue hover:bg-neon-blue/80 text-white"
            >
              {createDatasetMutation.isPending
                ? "Creating..."
                : "Create Mapping"}
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
