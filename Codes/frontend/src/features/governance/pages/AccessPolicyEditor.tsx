// src/features/governance/pages/AccessPolicyEditor.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { PolicyMatrix } from "../components/PolicyMatrix";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Switch } from "../../../components/ui/Switch";
import {
  Shield,
  Plus,
  Trash2,
  Edit,
  Play,
  Download,
  CheckCircle2,
  XCircle,
  FlaskConical,
  Sparkles,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { AccessPolicy } from "../types";

const SIMULATOR_PRESETS = [
  {
    label: "ML Eng reading dataset",
    context: {
      user: "alex.rivas",
      role: "ml_engineer",
      action: "read:dataset",
      resource: "dataset:training-features",
      ip: "192.168.1.104",
    },
  },
  {
    label: "Viewer deleting prod model",
    context: {
      user: "guest_viewer",
      role: "viewer",
      action: "delete:model",
      resource: "model:prod-svd-transformer",
      ip: "198.51.100.22",
    },
  },
  {
    label: "Auditor exporting logs",
    context: {
      user: "ey_partner",
      role: "compliance_auditor",
      action: "export:audit_log",
      resource: "governance:logs",
      ip: "10.0.1.5",
    },
  },
];

export default function AccessPolicyEditor({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<AccessPolicy | null>(null);
  const [testContext, setTestContext] = useState(
    JSON.stringify(SIMULATOR_PRESETS[0].context, null, 2),
  );
  const [testResult, setTestResult] = useState<any>(null);

  // New Policy Form State
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newEffect, setNewEffect] = useState<"allow" | "deny">("allow");
  const [newActions, setNewActions] = useState("read:*");
  const [newResources, setNewResources] = useState("model:*");
  const [newSubjects, setNewSubjects] = useState("role:ml_engineer");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: policies, isLoading } = useQuery({
    queryKey: ["access-policies"],
    queryFn: () => governanceAPI.getAccessPolicies(),
  });

  // Auto select first policy
  useEffect(() => {
    if (policies && policies.length > 0 && !editingPolicy) {
      setEditingPolicy(policies[0]);
    }
  }, [policies, editingPolicy]);

  const createPolicy = useMutation({
    mutationFn: () =>
      governanceAPI.createAccessPolicy({
        name: newName,
        description: newDesc,
        effect: newEffect,
        actions: newActions.split(",").map((s) => s.trim()).filter(Boolean),
        resources: newResources.split(",").map((s) => s.trim()).filter(Boolean),
        subjects: newSubjects.split(",").map((s) => s.trim()).filter(Boolean),
        priority: 50,
        enabled: true,
      }),
    onSuccess: (newPol) => {
      queryClient.invalidateQueries({ queryKey: ["access-policies"] });
      setEditingPolicy(newPol);
      setShowCreateModal(false);
      setNewName("");
      setNewDesc("");
      success("Access Policy created");
    },
  });

  const updatePolicy = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<AccessPolicy> }) =>
      governanceAPI.updateAccessPolicy(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["access-policies"] });
      success("Policy rules saved");
    },
  });

  const deletePolicy = useMutation({
    mutationFn: (id: string) => governanceAPI.deleteAccessPolicy(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["access-policies"] });
      setEditingPolicy(null);
      success("Policy deleted");
    },
  });

  const testPolicy = useMutation({
    mutationFn: () => {
      let parsed = {};
      try {
        parsed = JSON.parse(testContext);
      } catch {
        parsed = { raw: testContext };
      }
      return governanceAPI.testAccessPolicy(editingPolicy?.id || "", parsed);
    },
    onSuccess: (res) => {
      setTestResult(res.data);
    },
  });

  const handleExportPolicies = () => {
    if (!policies || policies.length === 0) return;
    const jsonStr = JSON.stringify(policies, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `access_policies_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    success("Policies exported", "JSON policy definitions downloaded");
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-neon-mint" /> Access Policy Editor & Test Simulator
          </h1>
          <p className="text-xs text-slate-400">
            Fine-grained RBAC & ABAC authorization rules for models, datasets, and pipelines
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={handleExportPolicies}>
            <Download size={14} className="mr-1" /> Export JSON
          </Button>
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Policy
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-12 gap-4">
        {/* Left Column: Policies List */}
        <div className="lg:col-span-5 space-y-2.5 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
          {policies?.map((p) => {
            const isSelected = editingPolicy?.id === p.id;
            return (
              <TerminalCard
                key={p.id}
                className={`p-3.5 cursor-pointer transition-all ${
                  isSelected
                    ? "border-neon-blue bg-neon-blue/10 shadow-lg"
                    : "hover:border-white/20 hover:bg-white/5"
                }`}
                onClick={() => {
                  setEditingPolicy(p);
                  setTestResult(null);
                }}
              >
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-bold text-white text-xs">{p.name}</span>
                      <Badge
                        variant={p.effect === "allow" ? "success" : "failed"}
                        className="text-[10px] uppercase font-bold py-0"
                      >
                        {p.effect}
                      </Badge>
                      <Badge
                        variant={p.enabled ? "success" : "secondary"}
                        className="text-[10px] py-0"
                      >
                        {p.enabled ? "Active" : "Disabled"}
                      </Badge>
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-2">
                      {p.description}
                    </p>
                    <div className="flex gap-2 pt-1 font-mono text-[10px] text-slate-500">
                      <span>Priority: <strong>{p.priority}</strong></span>
                      <span>•</span>
                      <span>Subjects: <strong>{p.subjects.length}</strong></span>
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deletePolicy.mutate(p.id);
                    }}
                    className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                    title="Delete Policy"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </TerminalCard>
            );
          })}
        </div>

        {/* Right Column: Policy Detail & Simulator */}
        <div className="lg:col-span-7 space-y-4">
          {editingPolicy ? (
            <div className="space-y-4">
              {/* Policy Header Settings */}
              <TerminalCard className="p-4 space-y-3">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Edit size={14} className="text-neon-mint" /> Policy Configuration
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">Status:</span>
                    <Switch
                      checked={editingPolicy.enabled}
                      onChange={(checked) =>
                        setEditingPolicy({ ...editingPolicy, enabled: checked })
                      }
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="col-span-2">
                    <Input
                      label="Policy Name"
                      value={editingPolicy.name}
                      onChange={(e) =>
                        setEditingPolicy({ ...editingPolicy, name: e.target.value })
                      }
                      className="text-xs"
                    />
                  </div>
                  <div>
                    <Select
                      label="Effect"
                      options={[
                        { value: "allow", label: "Allow (Permit)" },
                        { value: "deny", label: "Deny (Block)" },
                      ]}
                      value={editingPolicy.effect}
                      onChange={(e) =>
                        setEditingPolicy({ ...editingPolicy, effect: e.target.value as any })
                      }
                      className="text-xs"
                    />
                  </div>
                </div>

                <Input
                  label="Description / Governance Objective"
                  value={editingPolicy.description}
                  onChange={(e) =>
                    setEditingPolicy({ ...editingPolicy, description: e.target.value })
                  }
                  className="text-xs"
                />
              </TerminalCard>

              {/* Policy Matrix: Actions, Resources, Roles */}
              <PolicyMatrix policy={editingPolicy} onChange={setEditingPolicy} />

              {/* Policy Simulator Tool */}
              <TerminalCard className="p-4 space-y-3">
                <div className="flex justify-between items-center flex-wrap gap-2">
                  <h3 className="text-xs font-bold text-white flex items-center gap-1.5 uppercase tracking-wide">
                    <FlaskConical size={14} className="text-neon-mint" /> Policy Test Simulator
                  </h3>
                  <div className="flex gap-1">
                    {SIMULATOR_PRESETS.map((p, idx) => (
                      <button
                        key={idx}
                        onClick={() => setTestContext(JSON.stringify(p.context, null, 2))}
                        className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white border border-white/5 text-[10px]"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                <textarea
                  rows={3}
                  className="w-full bg-slate-900 border border-white/10 rounded p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-neon-mint"
                  value={testContext}
                  onChange={(e) => setTestContext(e.target.value)}
                  placeholder='{"user": "alex", "role": "ml_engineer", "action": "read:dataset", "resource": "dataset:training-v2"}'
                />

                <div className="flex justify-between items-center">
                  <Button
                    size="sm"
                    onClick={() => testPolicy.mutate()}
                    disabled={testPolicy.isPending}
                    className="bg-neon-blue text-white"
                  >
                    <Play size={12} className="mr-1" />{" "}
                    {testPolicy.isPending ? "Evaluating..." : "Simulate Authorization"}
                  </Button>

                  {testResult && (
                    <div className="flex items-center gap-2 font-mono text-xs">
                      {testResult.allowed ? (
                        <span className="flex items-center gap-1 text-green-400 font-bold bg-green-500/10 px-2 py-1 rounded border border-green-500/20">
                          <CheckCircle2 size={13} /> AUTHORIZED (ALLOW)
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-red-400 font-bold bg-red-500/10 px-2 py-1 rounded border border-red-500/20">
                          <XCircle size={13} /> ACCESS DENIED (DENY)
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {testResult && (
                  <div className="p-2 bg-slate-900 rounded border border-white/5 text-[11px] font-mono text-slate-300">
                    <div className="text-slate-400">Rule: {testResult.matched_rule}</div>
                    <div className="text-slate-500 text-[10px] mt-0.5">{testResult.reason}</div>
                  </div>
                )}
              </TerminalCard>

              {/* Action Buttons */}
              <div className="flex justify-end gap-2 pt-1">
                <Button
                  onClick={() =>
                    updatePolicy.mutate({
                      id: editingPolicy.id,
                      data: editingPolicy,
                    })
                  }
                  disabled={updatePolicy.isPending}
                  className="bg-neon-mint text-slate-950 font-bold hover:bg-neon-mint/90"
                >
                  {updatePolicy.isPending ? "Saving..." : "Save Policy Changes"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-500 py-20">
              <Shield size={32} className="mx-auto mb-2 opacity-40" />
              <p>Select a policy to edit or simulate authorization rules.</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Access Policy Modal */}
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Access Control Policy"
      >
        <div className="space-y-4">
          <Input
            label="Policy Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="e.g. Fine-Tuning Cluster Execution Policy"
          />

          <Input
            label="Description"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            placeholder="Enforces maximum 8 GPU allocation per training run"
          />

          <Select
            label="Effect"
            options={[
              { value: "allow", label: "Allow (Permit Actions)" },
              { value: "deny", label: "Deny (Explicit Block)" },
            ]}
            value={newEffect}
            onChange={(e) => setNewEffect(e.target.value as any)}
          />

          <Input
            label="Actions (comma-separated)"
            value={newActions}
            onChange={(e) => setNewActions(e.target.value)}
            placeholder="create:experiment, read:dataset, *"
          />

          <Input
            label="Target Resources (comma-separated)"
            value={newResources}
            onChange={(e) => setNewResources(e.target.value)}
            placeholder="dataset:*, model:candidate-*"
          />

          <Input
            label="Principal Roles (comma-separated)"
            value={newSubjects}
            onChange={(e) => setNewSubjects(e.target.value)}
            placeholder="role:ml_engineer, role:researcher"
          />

          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createPolicy.mutate()}
              disabled={!newName.trim() || createPolicy.isPending}
            >
              {createPolicy.isPending ? "Creating..." : "Create Policy"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

