import { Input } from "../../../components/ui/Input";
import { AccessPolicy } from "../types";
import { Plus, Tag, Shield, Database, Cpu } from "lucide-react";

export function PolicyMatrix({
  policy,
  onChange,
}: {
  policy: AccessPolicy;
  onChange: (policy: AccessPolicy) => void;
}) {
  const updateActions = (value: string) =>
    onChange({ ...policy, actions: value.split(",").map((s) => s.trim()).filter(Boolean) });
  const updateResources = (value: string) =>
    onChange({ ...policy, resources: value.split(",").map((s) => s.trim()).filter(Boolean) });
  const updateSubjects = (value: string) =>
    onChange({ ...policy, subjects: value.split(",").map((s) => s.trim()).filter(Boolean) });
  const updatePriority = (value: number) =>
    onChange({ ...policy, priority: isNaN(value) ? 50 : value });

  const addActionPill = (action: string) => {
    if (!policy.actions.includes(action)) {
      onChange({ ...policy, actions: [...policy.actions, action] });
    }
  };

  const addResourcePill = (res: string) => {
    if (!policy.resources.includes(res)) {
      onChange({ ...policy, resources: [...policy.resources, res] });
    }
  };

  const addSubjectPill = (subj: string) => {
    if (!policy.subjects.includes(subj)) {
      onChange({ ...policy, subjects: [...policy.subjects, subj] });
    }
  };

  return (
    <div className="space-y-4 p-4 bg-slate-900/90 rounded border border-white/10 text-xs">
      {/* Actions */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <label className="font-semibold text-slate-300">
            Allowed Actions (comma-separated or quick-add)
          </label>
        </div>
        <Input
          value={policy.actions.join(", ")}
          onChange={(e) => updateActions(e.target.value)}
          placeholder="create:model, read:dataset, export:audit_log, *"
          className="text-xs font-mono"
        />
        <div className="flex flex-wrap gap-1 items-center pt-0.5">
          <span className="text-[10px] text-slate-500 mr-1">Quick Add:</span>
          {["*", "read:*", "create:*", "delete:*", "export:audit_log", "promote:production", "trigger:snapshot"].map((act) => (
            <button
              key={act}
              type="button"
              onClick={() => addActionPill(act)}
              className="px-1.5 py-0.5 rounded bg-white/5 hover:bg-neon-blue/20 text-slate-400 hover:text-white border border-white/5 font-mono text-[10px]"
            >
              +{act}
            </button>
          ))}
        </div>
      </div>

      {/* Target Resources */}
      <div className="space-y-1.5">
        <label className="font-semibold text-slate-300">
          Target Resources (comma-separated)
        </label>
        <Input
          value={policy.resources.join(", ")}
          onChange={(e) => updateResources(e.target.value)}
          placeholder="dataset:*, model:prod-*, governance:*, *"
          className="text-xs font-mono"
        />
        <div className="flex flex-wrap gap-1 items-center pt-0.5">
          <span className="text-[10px] text-slate-500 mr-1">Quick Add:</span>
          {["*", "model:*", "model:prod-*", "dataset:*", "governance:*", "experiment:*"].map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => addResourcePill(r)}
              className="px-1.5 py-0.5 rounded bg-white/5 hover:bg-neon-mint/20 text-slate-400 hover:text-white border border-white/5 font-mono text-[10px]"
            >
              +{r}
            </button>
          ))}
        </div>
      </div>

      {/* Subjects & Roles */}
      <div className="space-y-1.5">
        <label className="font-semibold text-slate-300">
          Subjects / Principal Roles (comma-separated)
        </label>
        <Input
          value={policy.subjects.join(", ")}
          onChange={(e) => updateSubjects(e.target.value)}
          placeholder="role:superadmin, role:ml_engineer, role:auditor, user:alice"
          className="text-xs font-mono"
        />
        <div className="flex flex-wrap gap-1 items-center pt-0.5">
          <span className="text-[10px] text-slate-500 mr-1">Quick Add:</span>
          {["role:superadmin", "role:ml_engineer", "role:compliance_auditor", "role:researcher", "role:viewer"].map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => addSubjectPill(s)}
              className="px-1.5 py-0.5 rounded bg-white/5 hover:bg-purple-500/20 text-slate-400 hover:text-white border border-white/5 font-mono text-[10px]"
            >
              +{s}
            </button>
          ))}
        </div>
      </div>

      {/* Priority and ABAC Conditions */}
      <div className="grid grid-cols-2 gap-3 pt-1">
        <div>
          <label className="font-semibold text-slate-300 mb-1 block">
            Priority (lower number = higher precedence)
          </label>
          <Input
            type="number"
            value={policy.priority}
            onChange={(e) => updatePriority(parseInt(e.target.value))}
            className="text-xs font-mono"
          />
        </div>
        <div>
          <label className="font-semibold text-slate-300 mb-1 block">
            ABAC Context Conditions (JSON)
          </label>
          <input
            className="w-full bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-neon-blue font-mono focus:outline-none focus:border-neon-mint"
            value={JSON.stringify(policy.conditions || {})}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                onChange({ ...policy, conditions: parsed });
              } catch {}
            }}
            placeholder='{"mfa_verified": true}'
          />
        </div>
      </div>
    </div>
  );
}

