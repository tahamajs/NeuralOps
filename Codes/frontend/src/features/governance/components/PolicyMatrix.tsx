import { Input } from "../../../components/ui/Input";
import { AccessPolicy } from "../types";

export function PolicyMatrix({
  policy,
  onChange,
}: {
  policy: AccessPolicy;
  onChange: (policy: AccessPolicy) => void;
}) {
  const updateActions = (value: string) =>
    onChange({ ...policy, actions: value.split(",").map((s) => s.trim()) });
  const updateResources = (value: string) =>
    onChange({ ...policy, resources: value.split(",").map((s) => s.trim()) });
  const updateSubjects = (value: string) =>
    onChange({ ...policy, subjects: value.split(",").map((s) => s.trim()) });
  const updatePriority = (value: number) =>
    onChange({ ...policy, priority: value });

  return (
    <div className="space-y-3 p-4 bg-white/5 rounded">
      <div>
        <label className="text-xs text-slate-400 mb-1 block">
          Actions (comma separated)
        </label>
        <Input
          value={policy.actions.join(", ")}
          onChange={(e) => updateActions(e.target.value)}
          placeholder="create, read, update, delete"
        />
      </div>
      <div>
        <label className="text-xs text-slate-400 mb-1 block">
          Resources (comma separated)
        </label>
        <Input
          value={policy.resources.join(", ")}
          onChange={(e) => updateResources(e.target.value)}
          placeholder="dataset:*, model:prod-*"
        />
      </div>
      <div>
        <label className="text-xs text-slate-400 mb-1 block">
          Subjects (comma separated)
        </label>
        <Input
          value={policy.subjects.join(", ")}
          onChange={(e) => updateSubjects(e.target.value)}
          placeholder="role:admin, user:alice"
        />
      </div>
      <div>
        <label className="text-xs text-slate-400 mb-1 block">
          Priority (lower = higher priority)
        </label>
        <Input
          type="number"
          value={policy.priority}
          onChange={(e) => updatePriority(parseInt(e.target.value))}
        />
      </div>
    </div>
  );
}
