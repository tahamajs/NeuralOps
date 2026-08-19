// src/features/llmHub/pages/GuardrailEditor.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Textarea } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Switch } from "../../../components/ui/Switch";
import { Select } from "../../../components/ui/Select";
import { Shield, Plus, Trash2, TestTube } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function GuardrailEditor({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [testText, setTestText] = useState("");
  const [testResult, setTestResult] = useState<any>(null);

  const [newName, setNewName] = useState("");
  const [type, setType] = useState("input");
  const [action, setAction] = useState("block");
  const [pattern, setPattern] = useState("");
  const [severity, setSeverity] = useState("medium");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: guardrails, isLoading } = useQuery({
    queryKey: ["guardrails"],
    queryFn: () => llmHubAPI.getGuardrails(),
  });
  const createGuardrail = useMutation({
    mutationFn: (data: any) => llmHubAPI.createGuardrail(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["guardrails"] });
      setShowCreateModal(false);
      success("Guardrail created");
    },
  });
  const deleteGuardrail = useMutation({
    mutationFn: (id: string) => llmHubAPI.deleteGuardrail(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["guardrails"] }),
  });
  const [selectedRuleId, setSelectedRuleId] = useState<string>("");

  const testGuardrail = useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      llmHubAPI.testGuardrail(id, text),
    onSuccess: (res: any) => {
      setTestResult(res?.data || res);
      success("Guardrail test completed");
    },
    onError: () => error("Guardrail test failed"),
  });

  if (isLoading) return <Spinner />;

  const activeTestId = selectedRuleId || guardrails?.[0]?.id || "default";

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-red-400" /> Guardrail Editor
          </h1>
          <p className="text-xs text-slate-400">
            Content safety rules for LLM inputs/outputs
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> Add Rule
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="space-y-3">
          {guardrails?.map((g) => (
            <TerminalCard
              key={g.id}
              className={`p-3 transition-colors ${selectedRuleId === g.id ? "border-neon-blue" : ""}`}
            >
              <div className="flex justify-between items-start">
                <div
                  className="cursor-pointer flex-1"
                  onClick={() => setSelectedRuleId(g.id)}
                >
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="font-bold text-white">{g.name}</h3>
                    <Badge variant={g.type as any}>{g.type}</Badge>
                    <Badge variant={g.action === "block" ? "failed" : "warning"}>
                      {g.action}
                    </Badge>
                    <Badge
                      variant={g.severity === "high" ? "failed" : "secondary"}
                    >
                      {g.severity}
                    </Badge>
                  </div>
                  <div className="text-xs font-mono text-slate-400 mt-1">
                    Pattern: {g.pattern}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={selectedRuleId === g.id ? "primary" : "ghost"}
                    onClick={() => {
                      setSelectedRuleId(g.id);
                      testGuardrail.mutate({ id: g.id, text: testText || "sample prompt test" });
                    }}
                  >
                    <TestTube size={14} className="mr-1" /> Test
                  </Button>
                  <button
                    onClick={() => deleteGuardrail.mutate(g.id)}
                    className="p-1 hover:bg-white/5 rounded text-red-400 hover:text-red-300"
                    title="Delete rule"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </TerminalCard>
          ))}
        </div>

        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
            <TestTube size={16} className="text-neon-mint" /> Test Guardrail Workbench
          </h3>
          <p className="text-xs text-slate-400 mb-2">
            Testing against rule: <span className="text-white font-mono">{guardrails?.find(g => g.id === activeTestId)?.name || "All Rules"}</span>
          </p>
          <Textarea
            rows={3}
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter prompt or response text to test against guardrails..."
          />
          <div className="flex gap-2 mt-2">
            <Button
              size="sm"
              onClick={() =>
                testGuardrail.mutate({ id: activeTestId, text: testText || "Test string" })
              }
              disabled={testGuardrail.isPending}
            >
              {testGuardrail.isPending ? "Testing..." : "Run Guardrail Check"}
            </Button>
          </div>
          {testResult && (
            <div className={`mt-3 p-3 rounded text-sm border ${testResult.flagged ? "bg-rose-950/40 border-rose-500/30 text-rose-200" : "bg-emerald-950/40 border-emerald-500/30 text-emerald-200"}`}>
              <div className="font-bold flex items-center gap-2">
                {testResult.flagged ? "⚠️ Violation Flagged" : "✅ Safe (Passed Guardrail)"}
                {testResult.confidence && (
                  <span className="text-xs font-normal opacity-75">
                    ({(testResult.confidence * 100).toFixed(0)}% confidence)
                  </span>
                )}
              </div>
              {testResult.matches && testResult.matches.length > 0 && (
                <div className="text-xs mt-1.5 space-y-1">
                  <div className="font-semibold text-slate-400">Triggered Patterns:</div>
                  {testResult.matches.map((m: string, idx: number) => (
                    <div key={idx} className="font-mono bg-black/40 px-2 py-0.5 rounded">
                      {m}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </TerminalCard>
      </div>
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Guardrail Rule"
      >
        <div className="space-y-4">
          <Input
            label="Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <Select
            label="Type"
            options={[
              { value: "input", label: "Input" },
              { value: "output", label: "Output" },
              { value: "both", label: "Both" },
            ]}
            value={type}
            onChange={(e) => setType(e.target.value)}
          />
          <Select
            label="Action"
            options={[
              { value: "block", label: "Block" },
              { value: "warn", label: "Warn" },
              { value: "replace", label: "Replace" },
            ]}
            value={action}
            onChange={(e) => setAction(e.target.value)}
          />
          <Input
            label="Pattern (regex)"
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            placeholder="violence|hate|pii"
          />
          <Select
            label="Severity"
            options={[
              { value: "low", label: "Low" },
              { value: "medium", label: "Medium" },
              { value: "high", label: "High" },
            ]}
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
          />
          <Button
            onClick={() =>
              createGuardrail.mutate({
                name: newName,
                type,
                action,
                pattern,
                severity,
              })
            }
          >
            Create
          </Button>
        </div>
      </Modal>
    </div>
  );
}
