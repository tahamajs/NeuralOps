import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Textarea } from "../../../components/ui/Input";
import { FlaskConical, Plus, Trash2 } from "lucide-react";

export default function PromptTestingPage() {
  const [template, setTemplate] = useState("");
  const [testCases, setTestCases] = useState<Array<Record<string, string>>>([]);
  const [metric, setMetric] = useState("contains");
  const [results, setResults] = useState<any>(null);

  const testMut = useMutation({
    mutationFn: async () => {
      const res = await api.post("/agents/prompts/test/", {
        template,
        test_cases: testCases,
        metric,
      });
      return res.data;
    },
    onSuccess: setResults,
  });

  const addTestCase = () => {
    setTestCases([...testCases, {}]);
  };

  const updateTestCase = (idx: number, field: string, value: string) => {
    const updated = [...testCases];
    updated[idx] = { ...updated[idx], [field]: value };
    setTestCases(updated);
  };

  const removeTestCase = (idx: number) => {
    setTestCases(testCases.filter((_, i) => i !== idx));
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <FlaskConical size={24} className="text-neon-blue" /> Prompt Testing
        </h1>
      </div>

      <Card>
        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">
          Prompt Template (use {"{{variable}}"} syntax)
        </label>
        <Textarea
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder="Analyze experiment {{experiment_id}} for overfitting."
          rows={4}
        />
      </Card>

      <Card>
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-sm font-bold text-white">Test Cases</h3>
          <Button size="sm" variant="ghost" onClick={addTestCase}>
            <Plus size={14} className="mr-1" /> Add
          </Button>
        </div>
        {testCases.map((tc, idx) => (
          <div key={idx} className="flex gap-2 mb-2 items-start">
            <Input
              placeholder="Variable: value"
              value={Object.entries(tc)
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ")}
              onChange={(e) => {
                const parts = e.target.value.split(":");
                if (parts.length >= 2) {
                  updateTestCase(
                    idx,
                    parts[0].trim(),
                    parts.slice(1).join(":").trim(),
                  );
                }
              }}
              className="flex-1"
            />
            <Input
              placeholder="Expected output"
              value={tc.expected || ""}
              onChange={(e) => updateTestCase(idx, "expected", e.target.value)}
              className="flex-1"
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => removeTestCase(idx)}
            >
              <Trash2 size={14} />
            </Button>
          </div>
        ))}
        <Select
          label="Metric"
          options={[
            { value: "contains", label: "Contains" },
            { value: "exact_match", label: "Exact Match" },
            { value: "llm_judge", label: "LLM Judge" },
          ]}
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
        />
        <Button
          className="mt-3"
          onClick={() => testMut.mutate()}
          disabled={!template || testCases.length === 0}
        >
          Run Tests
        </Button>
      </Card>

      {results && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">
            Results ({results.total_tests} tests)
          </h3>
          <div className="space-y-3">
            {results.results.map((r: any, i: number) => (
              <div key={i} className="bg-cyber-800 p-3 rounded text-xs">
                <div className="flex justify-between mb-1">
                  <span className="text-white font-bold">
                    Test {r.test_case}
                  </span>
                  <span className="text-green-400">{r.status}</span>
                </div>
                <p className="text-slate-400 line-clamp-3">
                  {r.substituted_prompt}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
