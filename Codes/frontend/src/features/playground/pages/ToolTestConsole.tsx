// src/features/playground/pages/ToolTestConsole.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { playgroundAPI } from "../services/api";
import { ToolCallPanel } from "../components/ToolCallPanel";
import { ToolResultCard } from "../components/ToolResultCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Wrench, Play, Code2, CheckCircle, AlertCircle } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

const SAMPLE_PAYLOADS: Record<string, string> = {
  svd_tensor_compressor: JSON.stringify(
    { tensor_shape: [768, 768], rank_k: 64, target_variance: 0.99 },
    null,
    2,
  ),
  arxiv_paper_search: JSON.stringify(
    { query: "linear attention low rank approximation", max_results: 5 },
    null,
    2,
  ),
  code_interpreter: JSON.stringify(
    { code: "import numpy as np\nA = np.random.randn(100, 100)\nU, S, Vt = np.linalg.svd(A)\nprint('Singular values top 5:', S[:5])" },
    null,
    2,
  ),
  execute_python_sandbox: JSON.stringify(
    { code: "import torch\nq = torch.randn(2, 8, 128, 64)\nk = torch.randn(2, 8, 128, 64)\nv = torch.randn(2, 8, 128, 64)\nout = torch.nn.functional.scaled_dot_product_attention(q, k, v)\nprint('Output shape:', out.shape)" },
    null,
    2,
  ),
};

export default function ToolTestConsole({ onClose }: { onClose?: () => void }) {
  const [selectedTool, setSelectedTool] = useState("");
  const [inputJson, setInputJson] = useState("{\n  \"query\": \"SVD linear attention\"\n}");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const { success, error: toastError } = useToast();

  const { data: tools, isLoading } = useQuery({
    queryKey: ["playground-tools"],
    queryFn: () => playgroundAPI.getTools(),
  });

  useEffect(() => {
    if (tools && tools.length > 0 && !selectedTool) {
      const defaultTool = tools[0].name;
      setSelectedTool(defaultTool);
      if (SAMPLE_PAYLOADS[defaultTool]) {
        setInputJson(SAMPLE_PAYLOADS[defaultTool]);
      }
    }
  }, [tools, selectedTool]);

  const handleToolChange = (toolName: string) => {
    setSelectedTool(toolName);
    setResult(null);
    setJsonError(null);
    if (SAMPLE_PAYLOADS[toolName]) {
      setInputJson(SAMPLE_PAYLOADS[toolName]);
    } else {
      setInputJson(JSON.stringify({ query: "example input" }, null, 2));
    }
  };

  const testMutation = useMutation({
    mutationFn: (parsedInput: any) =>
      playgroundAPI.testTool(selectedTool, parsedInput),
    onSuccess: (res: any) => {
      setResult(res?.data || res);
      success("Tool executed successfully");
    },
    onError: () => toastError("Tool test failed"),
  });

  const handleTest = () => {
    if (!selectedTool) return;
    try {
      const parsed = JSON.parse(inputJson);
      setJsonError(null);
      testMutation.mutate(parsed);
    } catch (e: any) {
      setJsonError(`Invalid JSON: ${e.message}`);
      toastError("JSON Syntax Error", e.message);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Wrench size={20} className="text-neon-mint" /> Tool Test Console
          </h1>
          <p className="text-xs text-slate-400">
            Interactive function-calling verification & parameter inspector
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {isLoading ? (
          <Spinner />
        ) : (
          <TerminalCard className="p-4 space-y-4">
            <div className="flex justify-between items-end gap-3 flex-wrap">
              <div className="flex-1 min-w-[200px]">
                <Select
                  label="Target Tool"
                  options={
                    tools?.map((t) => ({ value: t.name, label: t.name })) || [
                      { value: "svd_tensor_compressor", label: "svd_tensor_compressor" },
                      { value: "arxiv_paper_search", label: "arxiv_paper_search" },
                      { value: "code_interpreter", label: "code_interpreter" },
                    ]
                  }
                  value={selectedTool}
                  onChange={(e) => handleToolChange(e.target.value)}
                />
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  if (selectedTool && SAMPLE_PAYLOADS[selectedTool]) {
                    setInputJson(SAMPLE_PAYLOADS[selectedTool]);
                    setJsonError(null);
                  }
                }}
              >
                <Code2 size={14} className="mr-1" /> Load Sample Input
              </Button>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs text-slate-400">
                  Input Arguments (JSON)
                </label>
                {jsonError && (
                  <span className="text-xs text-rose-400 flex items-center gap-1">
                    <AlertCircle size={12} /> {jsonError}
                  </span>
                )}
              </div>
              <textarea
                rows={7}
                value={inputJson}
                onChange={(e) => {
                  setInputJson(e.target.value);
                  try {
                    JSON.parse(e.target.value);
                    setJsonError(null);
                  } catch (err: any) {
                    setJsonError(err.message);
                  }
                }}
                className={`w-full bg-slate-900/60 border rounded p-3 font-mono text-xs text-slate-200 outline-none transition-colors ${
                  jsonError ? "border-rose-500/50" : "border-white/10 focus:border-neon-blue/50"
                }`}
              />
            </div>

            <Button
              onClick={handleTest}
              disabled={!selectedTool || !!jsonError || testMutation.isPending}
            >
              <Play size={14} className="mr-1" />{" "}
              {testMutation.isPending ? "Executing..." : "Run Tool Function"}
            </Button>
          </TerminalCard>
        )}

        {result && <ToolResultCard result={result} />}
      </div>
    </div>
  );
}
