import { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import MonacoEditor from "../../../components/editor/MonacoEditor";
import {
  useFloatingWindowStore,
  WINDOW_TYPES,
} from "../../../components/floating/FloatingWindowManager";

// Templates
const TEMPLATES: Record<string, string> = {
  empty: `\\documentclass{article}
\\usepackage{graphicx}
\\title{My Document}
\\author{You}
\\date{\\today}
\\begin{document}
\\maketitle
\\section{Introduction}
Your content here.
\\end{document}`,

  glue: `\\documentclass{article}
\\usepackage{graphicx, booktabs}
\\title{GLUE Benchmark Experiment Report}
\\author{Transformer Toolkit}
\\date{\\today}
\\begin{document}
\\maketitle
\\section{Experimental Setup}
We evaluate on GLUE tasks: MRPC, SST-2, CoLA, QQP, MNLI.
\\section{Results}
\\begin{table}[h]
\\centering
\\begin{tabular}{lcc}
\\toprule
Model & Accuracy & F1 \\\\
\\midrule
Baseline & 0.892 & 0.901 \\\\
SVD (rank 64) & 0.885 & 0.894 \\\\
GoldFinch & 0.901 & 0.912 \\\\
\\bottomrule
\\end{tabular}
\\caption{Main results.}
\\end{table}
\\section{Conclusion}
Our proposed attention achieves competitive results with lower complexity.
\\end{document}`,

  svd: `\\documentclass{article}
\\usepackage{graphicx}
\\title{SVD Compression Analysis}
\\author{Transformer Toolkit}
\\date{\\today}
\\begin{document}
\\maketitle
\\section{Singular Value Spectra}
\\begin{figure}[h]
\\centering
\\includegraphics[width=0.7\\linewidth]{svd_spectrum.png}
\\caption{Singular value decay.}
\\end{figure}
\\section{Energy Retention}
The cumulative energy plot shows that rank 64 retains 95\\% of the information.
\\end{document}`,
};

const templateOptions = [
  { value: "empty", label: "Empty Document" },
  { value: "glue", label: "GLUE Experiment" },
  { value: "svd", label: "SVD Compression Report" },
  { value: "run", label: "Paper with Run Metrics" },
];

export default function PaperEditorPage() {
  const { runId } = useParams<{ runId?: string }>();
  const { addWindow } = useFloatingWindowStore();

  const [latexSource, setLatexSource] = useState(TEMPLATES.empty);
  const [compileStatus, setCompileStatus] = useState<
    "idle" | "compiling" | "success" | "error"
  >("idle");
  const [compileError, setCompileError] = useState("");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const editorRef = useRef<any>(null);

  // Fetch run metric-based template if runId exists
  const { data: run } = useQuery({
    queryKey: ["run", runId],
    queryFn: async () => (await api.get(`/runs/${runId}/`)).data,
    enabled: !!runId,
  });

  // Build run-based template
  useEffect(() => {
    if (run) {
      const metrics = run.final_metrics || {};
      const acc = metrics.eval_accuracy?.toFixed(4) ?? "N/A";
      const loss = metrics.eval_loss?.toFixed(4) ?? "N/A";
      const config = run.config || {};
      const runTemplate = `\\documentclass{article}
\\usepackage{graphicx, booktabs}
\\title{Experiment Report: ${config.task_name?.toUpperCase() || "Task"}}
\\author{Researcher}
\\date{\\today}
\\begin{document}
\\maketitle
\\section{Configuration}
\\begin{tabular}{ll}
Model: & ${config.model_name || "N/A"} \\\\
Attention: & ${config.attention_type || "full"} \\\\
Epochs: & ${config.epochs || "N/A"} \\\\
Batch size: & ${config.batch_size || "N/A"} \\\\
Learning rate: & ${config.learning_rate || "N/A"} \\\\
\\end{tabular}
\\section{Results}
\\begin{tabular}{lccc}
\\toprule
Metric & Accuracy & Loss & F1 \\\\
\\midrule
Value & ${acc} & ${loss} & ${metrics.eval_f1?.toFixed(4) ?? "N/A"} \\\\
\\bottomrule
\\end{tabular}
\\end{document}`;
      TEMPLATES.run = runTemplate;
    }
  }, [run]);

  // Compilation mutation (Mocked for standalone editor)
  const compileMut = useMutation({
    mutationFn: async (source: string) => {
      // Simulate compilation delay
      await new Promise((resolve) => setTimeout(resolve, 1500));
      return {
        pdf_url:
          "data:application/pdf;base64,JVBERi0xLjQKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCA2MTIgNzkyXS9QYXJlbnQgMiAwIFIvUmVzb3VyY2VzPDw+Pj4+ZW5kb2JqCnhyZWYKMCA0CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDUyIDAwMDAwIG4gCjAwMDAwMDAxMDEgMDAwMDAgbiAKdHJhaWxlcjw8L1NpemUgNC9Sb290IDEgMCBSPj4Kc3RhcnR4cmVmCjE3OAolJUVPRg==",
      };
    },
    onMutate: () => {
      setCompileStatus("compiling");
      setCompileError("");
    },
    onSuccess: (data) => {
      setPdfUrl(data.pdf_url);
      setCompileStatus("success");
    },
    onError: (err: any) => {
      setCompileStatus("error");
      setCompileError(err.message || "Compilation failed");
    },
  });

  // Save source to run (if runId)
  const saveMut = useMutation({
    mutationFn: async (source: string) => {
      await api.post(`/runs/${runId}/paper_source/`, source, {
        headers: { "Content-Type": "text/plain" },
      });
    },
    onSuccess: () => alert("LaTeX source saved to run."),
    onError: () => alert("Failed to save source."),
  });

  // Load source from run
  const loadSource = async () => {
    if (!runId) return;
    try {
      const res = await api.get(`/runs/${runId}/paper_source/`);
      setLatexSource(res.data);
    } catch {
      alert("No saved LaTeX found.");
    }
  };

  // Load source on mount if runId
  useEffect(() => {
    if (runId) loadSource();
  }, [runId]);

  // Template handler
  const handleTemplate = (value: string) => {
    if (value === "run" && runId) {
      // Already built TEMPLATES.run
      setLatexSource(TEMPLATES.run || TEMPLATES.empty);
    } else {
      setLatexSource(TEMPLATES[value] || TEMPLATES.empty);
    }
  };

  // Insert image (Monaco action)
  const insertImage = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = (e: any) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const base64 = ev.target?.result as string;
        const snippet = `\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{${base64}}
\\caption{${file.name}}
\\label{fig:${file.name.replace(/\./g, "")}}
\\end{figure}`;
        // Insert into Monaco via editor reference
        if (editorRef.current) {
          const selection = editorRef.current.getSelection();
          editorRef.current.executeEdits("insert-image", [
            { range: selection, text: snippet, forceMoveMarkers: true },
          ]);
        }
      };
      reader.readAsDataURL(file);
    };
    input.click();
  };

  // Compile button
  const handleCompile = () => {
    compileMut.mutate(latexSource);
  };

  return (
    <div className="flex flex-col h-full bg-cyber-900">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 bg-cyber-800 border-b border-white/5 flex-wrap">
        <Button
          size="sm"
          onClick={handleCompile}
          disabled={compileMut.isPending}
        >
          {compileMut.isPending ? "⏳ Compiling..." : "▶ Compile"}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          disabled={!pdfUrl}
          onClick={() => pdfUrl && window.open(pdfUrl, "_blank")}
        >
          📥 Download PDF
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => runId && saveMut.mutate(latexSource)}
          disabled={!runId}
        >
          💾 Save to Run
        </Button>
        <Button variant="ghost" size="sm" onClick={insertImage}>
          🖼️ Insert Image
        </Button>
        <Select
          options={templateOptions}
          value="empty"
          onChange={(e) => handleTemplate(e.target.value)}
          className="w-40"
        />
        <div className="flex-1" />
        <span
          className={`text-xs font-mono px-3 py-1 rounded-full ${
            compileStatus === "success"
              ? "bg-green-500/20 text-green-400"
              : compileStatus === "error"
                ? "bg-red-500/20 text-red-400"
                : compileStatus === "compiling"
                  ? "bg-yellow-500/20 text-yellow-400"
                  : "bg-white/5 text-slate-400"
          }`}
        >
          {compileStatus === "idle"
            ? "Ready"
            : compileStatus === "compiling"
              ? "Compiling..."
              : compileStatus === "success"
                ? "Compiled"
                : "Error"}
        </span>
      </div>

      {/* Error Panel */}
      {compileError && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 text-red-400 text-xs font-mono whitespace-pre-wrap max-h-24 overflow-auto">
          {compileError}
        </div>
      )}

      {/* Split Editor and Preview */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor Pane */}
        <div className="w-1/2 h-full border-r border-white/5">
          <div className="h-full">
            <MonacoEditor
              value={latexSource}
              onChange={(val) => setLatexSource(val || "")}
              editorRef={editorRef}
            />
          </div>
        </div>

        {/* Preview Pane */}
        <div className="w-1/2 h-full bg-slate-100">
          {pdfUrl ? (
            <iframe
              src={pdfUrl}
              className="w-full h-full border-0"
              title="PDF Preview"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <span className="text-4xl mb-4">📄</span>
              <p className="text-sm">Compile to see preview</p>
              <Button
                size="sm"
                variant="secondary"
                className="mt-3"
                onClick={handleCompile}
              >
                Compile PDF
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
