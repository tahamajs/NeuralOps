import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { codingAssistantAPI } from "../services/api";
import { useCodingAssistantWebSocket } from "../hooks/useCodingAssistantWebSocket";
import { ChatPanel } from "../components/ChatPanel";
import { FileBrowser } from "../components/FileBrowser";
import { CodeEditorPanel } from "../components/CodeEditorPanel";
import { CodeDiffViewer } from "../components/CodeDiffViewer";
import { ExplanationPanel } from "../components/ExplanationPanel";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Message, GeneratedFile, CodeSuggestion } from "../types";
import { Code2, Sparkles, BookOpen, Layers, GitCompare } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function CodeAssistantPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>("code-sess-svd-01");
  const [activeCenterTab, setActiveCenterTab] = useState("editor");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "m-1",
      role: "assistant",
      content:
        "Hello! I am your AI Research Coding Copilot. I can help write custom attention CUDA/PyTorch modules, formulate Optuna sweeps, or inspect algorithmic complexity.",
      timestamp: new Date(),
    },
  ]);

  const defaultFiles: GeneratedFile[] = [
    {
      id: "f-1",
      name: "svd_linear_attention.py",
      language: "python",
      content: `import torch
import torch.nn as nn

class SVDLinearAttention(nn.Module):
    """
    Sub-quadratic O(N) linear attention using truncated SVD low-rank factorizations.
    """
    def __init__(self, dim: int = 768, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # q: [B, T, D]
        q = torch.relu(self.q_proj(x))
        # k_comp: [B, T, R]
        k_comp = self.u_proj(x)
        # Linear aggregation along sequence length T: O(T * R * D)
        kv = torch.einsum("btr, btd -> brd", k_comp, x)
        out = torch.einsum("btd, brd -> btd", q, kv)
        return out
`,
    },
    {
      id: "f-2",
      name: "train_hpo_sweep.py",
      language: "python",
      content: `import optuna
from svd_linear_attention import SVDLinearAttention

def objective(trial):
    rank = trial.suggest_categorical("svd_rank", [32, 64, 128])
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    # Model training stub
    return 1.842 - (rank / 1000.0)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best parameters:", study.best_params)
`,
    },
    {
      id: "f-3",
      name: "model_config.json",
      language: "json",
      content: `{
  "model_name": "bert-svd-linear-base",
  "hidden_size": 768,
  "num_attention_heads": 12,
  "svd_rank": 64,
  "max_position_embeddings": 32768,
  "activation": "relu"
}`,
    },
  ];

  const [files, setFiles] = useState<GeneratedFile[]>(defaultFiles);
  const [currentFileId, setCurrentFileId] = useState<string>("f-1");
  const [currentFileContent, setCurrentFileContent] = useState("");
  const [currentFileLanguage, setCurrentFileLanguage] = useState("python");
  const [isSaving, setIsSaving] = useState(false);
  const [currentDiff, setCurrentDiff] = useState<CodeSuggestion | null>(null);
  const { success } = useToast();

  const currentFile = files.find((f) => f.id === currentFileId) || files[0];

  useEffect(() => {
    if (currentFile) {
      setCurrentFileContent(currentFile.content || "");
      setCurrentFileLanguage(currentFile.language || "python");
    }
  }, [currentFileId, files]);

  const {
    isConnected,
    streamingContent,
    suggestions,
    sendMessage,
  } = useCodingAssistantWebSocket(sessionId);

  const handleSendMessage = (content: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content,
        timestamp: new Date(),
      },
    ]);
    sendMessage(content);
  };

  const handleSaveFile = (content: string) => {
    setIsSaving(true);
    setFiles((prev) =>
      prev.map((f) => (f.id === currentFileId ? { ...f, content } : f))
    );
    setTimeout(() => {
      setIsSaving(false);
      success("File saved successfully");
    }, 300);
  };

  const handleCreateFile = (name: string) => {
    const ext = name.split(".").pop() || "py";
    const lang = ext === "json" ? "json" : ext === "ts" || ext === "tsx" ? "typescript" : "python";
    const newFile: GeneratedFile = {
      id: `f-${Date.now().toString(36)}`,
      name,
      language: lang,
      content: `# ${name}\n`,
    };
    setFiles((prev) => [...prev, newFile]);
    setCurrentFileId(newFile.id);
    success(`Created ${name}`);
  };

  const handleDeleteFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
    if (currentFileId === id) {
      const remaining = files.filter((f) => f.id !== id);
      if (remaining.length > 0) setCurrentFileId(remaining[0].id);
    }
    success("File deleted");
  };

  const handleApplySuggestion = (suggestion: CodeSuggestion) => {
    if (suggestion.modifiedCode) {
      setCurrentFileContent(suggestion.modifiedCode);
      handleSaveFile(suggestion.modifiedCode);
      setCurrentDiff(suggestion);
      setActiveCenterTab("diff");
      success("Applied code suggestion to editor");
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Code2 size={20} className="text-neon-mint" /> AI Research Coding Copilot
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Interactive PyTorch kernel generation, Optuna sweeps, and structural diff analysis
          </p>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>

      {/* Main 3-Column Layout */}
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-12 gap-4 h-full">
          {/* Column 1: File Browser (3 cols) */}
          <div className="lg:col-span-3 h-full overflow-hidden">
            <FileBrowser
              files={files}
              currentFileId={currentFileId}
              onSelectFile={setCurrentFileId}
              onDeleteFile={handleDeleteFile}
              onCreateFile={handleCreateFile}
            />
          </div>

          {/* Column 2: Code Editor & Diff / Insights (5 cols) */}
          <div className="lg:col-span-5 h-full flex flex-col overflow-hidden">
            <TerminalCard className="h-full flex flex-col p-2 bg-slate-950/40">
              <Tabs
                value={activeCenterTab}
                onValueChange={setActiveCenterTab}
                className="h-full flex flex-col"
              >
                <TabList className="shrink-0 mb-2">
                  <Tab value="editor">
                    <Code2 size={13} className="mr-1" /> Editor ({currentFile?.name})
                  </Tab>
                  <Tab value="diff">
                    <GitCompare size={13} className="mr-1" /> Diff Viewer
                  </Tab>
                  <Tab value="insights">
                    <BookOpen size={13} className="mr-1" /> Optimization Insights
                  </Tab>
                </TabList>

                <TabPanel value="editor" className="flex-1 min-h-[450px]">
                  <CodeEditorPanel
                    content={currentFileContent}
                    language={currentFileLanguage}
                    onSave={handleSaveFile}
                    isSaving={isSaving}
                  />
                </TabPanel>

                <TabPanel value="diff" className="flex-1 overflow-y-auto min-h-[450px]">
                  <CodeDiffViewer
                    originalCode={currentFile?.content}
                    modifiedCode={currentDiff?.modifiedCode || currentFileContent}
                    diffText={currentDiff?.diff}
                  />
                </TabPanel>

                <TabPanel value="insights" className="flex-1 overflow-y-auto min-h-[450px]">
                  <ExplanationPanel
                    title="Low-Rank Linear Attention Complexity Analysis"
                    explanation="By utilizing singular value truncation with rank r=64 on embedding dimension d=768, the space complexity shrinks from quadratic O(N^2) to linear O(N * r). Arithmetic intensity shifts from memory-bound softmax compute to tensor contractions."
                    highlights={[
                      "Time Complexity: O(N * r * d)",
                      "Memory Footprint: O(N * d + r * d)",
                      "Backward Gradient Pass: Bounded O(r * d)",
                      "Singular Energy Retained: 99.1%",
                    ]}
                    suggestions={[
                      "Compile with torch.compile(mode='max-autotune') for CUDA graph fusion.",
                      "Allocate higher rank (r=128) in the top 3 Transformer layers for complex semantic retrieval.",
                      "Use FP16/BF16 tensor contractions on Tensor Cores to reach peak FLOPS utilization.",
                    ]}
                  />
                </TabPanel>
              </Tabs>
            </TerminalCard>
          </div>

          {/* Column 3: Copilot Chat Console (4 cols) */}
          <div className="lg:col-span-4 h-full overflow-hidden">
            <ChatPanel
              messages={messages}
              onSendMessage={handleSendMessage}
              isStreaming={!!streamingContent}
              streamingContent={streamingContent}
              suggestions={suggestions}
              onApplySuggestion={handleApplySuggestion}
              isConnected={isConnected}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
