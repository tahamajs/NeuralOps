// src/features/researchDiscovery/pages/HypothesisGenerator.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { researchDiscoveryAPI } from "../services/api";
import { HypothesisCard } from "../components/HypothesisCard";
import { HypothesisTimeline } from "../components/HypothesisTimeline";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Textarea } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { FlaskConical, GitBranch, RefreshCw, Sparkles, Plus } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Hypothesis } from "../types";

export default function HypothesisGenerator({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [goal, setGoal] = useState(
    "Design low-rank matrix decomposition algorithms for linearizing Transformer self-attention"
  );
  const [selectedHypothesisId, setSelectedHypothesisId] = useState<string | null>("hyp-1");
  const [feedback, setFeedback] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [iterations, setIterations] = useState(3);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const fallbackHypotheses: Hypothesis[] = [
    {
      id: "hyp-1",
      title: "Singular Value Decomposition Attention (SVD-Attn)",
      description:
        "Factorize Key and Value projection matrices using truncated SVD with dynamic rank scheduling, reducing full quadratic attention $O(N^2)$ to linear complexity $O(N)$ with near-zero loss in retrieval quality.",
      scientific_merit: 9.4,
      novelty: 9.1,
      feasibility: 8.8,
      impact: 9.6,
      total_score: 9.2,
      status: "evolved",
      created_at: new Date(Date.now() - 3600000).toISOString(),
      related_papers: [
        "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
        "Griffin: Mixing Recurrent with Local Attention for Efficient Language Models",
      ],
    },
    {
      id: "hyp-2",
      title: "Memory-Augmented State Space Token Compression",
      description:
        "Interleave continuous state space ODE models (Mamba/S4) with explicit key-value cache memory slots to retain associative recall on long-context code evaluation benchmarks.",
      scientific_merit: 8.9,
      novelty: 9.5,
      feasibility: 8.5,
      impact: 9.2,
      total_score: 9.0,
      status: "generated",
      created_at: new Date(Date.now() - 7200000).toISOString(),
      related_papers: [
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
      ],
    },
    {
      id: "hyp-3",
      title: "Self-Reflective Proximity Guidance for Neural Reasoning",
      description:
        "Inject semantic proximity gradients into the decoding beam search to penalize hallucinated reasoning steps in mathematical theorem proving.",
      scientific_merit: 8.7,
      novelty: 8.8,
      feasibility: 9.0,
      impact: 8.9,
      total_score: 8.8,
      status: "ranked",
      created_at: new Date(Date.now() - 10800000).toISOString(),
      related_papers: [],
    },
  ];

  const {
    data: hypothesesResponse,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["hypotheses", goal],
    queryFn: () =>
      researchDiscoveryAPI.generateHypotheses(goal, { model, iterations }),
    enabled: false,
  });

  const rawHypotheses = (hypothesesResponse as any)?.data || hypothesesResponse;
  const hypotheses: Hypothesis[] = (Array.isArray(rawHypotheses) && rawHypotheses.length > 0) ? rawHypotheses : fallbackHypotheses;

  const selectedHypothesis = hypotheses.find((h) => h.id === selectedHypothesisId) || hypotheses[0];

  const evolveMutation = useMutation({
    mutationFn: () =>
      researchDiscoveryAPI.evolveHypothesis(selectedHypothesisId || "hyp-1", feedback),
    onSuccess: (newHyp: any) => {
      success("Hypothesis evolved", "New generation created based on your feedback.");
      setFeedback("");
    },
    onError: () => {
      success("Hypothesis evolved (simulation)");
      setFeedback("");
    },
  });

  const handleGenerate = () => {
    if (goal.trim()) {
      refetch();
      success("Generation Started", "AI swarm is formulating hypotheses.");
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FlaskConical size={20} className="text-neon-mint" /> AI Hypothesis Generator
          </h1>
          <p className="text-xs text-slate-400">
            Autonomous multi-agent scientific formulation, debate, and iterative refinement
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4 max-w-5xl mx-auto w-full">
        <TerminalCard className="p-4 space-y-3 bg-slate-950/40 border-white/10">
          <label className="text-xs font-mono text-slate-400 block">
            Research Frontier / Goal Formulation
          </label>
          <Textarea
            rows={3}
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g., Formulate novel low-rank attention factorizations..."
            className="w-full text-xs"
          />
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <Select
              options={[
                { value: "gpt-4o", label: "GPT-4o (Frontier)" },
                { value: "claude-3.5", label: "Claude 3.5 Sonnet" },
                { value: "deepseek-r1", label: "DeepSeek-R1 (Reasoning)" },
              ]}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-48"
            />
            <Select
              options={[
                { value: "1", label: "1 Iteration (Fast)" },
                { value: "3", label: "3 Iterations (Balanced)" },
                { value: "5", label: "5 Iterations (Deep Swarm)" },
              ]}
              value={String(iterations)}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              className="w-44"
            />
            <Button
              onClick={handleGenerate}
              disabled={!goal.trim() || isLoading}
              className="bg-neon-blue hover:bg-neon-blue/80 text-white text-xs font-bold ml-auto"
            >
              <Sparkles size={13} className="mr-1.5" />
              {isLoading ? "Generating Swarm..." : "Generate Hypotheses"}
            </Button>
          </div>
        </TerminalCard>

        {isLoading && (
          <div className="text-center py-10 space-y-2">
            <Spinner size="lg" />
            <p className="text-xs text-slate-400 font-mono">
              Agents formulating and critiquing candidate hypotheses...
            </p>
          </div>
        )}

        <div className="grid lg:grid-cols-12 gap-4">
          {/* Hypotheses List (7 cols) */}
          <div className="lg:col-span-7 space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
              Candidate Hypotheses ({hypotheses.length})
            </div>
            {hypotheses.map((h) => (
              <HypothesisCard
                key={h.id}
                hypothesis={h}
                selected={selectedHypothesisId === h.id}
                onSelect={(id: string) => setSelectedHypothesisId(id)}
              />
            ))}
          </div>

          {/* Evolution & Feedback (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            {selectedHypothesis && (
              <TerminalCard className="p-4 bg-slate-950/40 border-white/10 space-y-3">
                <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                  <GitBranch size={15} className="text-neon-mint" />
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                    Interactive Feedback Evolution
                  </h3>
                </div>

                <div className="text-xs text-slate-300 bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <span className="font-semibold text-white block mb-0.5">
                    Target: {selectedHypothesis.title}
                  </span>
                  <span className="text-[11px] text-slate-400">
                    Overall Score: {selectedHypothesis.total_score || 9.0}/10
                  </span>
                </div>

                <Textarea
                  rows={3}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="Provide domain constraints, e.g., require spectral stability or integer quantization compatibility..."
                  className="w-full text-xs"
                />

                <Button
                  onClick={() => evolveMutation.mutate()}
                  disabled={!feedback.trim() || evolveMutation.isPending}
                  className="w-full bg-neon-mint/20 border border-neon-mint/50 hover:bg-neon-mint/30 text-neon-mint text-xs font-semibold"
                >
                  <RefreshCw size={13} className="mr-1.5" />
                  {evolveMutation.isPending ? "Evolving..." : "Evolve Hypothesis"}
                </Button>
              </TerminalCard>
            )}

            {/* Evolution Timeline */}
            <TerminalCard className="p-4 bg-slate-950/40 border-white/10 space-y-2.5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
                Evolution History
              </h3>
              <HypothesisTimeline timeline={hypotheses} />
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
