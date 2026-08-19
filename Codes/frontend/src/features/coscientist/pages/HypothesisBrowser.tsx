// src/features/coscientist/pages/HypothesisBrowser.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { coscientistAPI } from "../services/api";
import { HypothesisCard } from "../components/HypothesisCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";
import { Search, Brain, Filter, Sparkles, Trophy, Plus } from "lucide-react";
import { Hypothesis } from "../types";

export default function HypothesisBrowser({
  onClose,
  onStartTournament,
}: {
  onClose?: () => void;
  onStartTournament?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("total_score");
  const [statusFilter, setStatusFilter] = useState("all");

  const { data: sessionsResponse, isLoading } = useQuery({
    queryKey: ["coscientist-sessions"],
    queryFn: () => coscientistAPI.listSessions(),
  });

  // Built-in sample hypotheses if backend has no sessions yet
  const fallbackHypotheses: Hypothesis[] = [
    {
      id: "hyp-1",
      title: "Singular Value Decomposition Attention (SVD-Attn)",
      description:
        "Factorize Key and Value projection matrices using truncated SVD with dynamic rank scheduling, reducing full quadratic attention $O(N^2)$ to linear complexity $O(N)$ with near-zero loss in retrieval quality.",
      scientific_merit: 9.4,
      novelty_score: 9.1,
      testability_score: 8.8,
      impact_score: 9.6,
      total_score: 9.2,
      elo_rating: 1420,
      generation_round: 3,
      status: "selected",
      created_at: new Date(Date.now() - 3600000).toISOString(),
      metrics: {
        citation_count: 14,
        relevant_papers: [
          "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
          "Griffin: Mixing Recurrent with Local Attention for Efficient Language Models",
        ],
      },
    },
    {
      id: "hyp-2",
      title: "Memory-Augmented State Space Token Compression",
      description:
        "Interleave continuous state space ODE models (Mamba/S4) with explicit key-value cache memory slots to retain associative recall on long-context code evaluation benchmarks.",
      scientific_merit: 8.9,
      novelty_score: 9.5,
      testability_score: 8.5,
      impact_score: 9.2,
      total_score: 9.0,
      elo_rating: 1380,
      generation_round: 2,
      status: "evolved",
      created_at: new Date(Date.now() - 7200000).toISOString(),
      metrics: {
        citation_count: 8,
        relevant_papers: [
          "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        ],
      },
    },
    {
      id: "hyp-3",
      title: "Self-Reflective Proximity Guidance for Neural Reasoning",
      description:
        "Inject semantic proximity gradients into the decoding beam search to penalize hallucinated reasoning steps in mathematical theorem proving.",
      scientific_merit: 8.7,
      novelty_score: 8.8,
      testability_score: 9.0,
      impact_score: 8.9,
      total_score: 8.8,
      elo_rating: 1310,
      generation_round: 2,
      status: "ranked",
      created_at: new Date(Date.now() - 10800000).toISOString(),
      metrics: {
        citation_count: 5,
      },
    },
    {
      id: "hyp-4",
      title: "Sparse Kernel Interpolation for Long-Sequence Multi-Modal Context",
      description:
        "Utilize adaptive random Fourier features with localized kernel decay to compress cross-modal image-text attention without retraining foundational weights.",
      scientific_merit: 8.2,
      novelty_score: 8.6,
      testability_score: 8.4,
      impact_score: 8.5,
      total_score: 8.4,
      elo_rating: 1250,
      generation_round: 1,
      status: "proposed",
      created_at: new Date(Date.now() - 14400000).toISOString(),
    },
  ];

  // Extract hypotheses safely
  const rawSessions = sessionsResponse?.data?.results || (Array.isArray(sessionsResponse?.data) ? sessionsResponse?.data : []);
  const extractedHypotheses: Hypothesis[] = rawSessions.flatMap((s: any) => s.hypotheses || []);

  const allHypotheses = extractedHypotheses.length > 0 ? extractedHypotheses : fallbackHypotheses;

  // Filter and Sort
  const filtered = allHypotheses.filter((h) => {
    const matchesSearch =
      (h.title || "").toLowerCase().includes(search.toLowerCase()) ||
      (h.description || "").toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === "all" || h.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const sorted = [...filtered].sort((a, b) => {
    const scoreA = Number((a as any)[sortBy] ?? (a as any).elo_rating ?? 0);
    const scoreB = Number((b as any)[sortBy] ?? (b as any).elo_rating ?? 0);
    return scoreB - scoreA;
  });

  // Aggregation metrics
  const avgScore =
    allHypotheses.length > 0
      ? (
          allHypotheses.reduce((acc, h) => acc + (Number(h.total_score) || 8.5), 0) /
          allHypotheses.length
        ).toFixed(1)
      : "8.5";

  const topElo = Math.max(
    ...allHypotheses.map((h) => Number(h.elo_rating || (h as any).elo_score || 1200))
  );

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Brain size={20} className="text-neon-mint" /> Hypothesis Browser
          </h1>
          <p className="text-xs text-slate-400">
            Explore, rank, and filter AI-generated research hypotheses
          </p>
        </div>
        <div className="flex gap-2">
          {onStartTournament && (
            <Button size="sm" onClick={onStartTournament} className="gap-1">
              <Trophy size={14} className="text-yellow-400" /> Start Tournament
            </Button>
          )}
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* KPI Row */}
        <div className="grid grid-cols-3 gap-3">
          <TerminalCard className="p-3">
            <div className="text-[11px] text-slate-400 font-mono">Total Hypotheses</div>
            <div className="text-xl font-bold text-white font-mono mt-0.5">
              {allHypotheses.length}
            </div>
          </TerminalCard>
          <TerminalCard className="p-3">
            <div className="text-[11px] text-slate-400 font-mono">Average Merit</div>
            <div className="text-xl font-bold text-neon-mint font-mono mt-0.5">
              {avgScore} / 10
            </div>
          </TerminalCard>
          <TerminalCard className="p-3">
            <div className="text-[11px] text-slate-400 font-mono">Top ELO Rating</div>
            <div className="text-xl font-bold text-yellow-400 font-mono mt-0.5">
              {topElo}
            </div>
          </TerminalCard>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-wrap gap-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search hypotheses by title, concept, or domain..."
              className="pl-9"
            />
          </div>
          <Select
            options={[
              { value: "total_score", label: "Sort: Total Score" },
              { value: "elo_rating", label: "Sort: ELO Rating" },
              { value: "scientific_merit", label: "Sort: Scientific Merit" },
              { value: "novelty_score", label: "Sort: Novelty" },
              { value: "impact_score", label: "Sort: Impact" },
            ]}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="w-48"
          />
          <Select
            options={[
              { value: "all", label: "All Statuses" },
              { value: "selected", label: "Selected / Finalist" },
              { value: "evolved", label: "Evolved" },
              { value: "ranked", label: "Ranked" },
              { value: "proposed", label: "Proposed" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-44"
          />
        </div>

        {/* Loading state */}
        {isLoading && (
          <div className="py-12 text-center">
            <Spinner size="lg" />
            <p className="text-xs text-slate-400 mt-2">Loading hypothesis catalogue...</p>
          </div>
        )}

        {/* Hypotheses Grid */}
        <div className="grid md:grid-cols-2 gap-3.5">
          {sorted.map((h) => (
            <HypothesisCard key={h.id} hypothesis={h} />
          ))}
          {sorted.length === 0 && !isLoading && (
            <div className="col-span-2 text-center py-12 text-slate-500 italic border border-dashed border-white/10 rounded-xl">
              No hypotheses matched your filter criteria.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
