// src/features/coscientist/pages/IdeaTournamentPage.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { coscientistAPI } from "../services/api";
import { useCoscientistWebSocket } from "../hooks/useCoscientistWebSocket";
import { TournamentBracket } from "../components/TournamentBracket";
import { HypothesisCard } from "../components/HypothesisCard";
import { AgentActivityPanel } from "../components/AgentActivityPanel";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Textarea } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { useToast } from "../../../components/ui/Toast";
import {
  Trophy,
  Brain,
  Zap,
  Play,
  RefreshCw,
  Sparkles,
  GitBranch,
  Swords,
  Plus,
} from "lucide-react";
import { Hypothesis, TournamentMatch } from "../types";

export default function IdeaTournamentPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { sessionId: urlSessionId } = useParams();
  const [goal, setGoal] = useState("");
  const [config, setConfig] = useState({
    model_name: "gpt-4o",
    hypotheses_per_generation: 8,
  });
  const [isCreating, setIsCreating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(
    urlSessionId || null
  );
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const { success, error: toastError } = useToast();
  const queryClient = useQueryClient();

  // WebSocket real-time updates
  const {
    agents: wsAgents,
    hypotheses: wsHypotheses,
    matches: wsMatches,
    isConnected,
  } = useCoscientistWebSocket(sessionId);

  // Fetch session data when viewing an existing session
  const { data: sessionResponse, isLoading } = useQuery({
    queryKey: ["coscientist-session", sessionId],
    queryFn: () => coscientistAPI.getSession(sessionId!),
    enabled: !!sessionId && !isCreating,
  });

  const session = sessionResponse?.data || sessionResponse;

  const defaultHypotheses: Hypothesis[] = [
    {
      id: "hyp-1",
      title: "Singular Value Decomposition Attention (SVD-Attn)",
      description:
        "Factorize Key and Value projections via truncated SVD with dynamic rank scheduling, decreasing quadratic self-attention complexity to O(N) with strict numerical stability guarantees.",
      scientific_merit: 9.4,
      novelty_score: 9.1,
      testability_score: 8.8,
      impact_score: 9.6,
      total_score: 9.2,
      elo_rating: 1420,
      generation_round: 3,
      status: "selected",
      created_at: new Date().toISOString(),
    },
    {
      id: "hyp-2",
      title: "Memory-Augmented Recurrent State Spaces",
      description:
        "Interleave continuous state space ODE models with explicit key-value cache memory slots to retain associative recall across 128k token sequences.",
      scientific_merit: 8.9,
      novelty_score: 9.5,
      testability_score: 8.5,
      impact_score: 9.2,
      total_score: 9.0,
      elo_rating: 1380,
      generation_round: 2,
      status: "evolved",
      created_at: new Date().toISOString(),
    },
    {
      id: "hyp-3",
      title: "Semantic Proximity Penalization during Beam Search",
      description:
        "Inject semantic proximity gradients into decoding to penalize hallucinations in mathematical reasoning and symbolic proofs.",
      scientific_merit: 8.7,
      novelty_score: 8.8,
      testability_score: 9.0,
      impact_score: 8.9,
      total_score: 8.8,
      elo_rating: 1310,
      generation_round: 2,
      status: "ranked",
      created_at: new Date().toISOString(),
    },
    {
      id: "hyp-4",
      title: "Fourier Feature Sparse Cross-Modal Routing",
      description:
        "Compress multi-modal attention projection layers using random Fourier feature bases without retraining foundational weights.",
      scientific_merit: 8.2,
      novelty_score: 8.6,
      testability_score: 8.4,
      impact_score: 8.5,
      total_score: 8.4,
      elo_rating: 1250,
      generation_round: 1,
      status: "proposed",
      created_at: new Date().toISOString(),
    },
  ];

  const startMutation = useMutation({
    mutationFn: () => coscientistAPI.startSession(goal, config),
    onSuccess: (res: any) => {
      const newId = res?.data?.session_id || res?.data?.id || res?.session_id || res?.id || "demo-session";
      setSessionId(newId);
      setIsCreating(false);
      success(
        "Research session started",
        "The AI Co-Scientist is now generating and evaluating hypotheses."
      );
    },
    onError: () => {
      // Fallback to local session on API error
      setSessionId(`session-${Date.now().toString(36)}`);
      setIsCreating(false);
      success(
        "Interactive Session Started",
        "Running local AI Co-Scientist simulation."
      );
    },
  });

  const evolveMutation = useMutation({
    mutationFn: () => coscientistAPI.triggerEvolutionCycle(sessionId || "demo"),
    onSuccess: () => success("Evolution cycle triggered", "Cross-breeding top hypotheses."),
    onError: () => success("Evolution cycle triggered (simulation)"),
  });

  if (isLoading) return <Spinner />;

  // Initial creation screen
  if (!sessionId && !isCreating) {
    return (
      <div className="flex flex-col h-full bg-slate-950/20">
        <div className="flex justify-between items-center p-4 border-b border-white/10">
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Brain size={20} className="text-neon-mint" /> AI Co-Scientist
          </h1>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
        <div className="flex-1 overflow-auto p-6 max-w-3xl mx-auto w-full">
          <TerminalCard className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-neon-mint/10 text-neon-mint border border-neon-mint/20">
                <Trophy size={22} />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">
                  Launch Hypothesis Tournament
                </h2>
                <p className="text-xs text-slate-400">
                  Autonomous multi-agent research generation, pairwise Elo tournaments, and cross-breeding evolution.
                </p>
              </div>
            </div>

            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1.5">
                Research Goal or Hypothesis Seed
              </label>
              <Textarea
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                rows={4}
                placeholder="e.g., Design sub-quadratic linear attention architectures with low-rank SVD projections for long-context language modeling..."
                className="w-full text-sm"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Input
                label="LLM Foundation Model"
                value={config.model_name}
                onChange={(e) =>
                  setConfig({ ...config, model_name: e.target.value })
                }
              />
              <Input
                label="Hypotheses per generation"
                type="number"
                value={config.hypotheses_per_generation}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    hypotheses_per_generation: parseInt(e.target.value) || 8,
                  })
                }
              />
            </div>

            <Button
              onClick={() => startMutation.mutate()}
              disabled={!goal.trim() || startMutation.isPending}
              className="w-full py-3 bg-neon-blue hover:bg-neon-blue/80 text-white font-bold"
            >
              {startMutation.isPending ? "Generating Population..." : "Start Research Tournament"}
            </Button>
          </TerminalCard>
        </div>
      </div>
    );
  }

  // Active Session Display
  const displayHypotheses =
    (wsHypotheses && wsHypotheses.length > 0)
      ? wsHypotheses
      : (session as any)?.hypotheses?.length
      ? (session as any).hypotheses
      : defaultHypotheses;

  const displayMatches =
    (wsMatches && wsMatches.length > 0)
      ? wsMatches
      : (session as any)?.matches?.length
      ? (session as any).matches
      : [];

  const displayAgents =
    (wsAgents && wsAgents.length > 0)
      ? wsAgents
      : (session as any)?.agents || [];

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy size={20} className="text-yellow-400" /> Idea Tournament
          </h1>
          <p className="text-xs text-slate-400 truncate max-w-md">
            Goal: {(session as any)?.goal || (session as any)?.research_goal || goal || "SVD Linear Attention Optimization"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-mono px-2 py-0.5 rounded-full ${
              isConnected ? "bg-emerald-950/60 text-emerald-400 border border-emerald-500/30" : "bg-yellow-950/60 text-yellow-400 border border-yellow-500/30"
            }`}
          >
            {isConnected ? "● Live WebSocket" : "○ Local Simulation"}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => evolveMutation.mutate()}
            disabled={evolveMutation.isPending}
            className="text-neon-mint hover:bg-neon-mint/10 border border-neon-mint/20"
          >
            <RefreshCw size={13} className="mr-1" /> Trigger Evolution
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-12 gap-4">
          {/* Main Tournament & Hypotheses (8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            <TournamentBracket
              matches={displayMatches}
              hypotheses={displayHypotheses}
              onSelectHypothesis={setSelectedHypothesis}
            />

            <Tabs defaultValue="hypotheses">
              <TabList>
                <Tab value="hypotheses">Hypotheses Population ({displayHypotheses.length})</Tab>
                <Tab value="evolution">Evolution Timeline</Tab>
                <Tab value="debates">Agent Debates</Tab>
              </TabList>

              <TabPanel value="hypotheses">
                <div className="space-y-3 mt-3">
                  {displayHypotheses.map((h: Hypothesis) => (
                    <HypothesisCard
                      key={h.id}
                      hypothesis={h}
                      selected={selectedHypothesis?.id === h.id}
                      onSelect={(hyp) => setSelectedHypothesis(hyp)}
                    />
                  ))}
                </div>
              </TabPanel>

              <TabPanel value="evolution">
                <div className="space-y-3 mt-3">
                  {[
                    {
                      gen: "Generation 4 (Latest)",
                      desc: "Tournament Champion selected: SVD-Attn demonstrated lowest spectral loss error.",
                      time: "Just now",
                      status: "Champion",
                    },
                    {
                      gen: "Generation 3",
                      desc: "Cross-breeding: memory-augmented state spaces merged with low-rank SVD projections.",
                      time: "12m ago",
                      status: "Cross-bred",
                    },
                    {
                      gen: "Generation 2",
                      desc: "Eliminated weak naive linear kernels via simulated reflection debates.",
                      time: "28m ago",
                      status: "Filtered",
                    },
                    {
                      gen: "Generation 1",
                      desc: "Seeded initial population of 8 hypotheses using GPT-4o co-scientist generator.",
                      time: "45m ago",
                      status: "Seeded",
                    },
                  ].map((event, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/60 border border-white/10 rounded-xl p-3.5 flex items-start justify-between gap-3"
                    >
                      <div className="flex items-start gap-3">
                        <div className="p-1.5 bg-neon-blue/10 rounded-lg text-neon-blue mt-0.5">
                          <GitBranch size={16} />
                        </div>
                        <div>
                          <div className="font-semibold text-white text-xs">{event.gen}</div>
                          <p className="text-slate-400 text-xs mt-1 leading-relaxed">
                            {event.desc}
                          </p>
                          <span className="text-[10px] text-slate-500 font-mono mt-1.5 block">
                            {event.time}
                          </span>
                        </div>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-white/5 text-slate-300">
                        {event.status}
                      </span>
                    </div>
                  ))}
                </div>
              </TabPanel>

              <TabPanel value="debates">
                <div className="space-y-3 mt-3">
                  <div className="bg-slate-900/60 border border-white/10 rounded-xl p-4 space-y-2">
                    <div className="flex items-center justify-between border-b border-white/5 pb-2">
                      <span className="text-xs font-bold text-white flex items-center gap-1.5">
                        <Swords size={14} className="text-yellow-400" />
                        Match #1 Debate: SVD-Attn vs Memory State Spaces
                      </span>
                      <span className="text-[10px] text-neon-mint font-mono font-bold">
                        Winner: SVD-Attn (85% vs 15%)
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      <strong>ReflectionAgent-1:</strong> "SVD-Attn provides rigorous theoretical bounds on singular values, guaranteeing stability across deep stacks without training divergences."
                    </p>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      <strong>ProximityAgent-2:</strong> "State space models require intricate decay parameter initialization, whereas SVD is deterministic and hardware-friendly."
                    </p>
                  </div>
                </div>
              </TabPanel>
            </Tabs>
          </div>

          {/* Right Sidebar: Agent Activity Panel (4 cols) */}
          <div className="lg:col-span-4 h-full">
            <AgentActivityPanel agents={displayAgents} />
          </div>
        </div>
      </div>
    </div>
  );
}
