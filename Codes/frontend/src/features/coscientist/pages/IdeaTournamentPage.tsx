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
import { FlaskConical, Trophy, Brain, Activity } from "lucide-react";

export default function IdeaTournamentPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { sessionId: urlSessionId } = useParams();
  const [goal, setGoal] = useState("");
  const [config, setConfig] = useState({
    model_name: "gpt-4o",
    hypotheses_per_generation: 10,
  });
  const [isCreating, setIsCreating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(
    urlSessionId || null,
  );
  const { success, error: toastError } = useToast();
  const queryClient = useQueryClient();

  // WebSocket real‑time updates
  const {
    agents,
    hypotheses,
    matches,
    isConnected,
    error: wsError,
  } = useCoscientistWebSocket(sessionId);

  // Fetch session data when viewing an existing session
  const { data: session, isLoading } = useQuery({
    queryKey: ["coscientist-session", sessionId],
    queryFn: () => coscientistAPI.getSession(sessionId!),
    enabled: !!sessionId && !isCreating,
  });

  const startMutation = useMutation({
    mutationFn: () => coscientistAPI.startSession(goal, config),
    onSuccess: (res) => {
      setSessionId(res.data.session_id);
      setIsCreating(false);
      success(
        "Research session started",
        "The AI Co‑Scientist is now generating hypotheses.",
      );
    },
    onError: () => toastError("Failed to start session"),
  });

  const evolveMutation = useMutation({
    mutationFn: () => coscientistAPI.triggerEvolutionCycle(sessionId!),
    onSuccess: () => success("Evolution cycle triggered"),
  });

  if (isLoading) return <Spinner />;

  // Session creation form
  if (!sessionId && !isCreating) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex justify-between items-center p-4 border-b border-white/10">
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Brain size={20} className="text-neon-mint" /> AI Co‑Scientist
          </h1>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
        <div className="flex-1 overflow-auto p-6 max-w-2xl mx-auto w-full">
          <TerminalCard className="p-6">
            <h2 className="text-lg font-bold text-white mb-2">
              Define Your Research Goal
            </h2>
            <p className="text-sm text-slate-400 mb-4">
              The AI Co‑Scientist will generate, debate, and evolve hypotheses
              to help you break new ground.
            </p>
            <Textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={4}
              placeholder="e.g., Develop novel approaches for improving reasoning capabilities in large language models"
              className="mb-4"
            />
            <div className="grid grid-cols-2 gap-3 mb-4">
              <Input
                label="Model"
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
                    hypotheses_per_generation: parseInt(e.target.value),
                  })
                }
              />
            </div>
            <Button
              onClick={() => startMutation.mutate()}
              disabled={!goal.trim() || startMutation.isPending}
              className="w-full"
            >
              Start Research Tournament
            </Button>
          </TerminalCard>
        </div>
      </div>
    );
  }

  // Active session view
  const displayHypotheses = hypotheses.length
    ? hypotheses
    : session?.hypotheses || [];
  const displayMatches = matches.length ? matches : session?.matches || [];
  const displayAgents = agents.length ? agents : session?.agents || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy size={20} className="text-yellow-400" /> Idea Tournament
          </h1>
          <p className="text-xs text-slate-400 truncate max-w-md">
            {session?.goal || goal}
          </p>
        </div>
        <div className="flex gap-2">
          <span
            className={`text-xs ${isConnected ? "text-green-400" : "text-yellow-400"}`}
          >
            {isConnected ? "● Live" : "○ Reconnecting"}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => evolveMutation.mutate()}
            disabled={evolveMutation.isPending}
          >
            Evolve
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <TournamentBracket
              matches={displayMatches}
              hypotheses={displayHypotheses}
            />
            <Tabs defaultValue="hypotheses">
              <TabList>
                <Tab value="hypotheses">Hypotheses</Tab>
                <Tab value="evolution">Evolution Timeline</Tab>
              </TabList>
              <TabPanel value="hypotheses">
                <div className="space-y-3">
                  {displayHypotheses.map((h) => (
                    <HypothesisCard key={h.id} hypothesis={h} />
                  ))}
                </div>
              </TabPanel>
              <TabPanel value="evolution">
                <div className="text-slate-400 text-center py-8">
                  Evolution timeline coming soon
                </div>
              </TabPanel>
            </Tabs>
          </div>
          <div>
            <AgentActivityPanel agents={displayAgents} />
          </div>
        </div>
      </div>
    </div>
  );
}
