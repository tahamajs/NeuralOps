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
import { FlaskConical, GitBranch, TrendingUp, RefreshCw } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function HypothesisGenerator({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [goal, setGoal] = useState("");
  const [selectedHypothesis, setSelectedHypothesis] = useState<string | null>(
    null,
  );
  const [feedback, setFeedback] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [iterations, setIterations] = useState(3);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: hypotheses,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["hypotheses", goal],
    queryFn: () =>
      researchDiscoveryAPI.generateHypotheses(goal, { model, iterations }),
    enabled: false,
  });

  const { data: timeline } = useQuery({
    queryKey: ["hypothesis-timeline", selectedHypothesis],
    queryFn: () =>
      researchDiscoveryAPI.getHypothesisTimeline(selectedHypothesis!),
    enabled: !!selectedHypothesis,
  });

  const evolveMutation = useMutation({
    mutationFn: () =>
      researchDiscoveryAPI.evolveHypothesis(selectedHypothesis!, feedback),
    onSuccess: (newHyp) => {
      queryClient.invalidateQueries({ queryKey: ["hypotheses"] });
      if (selectedHypothesis)
        queryClient.invalidateQueries({
          queryKey: ["hypothesis-timeline", selectedHypothesis],
        });
      success("Hypothesis evolved", "New generation created");
      setFeedback("");
      setSelectedHypothesis(newHyp.id);
    },
  });

  const handleGenerate = () => {
    if (goal.trim()) refetch();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FlaskConical size={20} className="text-neon-mint" /> Hypothesis
            Generator
          </h1>
          <p className="text-xs text-slate-400">
            Multi‑agent AI hypothesis generation and evolution
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
          <Textarea
            rows={3}
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g., Develop novel approaches for improving reasoning capabilities in large language models"
            className="mb-3"
          />
          <div className="flex gap-3">
            <Select
              options={[
                { value: "gpt-4o", label: "GPT‑4o" },
                { value: "claude-3.5", label: "Claude 3.5" },
                { value: "gemini-1.5", label: "Gemini 1.5" },
              ]}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-40"
            />
            <Select
              options={[
                { value: 1, label: "1 iteration" },
                { value: 3, label: "3 iterations" },
                { value: 5, label: "5 iterations" },
              ]}
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              className="w-36"
            />
            <Button
              onClick={handleGenerate}
              disabled={!goal.trim() || isLoading}
            >
              Generate Hypotheses
            </Button>
          </div>
        </TerminalCard>
        {isLoading && (
          <div className="text-center py-12">
            <Spinner size="lg" />
            <p className="text-slate-400 mt-2">
              AI agents are generating and debating hypotheses...
            </p>
          </div>
        )}
        {hypotheses && (
          <Tabs defaultValue="list">
            <TabList>
              <Tab value="list">Hypotheses</Tab>
              <Tab value="timeline">Evolution Timeline</Tab>
            </TabList>
            <TabPanel value="list">
              <div className="space-y-3">
                {hypotheses.map((h) => (
                  <HypothesisCard
                    key={h.id}
                    hypothesis={h}
                    onSelect={() => setSelectedHypothesis(h.id)}
                    selected={selectedHypothesis === h.id}
                  />
                ))}
              </div>
            </TabPanel>
            <TabPanel value="timeline">
              {selectedHypothesis ? (
                <HypothesisTimeline timeline={timeline || []} />
              ) : (
                <div className="text-center text-slate-500 py-8">
                  Select a hypothesis to view its evolution timeline
                </div>
              )}
            </TabPanel>
          </Tabs>
        )}
        {selectedHypothesis && (
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">
              Provide Feedback to Evolve
            </h3>
            <Textarea
              rows={3}
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Suggest improvements or ask the AI to refine the hypothesis..."
            />
            <Button
              className="mt-2"
              onClick={() => evolveMutation.mutate()}
              disabled={!feedback.trim() || evolveMutation.isPending}
            >
              <RefreshCw size={14} className="mr-1" /> Evolve Hypothesis
            </Button>
          </TerminalCard>
        )}
      </div>
    </div>
  );
}
