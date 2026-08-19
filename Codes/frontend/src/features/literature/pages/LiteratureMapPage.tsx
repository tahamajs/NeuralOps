import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { literatureAPI } from "../services/api";
import { ForceGraph } from "../components/ForceGraph";
import { PathFinder } from "../components/PathFinder";
import { PaperRecommendations } from "../components/PaperRecommendations";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Spinner } from "../../../components/ui/Spinner";
import { Search, GitBranch, Sparkles, BookOpen, Layers, Zap } from "lucide-react";
import { CitationGraph, PaperNode } from "../types";

export default function LiteratureMapPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [paperId, setPaperId] = useState("svd-linear-attn-2024");
  const [depth, setDepth] = useState(2);
  const [searchPaper, setSearchPaper] = useState("");

  const sampleGraph: CitationGraph = {
    nodes: [
      {
        id: "svd-linear-attn-2024",
        title: "SVD Linear Attention Optimization for Long Sequences",
        authors: ["Majs, T.", "NeuralOps Lab"],
        year: 2024,
        citations: 18,
        group: "core",
      },
      {
        id: "vaswani-2017",
        title: "Attention Is All You Need",
        authors: ["Vaswani, A. et al."],
        year: 2017,
        citations: 110000,
        group: "influential",
      },
      {
        id: "katharopoulos-2020",
        title: "Transformers are RNNs: Fast Autoregressive Transformers",
        authors: ["Katharopoulos, A. et al."],
        year: 2020,
        citations: 1420,
        group: "influential",
      },
      {
        id: "dao-2023",
        title: "FlashAttention-2: Faster Attention with Better Parallelism",
        authors: ["Dao, T."],
        year: 2023,
        citations: 890,
        group: "influential",
      },
      {
        id: "de-2024",
        title: "Griffin: Mixing Recurrent with Local Attention",
        authors: ["De, S. et al."],
        year: 2024,
        citations: 120,
        group: "emerging",
      },
      {
        id: "gu-2023",
        title: "Mamba: Linear-Time Sequence Modeling with State Spaces",
        authors: ["Gu, A.", "Dao, T."],
        year: 2023,
        citations: 950,
        group: "influential",
      },
    ],
    links: [
      { source: "svd-linear-attn-2024", target: "vaswani-2017", type: "references" },
      { source: "svd-linear-attn-2024", target: "katharopoulos-2020", type: "references" },
      { source: "svd-linear-attn-2024", target: "dao-2023", type: "references" },
      { source: "svd-linear-attn-2024", target: "de-2024", type: "references" },
      { source: "de-2024", target: "vaswani-2017", type: "references" },
      { source: "gu-2023", target: "katharopoulos-2020", type: "references" },
      { source: "dao-2023", target: "vaswani-2017", type: "references" },
    ],
  };

  const {
    data: graphResponse,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["citation-graph", paperId, depth],
    queryFn: () => literatureAPI.getGraphForPaper(paperId, depth),
    enabled: !!paperId,
  });

  const graph: CitationGraph = (graphResponse as any)?.data || graphResponse || sampleGraph;

  const handleQuickSelect = (id: string) => {
    setPaperId(id);
    setSearchPaper(id);
  };

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-neon-mint" /> Literature Map & Citation Graph
          </h1>
          <p className="text-xs text-slate-400">
            Interactive multi-hop academic citation topology and pathfinding
          </p>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-12 gap-4">
          {/* Left Column: Search & PathFinder (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            <TerminalCard className="p-4 space-y-3 bg-slate-950/40">
              <div className="flex gap-2">
                <Input
                  placeholder="Paper ID, DOI, or title..."
                  value={searchPaper}
                  onChange={(e) => setSearchPaper(e.target.value)}
                  className="flex-1 text-xs"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && searchPaper.trim()) {
                      setPaperId(searchPaper.trim());
                      refetch();
                    }
                  }}
                />
                <Button
                  size="sm"
                  onClick={() => {
                    if (searchPaper.trim()) {
                      setPaperId(searchPaper.trim());
                      refetch();
                    }
                  }}
                  className="bg-neon-blue text-white"
                >
                  <Search size={14} />
                </Button>
              </div>

              {/* Quick Preset Buttons */}
              <div>
                <span className="text-[10px] uppercase font-mono text-slate-400 block mb-1.5">
                  Popular Seeds
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    { id: "svd-linear-attn-2024", label: "SVD Linear Attention" },
                    { id: "vaswani-2017", label: "Attention Is All You Need" },
                    { id: "dao-2023", label: "FlashAttention-2" },
                    { id: "de-2024", label: "Griffin" },
                  ].map((p) => (
                    <button
                      key={p.id}
                      onClick={() => handleQuickSelect(p.id)}
                      className={`text-[10px] px-2 py-1 rounded-lg border transition-all ${
                        paperId === p.id
                          ? "bg-neon-mint/20 border-neon-mint text-white"
                          : "bg-white/5 border-white/5 text-slate-400 hover:text-white"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-white/5">
                <label className="text-xs text-slate-400 font-mono">Graph Depth:</label>
                <select
                  value={depth}
                  onChange={(e) => setDepth(parseInt(e.target.value))}
                  className="bg-white/5 border border-white/10 rounded px-2.5 py-1 text-xs text-white"
                >
                  <option value={1}>1-hop (Direct Citations)</option>
                  <option value={2}>2-hop (Extended Network)</option>
                  <option value={3}>3-hop (Full Ecosystem)</option>
                </select>
              </div>
            </TerminalCard>

            <PathFinder />

            <Tabs defaultValue="recommendations">
              <TabList>
                <Tab value="recommendations">
                  <Sparkles size={12} className="mr-1" /> AI Recommendations
                </Tab>
              </TabList>
              <TabPanel value="recommendations" className="mt-2">
                <PaperRecommendations paperId={paperId} />
              </TabPanel>
            </Tabs>
          </div>

          {/* Right Column: Interactive 2D Graph (8 cols) */}
          <div className="lg:col-span-8">
            <TerminalCard className="p-2 h-[640px] flex flex-col bg-slate-950/40">
              <div className="flex items-center justify-between px-3 py-2 border-b border-white/5 text-xs text-slate-400 font-mono">
                <span className="flex items-center gap-1.5">
                  <Layers size={13} className="text-neon-mint" />
                  Showing {graph.nodes?.length || 0} nodes · {graph.links?.length || 0} citation edges
                </span>
                <span className="text-[10px] text-slate-500">
                  Click any node to focus its citation ego-network
                </span>
              </div>

              <div className="flex-1 w-full relative">
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <Spinner size="lg" />
                  </div>
                ) : (
                  <ForceGraph
                    nodes={graph.nodes || []}
                    links={graph.links || []}
                    onNodeClick={(node) => {
                      setPaperId(node.id);
                      setSearchPaper(node.title);
                    }}
                    height={580}
                  />
                )}
              </div>
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
