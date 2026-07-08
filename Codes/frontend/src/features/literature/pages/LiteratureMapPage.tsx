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
import { Search, GitBranch, Sparkles } from "lucide-react";

export default function LiteratureMapPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [paperId, setPaperId] = useState("");
  const [depth, setDepth] = useState(2);
  const [searchPaper, setSearchPaper] = useState("");
  const {
    data: graph,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["citation-graph", paperId, depth],
    queryFn: () => literatureAPI.getGraphForPaper(paperId, depth),
    enabled: !!paperId,
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white">Literature Map</h1>
        <div className="flex gap-2">
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="space-y-4">
            <TerminalCard className="p-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Paper ID or title"
                  value={searchPaper}
                  onChange={(e) => setSearchPaper(e.target.value)}
                  className="flex-1"
                />
                <Button
                  size="sm"
                  onClick={() => {
                    setPaperId(searchPaper);
                    refetch();
                  }}
                >
                  <Search size={14} />
                </Button>
              </div>
              <div className="mt-2">
                <label className="text-xs text-slate-400">Depth: </label>
                <select
                  value={depth}
                  onChange={(e) => setDepth(parseInt(e.target.value))}
                  className="bg-white/5 border border-white/10 rounded px-2 py-1 text-sm"
                >
                  <option value={1}>1 (direct citations)</option>
                  <option value={2}>2 (extended)</option>
                  <option value={3}>3 (full network)</option>
                </select>
              </div>
            </TerminalCard>
            <PathFinder />
            <Tabs defaultValue="recommendations">
              <TabList>
                <Tab value="recommendations">AI Recommendations</Tab>
              </TabList>
              <TabPanel value="recommendations">
                {paperId && <PaperRecommendations paperId={paperId} />}
              </TabPanel>
            </Tabs>
          </div>
          <div className="lg:col-span-2">
            <TerminalCard className="p-2" style={{ height: "600px" }}>
              {isLoading ? (
                <Spinner />
              ) : graph?.data ? (
                <ForceGraph
                  nodes={graph.data.nodes}
                  links={graph.data.links}
                  onNodeClick={(node) => setPaperId(node.id)}
                  height={560}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500">
                  Enter a paper ID to view its citation network
                </div>
              )}
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
