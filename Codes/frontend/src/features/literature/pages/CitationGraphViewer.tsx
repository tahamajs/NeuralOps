import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { literatureAPI } from "../services/api";
import { ForceGraph } from "../components/ForceGraph";
import { PaperRecommendations } from "../components/PaperRecommendations";
import { LandscapeCard } from "../components/LandscapeCard";
import { FeedItemComponent } from "../components/FeedItem";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";
import { Link } from "react-router-dom";

export default function CitationGraphViewer({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { paperId: urlPaperId } = useParams();
  const [paperId, setPaperId] = useState(urlPaperId || "");

  const { data: graph, isLoading } = useQuery({
    queryKey: ["citation-graph", paperId, 2],
    queryFn: () => literatureAPI.getGraphForPaper(paperId, 2),
    enabled: !!paperId,
  });

  const { data: landscapes } = useQuery({
    queryKey: ["landscapes"],
    queryFn: () => literatureAPI.getLandscapes(),
  });
  const { data: feed } = useQuery({
    queryKey: ["feed"],
    queryFn: () => literatureAPI.getFeed(),
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white">Citation Graph Viewer</h1>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              {isLoading ? (
                <Spinner />
              ) : graph?.data ? (
                <ForceGraph
                  nodes={graph.data.nodes}
                  links={graph.data.links}
                  onNodeClick={(node) => setPaperId(node.id)}
                  height={460}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500">
                  No paper selected
                </div>
              )}
            </TerminalCard>
          </div>
          <div>
            <TerminalCard className="p-4 h-full">
              <PaperRecommendations paperId={paperId} />
            </TerminalCard>
          </div>
        </div>
        <Tabs defaultValue="landscapes">
          <TabList>
            <Tab value="landscapes">Research Landscapes</Tab>
            <Tab value="feed">Activity Feed</Tab>
          </TabList>
          <TabPanel value="landscapes">
            <div className="grid md:grid-cols-2 gap-3">
              {(landscapes?.data || []).map((l) => (
                <LandscapeCard key={l.id} landscape={l} />
              ))}
            </div>
          </TabPanel>
          <TabPanel value="feed">
            <div className="space-y-2">
              {(feed?.data || []).slice(0, 10).map((f) => (
                <FeedItemComponent key={f.id} item={f} />
              ))}
            </div>
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
