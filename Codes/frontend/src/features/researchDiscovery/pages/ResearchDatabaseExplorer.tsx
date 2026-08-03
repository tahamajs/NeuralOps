// src/features/researchDiscovery/pages/ResearchDatabaseExplorer.tsx
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { researchDiscoveryAPI } from "../services/api";
import { PaperCard } from "../components/PaperCard";
import { CitationGraph } from "../components/CitationGraph";
import { DatabaseSelector } from "../components/DatabaseSelector";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import {
  Database,
  Search,
  GitBranch,
  Filter,
  Globe,
  FileText,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

const ALL_DATABASES = [
  "arxiv",
  "pubmed",
  "semantic_scholar",
  "crossref",
  "openalex",
  "dblp",
  "doaj",
  "europe_pmc",
  "hal",
  "medrxiv",
  "unpaywall",
];

export default function ResearchDatabaseExplorer({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>([
    "arxiv",
    "pubmed",
  ]);
  const [selectedPaper, setSelectedPaper] = useState<string | null>(null);
  const { success, error } = useToast();

  const searchMutation = useMutation({
    mutationFn: () =>
      researchDiscoveryAPI.crossSearch(query, selectedDatabases, 50),
    onError: () => error("Search failed", "Please try again"),
  });

  const { data: graph, isLoading: graphLoading } = useQuery({
    queryKey: ["literature-map", selectedPaper],
    queryFn: () => researchDiscoveryAPI.getLiteratureMap(selectedPaper!, 2),
    enabled: !!selectedPaper,
  });

  const handleSearch = () => {
    if (query.trim()) searchMutation.mutate();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Database size={20} className="text-neon-mint" /> Research Database
            Explorer
          </h1>
          <p className="text-xs text-slate-400">
            Cross‑database academic search
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
          <div className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search across academic databases..."
              className="flex-1"
            />
            <Button
              onClick={handleSearch}
              disabled={!query.trim() || searchMutation.isPending}
            >
              <Search size={14} className="mr-1" /> Search
            </Button>
          </div>
          <div className="mt-3">
            <DatabaseSelector
              selected={selectedDatabases}
              onChange={setSelectedDatabases}
              databases={ALL_DATABASES}
            />
          </div>
        </TerminalCard>
        {searchMutation.isPending && (
          <div className="text-center py-12">
            <Spinner size="lg" />
            <p className="text-slate-400 mt-2">
              Searching across {selectedDatabases.length} databases...
            </p>
          </div>
        )}
        {searchMutation.data && (
          <Tabs defaultValue="results">
            <TabList>
              <Tab value="results">
                <FileText size={14} className="mr-1" /> Results (
                {searchMutation.data.length})
              </Tab>
              <Tab value="graph">
                <GitBranch size={14} className="mr-1" /> Citation Graph
              </Tab>
            </TabList>
            <TabPanel value="results">
              <div className="space-y-3">
                {searchMutation.data.map((paper) => (
                  <PaperCard
                    key={paper.id}
                    paper={paper}
                    onClick={() => setSelectedPaper(paper.id)}
                  />
                ))}
              </div>
            </TabPanel>
            <TabPanel value="graph">
              {graphLoading ? (
                <Spinner />
              ) : graph ? (
                <CitationGraph nodes={graph.nodes} links={graph.links} />
              ) : (
                <div className="text-center text-slate-500 py-12">
                  Click on a paper to view its citation network
                </div>
              )}
            </TabPanel>
          </Tabs>
        )}
      </div>
    </div>
  );
}
