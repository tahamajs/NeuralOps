import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { BookOpen, ExternalLink, Bookmark, Lightbulb } from "lucide-react";

interface Props {
  runId?: string;
  fromLibrary?: boolean;
}

export default function PaperRecommendationsPanel({
  runId,
  fromLibrary,
}: Props) {
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());

  const endpoint = fromLibrary
    ? "/agents/papers/recommendations/library/"
    : runId
      ? `/agents/papers/recommendations/${runId}/`
      : null;

  const { data, isLoading } = useQuery({
    queryKey: ["paperRecommendations", runId, fromLibrary],
    queryFn: async () => {
      if (!endpoint) return { recommendations: [] };
      try {
        const res = await api.get(endpoint);
        return res.data;
      } catch (err) {
        return { recommendations: [] };
      }
    },
    enabled: !!endpoint,
  });

  const savePaper = async (paper: any) => {
    await api.post("/agents/papers/save/", {
      title: paper.title,
      authors: paper.authors || [],
      abstract: paper.abstract || "",
      url: paper.url || "",
      source: "semantic_scholar",
      source_id: paper.paperId || "",
      year: paper.year,
      citation_count: paper.citationCount || 0,
    });
    setSavedIds((prev) => new Set([...prev, paper.paperId]));
  };

  if (isLoading) return <Spinner className="py-4" />;
  const papers = data?.recommendations || [];

  if (!papers.length) return null;

  return (
    <Card>
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb size={18} className="text-yellow-400" />
        <h3 className="text-sm font-display font-bold text-white">
          Related Papers ({papers.length})
        </h3>
      </div>

      <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin">
        {papers.map((paper: any) => (
          <div key={paper.paperId} className="bg-cyber-800 p-3 rounded-lg">
            <h4 className="text-xs font-semibold text-white">{paper.title}</h4>
            <p className="text-[10px] text-slate-400 mt-0.5">
              {paper.authors?.slice(0, 3).join(", ")}
              {paper.year && ` (${paper.year})`}
              {paper.citationCount > 0 && ` · ${paper.citationCount} cites`}
            </p>
            <p className="text-[10px] text-slate-500 mt-1 line-clamp-2">
              {paper.abstract}
            </p>
            <div className="flex gap-1 mt-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => window.open(paper.url, "_blank")}
              >
                <ExternalLink size={12} className="mr-0.5" /> View
              </Button>
              {!savedIds.has(paper.paperId) && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => savePaper(paper)}
                >
                  <Bookmark size={12} className="mr-0.5" /> Save
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
