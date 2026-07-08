import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { FileText, ExternalLink, Bookmark } from "lucide-react";

export function PaperCard({ paper, onClick }: any) {
  return (
    <TerminalCard
      className="p-3 cursor-pointer hover:border-neon-blue transition-colors"
      onClick={() => onClick?.(paper.id)}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-white">{paper.title}</h3>
            <Badge variant="info">{paper.source}</Badge>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {paper.authors.join(", ")} · {paper.year} · {paper.citations}{" "}
            citations
          </div>
          <p className="text-sm text-slate-300 mt-2 line-clamp-2">
            {paper.abstract.slice(0, 200)}...
          </p>
        </div>
        <div className="flex gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              window.open(paper.url, "_blank");
            }}
          >
            <ExternalLink
              size={14}
              className="text-slate-400 hover:text-white"
            />
          </button>
          <button>
            <Bookmark
              size={14}
              className="text-slate-400 hover:text-yellow-400"
            />
          </button>
        </div>
      </div>
    </TerminalCard>
  );
}
