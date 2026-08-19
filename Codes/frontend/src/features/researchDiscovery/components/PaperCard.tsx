import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { FileText, ExternalLink, Bookmark, Sparkles } from "lucide-react";
import { useState } from "react";
import { useToast } from "../../../components/ui/Toast";

export function PaperCard({ paper, onClick }: any) {
  const [saved, setSaved] = useState(false);
  const { success } = useToast();

  if (!paper) return null;

  const authors = Array.isArray(paper.authors)
    ? paper.authors.join(", ")
    : typeof paper.authors === "string"
    ? paper.authors
    : "Unknown Authors";

  const abstract = paper.abstract || "No abstract available.";
  const title = paper.title || "Untitled Publication";
  const citations = Number(paper.citations || 0);

  const handleBookmark = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSaved(!saved);
    success(saved ? "Removed from library" : "Saved to research library");
  };

  return (
    <TerminalCard
      className="p-3.5 cursor-pointer hover:border-neon-blue/60 transition-all bg-slate-950/40 border-white/10"
      onClick={() => onClick?.(paper.id)}
    >
      <div className="flex justify-between items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <FileText size={14} className="text-neon-blue shrink-0" />
            <h3 className="font-bold text-white text-xs truncate" title={title}>
              {title}
            </h3>
            <Badge variant="info" className="text-[9px] uppercase font-mono">
              {paper.source || "arXiv"}
            </Badge>
          </div>

          <div className="text-[11px] text-slate-400 font-mono">
            {authors} · <span className="text-white">{paper.year || 2024}</span> ·{" "}
            <span className="text-neon-mint">{citations} citations</span>
          </div>

          <p className="text-xs text-slate-300 mt-2 line-clamp-2 leading-relaxed">
            {abstract}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleBookmark}
            title={saved ? "Saved" : "Save to library"}
            className={`p-1.5 rounded-lg border transition-all ${
              saved
                ? "bg-yellow-400/20 border-yellow-400/40 text-yellow-400"
                : "bg-white/5 border-white/5 text-slate-400 hover:text-white"
            }`}
          >
            <Bookmark size={13} />
          </button>
          {paper.url && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                window.open(paper.url, "_blank");
              }}
              title="Open Paper URL"
              className="p-1.5 rounded-lg bg-white/5 border border-white/5 text-slate-400 hover:text-white transition-colors"
            >
              <ExternalLink size={13} />
            </button>
          )}
        </div>
      </div>
    </TerminalCard>
  );
}
