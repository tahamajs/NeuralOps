import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { BookOpen, Search, Trash2, Tag, ExternalLink } from "lucide-react";

export default function PaperLibraryPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [noteInputs, setNoteInputs] = useState<Record<string, string>>({});

  const { data: papersRaw, isLoading } = useQuery({
    queryKey: ["savedPapers"],
    queryFn: async () => (await api.get("/agents/papers/saved/")).data,
  });
  const papers = Array.isArray(papersRaw)
    ? papersRaw
    : papersRaw?.results || [];

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.delete(`/agents/papers/saved/${id}/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["savedPapers"] }),
  });

  const annotateMut = useMutation({
    mutationFn: ({ id, notes }: { id: string; notes: string }) =>
      api.patch(`/agents/papers/saved/${id}/annotate/`, { notes }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["savedPapers"] }),
  });

  const filtered = papers.filter(
    (p: any) =>
      p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.authors.some((a: string) =>
        a.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
  );

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <BookOpen size={24} className="text-neon-blue" /> Paper Library
        </h1>
        <Input
          placeholder="Search papers..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-64"
        />
      </div>

      {filtered.length === 0 ? (
        <EmptyState title="No papers saved. Ask the agent to search for papers and save them!" />
      ) : (
        <div className="grid gap-4">
          {filtered.map((paper: any) => (
            <Card key={paper.id} hover className="space-y-3">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-white">
                    {paper.title}
                  </h3>
                  <p className="text-xs text-slate-400">
                    {paper.authors?.slice(0, 5).join(", ")} ({paper.year})
                    {paper.citation_count > 0 &&
                      ` · ${paper.citation_count} citations`}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => window.open(paper.url, "_blank")}
                  >
                    <ExternalLink size={14} />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMut.mutate(paper.id)}
                  >
                    <Trash2
                      size={14}
                      className="text-slate-400 hover:text-red-400"
                    />
                  </Button>
                </div>
              </div>

              <p className="text-xs text-slate-500 line-clamp-2">
                {paper.abstract}
              </p>

              <div className="flex flex-wrap gap-1">
                {paper.tags?.map((tag: string) => (
                  <Badge key={tag}>{tag}</Badge>
                ))}
              </div>

              <div className="flex gap-2 items-end">
                <Input
                  placeholder="Add notes..."
                  value={noteInputs[paper.id] || paper.notes || ""}
                  onChange={(e) =>
                    setNoteInputs({ ...noteInputs, [paper.id]: e.target.value })
                  }
                  className="flex-1 text-xs"
                />
                <Button
                  size="sm"
                  onClick={() =>
                    annotateMut.mutate({
                      id: paper.id,
                      notes: noteInputs[paper.id] || "",
                    })
                  }
                >
                  <Tag size={14} className="mr-1" /> Save
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
