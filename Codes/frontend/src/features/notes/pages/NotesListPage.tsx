import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Badge } from "../../../components/ui/Badge";
import { ResearcherNote } from "../types";
import {
  Plus,
  Search,
  Pin,
  Trash2,
  BookOpen,
  Users,
  Clock,
} from "lucide-react";

const VIEWS = ["all", "mine", "shared", "pinned", "public"] as const;
type ViewType = (typeof VIEWS)[number];

export default function NotesListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [view, setView] = useState<ViewType>(
    (searchParams.get("view") as ViewType) || "all",
  );
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortBy, setSortBy] = useState<"updated" | "created" | "title">(
    "updated",
  );
  const queryClient = useQueryClient();

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const { data: notes = [], isLoading } = useQuery<ResearcherNote[]>({
    queryKey: ["notes", view, debouncedSearch, sortBy],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (debouncedSearch) params.set("search", debouncedSearch);
      if (view === "pinned") params.set("pinned", "true");
      if (view === "public") params.set("is_public", "true");
      if (view === "shared") params.set("shared", "true");
      if (sortBy) params.set("sort", sortBy);
      return (
        (await api.get(`${ENDPOINTS.NOTES.LIST}?${params.toString()}`)).data
          .results || []
      );
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.delete(ENDPOINTS.NOTES.DETAIL(id)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notes"] }),
  });

  const pinMut = useMutation({
    mutationFn: ({ id, pinned }: { id: string; pinned: boolean }) =>
      api.patch(ENDPOINTS.NOTES.DETAIL(id), { pinned: !pinned }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notes"] }),
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <BookOpen size={24} className="text-neon-blue" /> Research Notes
        </h1>
        <Link to="/notes/new">
          <Button size="sm">
            <Plus size={14} className="mr-1" /> New Note
          </Button>
        </Link>
      </div>

      {/* View Tabs + Search */}
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex gap-1 p-1 bg-white/5 rounded-xl overflow-x-auto">
          {VIEWS.map((v) => (
            <button
              key={v}
              onClick={() => {
                setView(v);
                setSearchParams({ view: v });
              }}
              className={`px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider transition-all whitespace-nowrap ${view === v ? "bg-neon-blue text-white shadow" : "text-slate-400 hover:text-white"}`}
            >
              {v}
            </button>
          ))}
        </div>
        <div className="flex gap-2 items-center">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <input
              type="text"
              placeholder="Search notes..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:border-neon-blue/50 w-48"
            />
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-white"
          >
            <option value="updated">Updated</option>
            <option value="created">Created</option>
            <option value="title">Title</option>
          </select>
        </div>
      </div>

      {/* Notes Grid */}
      {isLoading ? (
        <Spinner />
      ) : notes.length === 0 ? (
        <EmptyState
          title={
            debouncedSearch
              ? "No notes match your search."
              : "No notes yet. Create your first one!"
          }
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {notes.map((note) => (
            <Card key={note.id} hover className="group relative flex flex-col">
              {note.pinned && (
                <Pin
                  size={12}
                  className="absolute top-2 right-2 text-yellow-400"
                />
              )}
              {note.shared_with?.length > 0 && (
                <span title={`Shared with ${note.shared_with.join(", ")}`} className="absolute top-2 right-8 text-neon-blue">
                  <Users size={12} />
                </span>
              )}
              <Link to={`/notes/${note.id}`} className="flex-1">
                <h3 className="text-sm font-semibold text-white line-clamp-1 group-hover:text-neon-blue transition-colors">
                  {note.title || "Untitled"}
                </h3>
                <p
                  className="text-xs text-slate-400 mt-1.5 line-clamp-3"
                  dangerouslySetInnerHTML={{
                    __html: note.content?.slice(0, 200) || "",
                  }}
                />
                <div className="flex flex-wrap gap-1 mt-2">
                  {note.tags?.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.5 rounded bg-white/5 text-[9px] text-slate-400 uppercase"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </Link>
              <div className="flex justify-between items-center mt-3 pt-2 border-t border-white/5 text-[10px] text-slate-500">
                <span>
                  {note.author_username} ·{" "}
                  <Clock size={10} className="inline" />{" "}
                  {new Date(note.updated_at).toLocaleDateString()}
                </span>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() =>
                      pinMut.mutate({ id: note.id, pinned: note.pinned })
                    }
                    className="p-1 hover:bg-white/10 rounded"
                    title="Pin"
                  >
                    <Pin size={12} />
                  </button>
                  <button
                    onClick={() => deleteMut.mutate(note.id)}
                    className="p-1 hover:bg-red-500/20 rounded"
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
