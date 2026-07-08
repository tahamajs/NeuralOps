import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import RichNoteEditor from "../components/RichNoteEditor";
import { ResearcherNote } from "../types";
import {
  Save,
  Trash2,
  Pin,
  Globe,
  Lock,
  Users,
  Share2,
  History,
  Reply,
  Eye,
  EyeOff,
} from "lucide-react";

// Optional: lazy load paper recommendations to avoid circular
const PaperRecommendationsPanel = React.lazy(
  () => import("../../agent/components/PaperRecommendationsPanel"),
);

export default function NoteDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");
  const [editTags, setEditTags] = useState("");
  const [editPublic, setEditPublic] = useState(false);
  const [editPinned, setEditPinned] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const [shareUsername, setShareUsername] = useState("");

  const { data: note, isLoading } = useQuery<ResearcherNote>({
    queryKey: ["note", id],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.DETAIL(id!))).data,
    enabled: !!id,
  });

  useEffect(() => {
    if (note && editing) {
      setEditTitle(note.title || "");
      setEditContent(note.content || "");
      setEditTags((note.tags || []).join(", "));
      setEditPublic(note.is_public);
      setEditPinned(note.pinned);
    }
  }, [note, editing]);

  const updateMut = useMutation({
    mutationFn: async (data: any) => {
      const tagsArray = (data.tags || "")
        .split(",")
        .map((t: string) => t.trim())
        .filter(Boolean);
      return api.patch(ENDPOINTS.NOTES.DETAIL(id!), {
        ...data,
        tags: tagsArray,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["note", id] });
      setEditing(false);
    },
  });

  const deleteMut = useMutation({
    mutationFn: () => api.delete(ENDPOINTS.NOTES.DETAIL(id!)),
    onSuccess: () => navigate("/notes"),
  });

  const shareMut = useMutation({
    mutationFn: (username: string) =>
      api.post(ENDPOINTS.NOTES.SHARE(id!), { username }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["note", id] }),
  });

  const pinMut = useMutation({
    mutationFn: (pinned: boolean) =>
      api.patch(ENDPOINTS.NOTES.DETAIL(id!), { pinned }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["note", id] }),
  });

  const reactMut = useMutation({
    mutationFn: (reaction: string) =>
      api.post(ENDPOINTS.NOTES.REACT(id!), { reaction }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["note", id] }),
  });

  if (isLoading || !note) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      {/* Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
          ← Back
        </Button>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => pinMut.mutate(!note.pinned)}
          >
            <Pin size={14} className={note.pinned ? "text-yellow-400" : ""} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowShare(!showShare)}
          >
            <Share2 size={14} />
          </Button>
          <Link to={`/notes/${id}/versions`}>
            <Button variant="ghost" size="sm">
              <History size={14} />
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setEditing(!editing)}
          >
            {editing ? <EyeOff size={14} /> : <Eye size={14} />}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => deleteMut.mutate()}>
            <Trash2 size={14} />
          </Button>
        </div>
      </div>

      {/* Share Panel */}
      {showShare && (
        <Card className="animate-scale-in">
          <h3 className="text-sm font-bold text-white mb-2">Share Note</h3>
          <div className="flex gap-2">
            <Input
              placeholder="Username"
              value={shareUsername}
              onChange={(e) => setShareUsername(e.target.value)}
            />
            <Button size="sm" onClick={() => shareMut.mutate(shareUsername)}>
              Share
            </Button>
          </div>
          {note.shared_with && note.shared_with.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="text-xs text-slate-400">Shared with:</span>
              {note.shared_with.map((u) => (
                <span
                  key={u}
                  className="px-2 py-1 bg-neon-blue/10 text-neon-blue rounded text-xs"
                >
                  {u}
                </span>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Content */}
      {editing ? (
        <Card className="space-y-4">
          <Input
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            placeholder="Title"
            className="text-lg font-bold"
          />
          <RichNoteEditor content={editContent} onChange={setEditContent} />
          <Input
            value={editTags}
            onChange={(e) => setEditTags(e.target.value)}
            placeholder="Tags (comma separated)"
          />
          <div className="flex gap-2">
            <Button
              onClick={() =>
                updateMut.mutate({
                  title: editTitle,
                  content: editContent,
                  tags: editTags,
                  is_public: editPublic,
                  pinned: editPinned,
                })
              }
              disabled={updateMut.isPending}
            >
              <Save size={14} className="mr-1" /> Save
            </Button>
            <Button variant="ghost" onClick={() => setEditing(false)}>
              Cancel
            </Button>
          </div>
        </Card>
      ) : (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-display font-bold text-white">
              {note.title || "Untitled"}
            </h1>
            <div className="flex items-center gap-2">
              {note.is_public ? (
                <Globe size={14} className="text-green-400" />
              ) : (
                <Lock size={14} className="text-slate-400" />
              )}
              <span className="text-xs text-slate-500">v{note.version}</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-1 mb-4">
            {note.tags?.map((tag) => (
              <Badge key={tag}>{tag}</Badge>
            ))}
          </div>
          <div
            className="prose prose-invert max-w-none text-slate-300"
            dangerouslySetInnerHTML={{ __html: note.content }}
          />
          <div className="mt-6 flex justify-between items-center text-xs text-slate-500 border-t border-white/5 pt-4">
            <span>
              By {note.author_username} · Updated{" "}
              {new Date(note.updated_at).toLocaleString()}
            </span>
            <div className="flex gap-1">
              {["👍", "❤️", "🔥", "💡", "❓"].map((emoji) => (
                <button
                  key={emoji}
                  onClick={() => reactMut.mutate(emoji)}
                  className={`px-2 py-1 rounded hover:bg-white/10 ${(note.reactions?.[emoji] || []).length > 0 ? "text-yellow-400" : "text-slate-500"}`}
                >
                  {emoji} {(note.reactions?.[emoji] || []).length}
                </button>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Related Papers (if run_id exists) */}
      {note.metadata?.run_id && (
        <React.Suspense fallback={<Spinner />}>
          <PaperRecommendationsPanel runId={note.metadata.run_id} />
        </React.Suspense>
      )}
    </div>
  );
}
