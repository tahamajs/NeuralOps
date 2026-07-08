import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";

interface FloatingNotesEditorProps {
  onClose?: () => void;
}

export default function FloatingNotesEditor({
  onClose,
}: FloatingNotesEditorProps) {
  const queryClient = useQueryClient();
  const [selectedNoteId, setSelectedNoteId] = useState<string>("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState("");

  // List of notes for selector
  const { data: notesData, isLoading: listLoading } = useQuery({
    queryKey: ["notes", "floating"],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.LIST)).data,
  });
  const notes = notesData?.results || [];

  // Load selected note
  const { data: note, isLoading: noteLoading } = useQuery({
    queryKey: ["note", selectedNoteId],
    queryFn: async () =>
      (await api.get(ENDPOINTS.NOTES.DETAIL(selectedNoteId))).data,
    enabled: !!selectedNoteId,
  });

  useEffect(() => {
    if (note) {
      setTitle(note.title || "");
      setContent(note.content || "");
      setTags((note.tags || []).join(", "));
    } else if (!selectedNoteId) {
      setTitle("");
      setContent("");
      setTags("");
    }
  }, [note, selectedNoteId]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const tagsArray = tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      if (selectedNoteId) {
        return api.patch(ENDPOINTS.NOTES.DETAIL(selectedNoteId), {
          title,
          content,
          tags: tagsArray,
        });
      } else {
        return api.post(ENDPOINTS.NOTES.LIST, {
          title,
          content,
          tags: tagsArray,
        });
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notes"] });
      if (!selectedNoteId) {
        setTitle("");
        setContent("");
        setTags("");
      }
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async () => {
      if (selectedNoteId) {
        await api.delete(ENDPOINTS.NOTES.DETAIL(selectedNoteId));
        setSelectedNoteId("");
        setTitle("");
        setContent("");
        setTags("");
      }
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notes"] }),
  });

  if (listLoading) return <Spinner />;

  const noteOptions = [
    { value: "", label: "➕ New Note..." },
    ...notes.map((n: any) => ({
      value: n.id,
      label: n.title || `Note ${n.id.slice(0, 8)}`,
    })),
  ];

  return (
    <div className="flex flex-col h-full bg-cyber-900 p-4 space-y-4 overflow-y-auto">
      <div className="flex justify-between items-center">
        <h2 className="text-white font-bold text-sm">✏️ Quick Note</h2>
        {onClose && (
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-xs"
          >
            ✕
          </button>
        )}
      </div>

      <Select
        label="Select / New"
        options={noteOptions}
        value={selectedNoteId}
        onChange={(e) => setSelectedNoteId(e.target.value)}
      />

      <Input
        placeholder="Title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />

      <div>
        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">
          Content (Markdown)
        </label>
        <textarea
          className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[200px] focus:border-neon-blue/50 resize-none"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Write your note..."
        />
      </div>

      <Input
        placeholder="Tags (comma separated)"
        value={tags}
        onChange={(e) => setTags(e.target.value)}
      />

      <div className="flex gap-2">
        <Button
          size="sm"
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending || !content.trim()}
        >
          💾 Save
        </Button>
        {selectedNoteId && (
          <Button
            size="sm"
            variant="danger"
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
          >
            🗑️ Delete
          </Button>
        )}
      </div>

      {saveMutation.isError && (
        <p className="text-red-400 text-xs">Failed to save note.</p>
      )}
    </div>
  );
}
