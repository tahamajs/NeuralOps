import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import RichNoteEditor from "../components/RichNoteEditor";
import { Save, Globe, Lock, Pin } from "lucide-react";

export default function NoteCreatePage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState("");
  const [isPublic, setIsPublic] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [saving, setSaving] = useState(false);

  const createMut = useMutation({
    mutationFn: async () => {
      const tagsArray = tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const res = await api.post(ENDPOINTS.NOTES.LIST, {
        title,
        content,
        tags: tagsArray,
        is_public: isPublic,
        pinned,
      });
      return res.data;
    },
    onSuccess: (data) => navigate(`/notes/${data.id}`),
  });

  const handleSave = async () => {
    setSaving(true);
    createMut.mutate();
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-display font-bold text-white">New Note</h1>
        <Button onClick={handleSave} disabled={saving || !content.trim()}>
          <Save size={14} className="mr-1" />{" "}
          {saving ? "Saving..." : "Save Note"}
        </Button>
      </div>

      <Input
        placeholder="Untitled Note"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        className="text-lg font-bold"
      />

      <RichNoteEditor
        content={content}
        onChange={setContent}
        placeholder="Write your research notes..."
      />

      <div className="flex gap-4 items-center">
        <Input
          placeholder="Tags (comma separated)"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          className="flex-1"
        />
        <button
          onClick={() => setIsPublic(!isPublic)}
          className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold border transition-colors ${isPublic ? "bg-green-500/10 border-green-500/30 text-green-400" : "bg-white/5 border-white/10 text-slate-400 hover:bg-white/10"}`}
        >
          {isPublic ? <Globe size={14} /> : <Lock size={14} />}{" "}
          {isPublic ? "Public" : "Private"}
        </button>
        <button
          onClick={() => setPinned(!pinned)}
          className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold border transition-colors ${pinned ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-400" : "bg-white/5 border-white/10 text-slate-400 hover:bg-white/10"}`}
        >
          <Pin size={14} /> {pinned ? "Pinned" : "Pin"}
        </button>
      </div>
    </div>
  );
}
