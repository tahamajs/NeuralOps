import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";

export default function BulkNotesPage() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<string[]>([]);

  const { data: notes = [] } = useQuery({
    queryKey: ["notes"],
    queryFn: async () => (await api.get("/notes/v1/notes/")).data.results || [],
  });

  const deleteMutation = useMutation({
    mutationFn: async (ids: string[]) =>
      api.post("/notes/v1/notes/bulk/", { action: "delete", ids }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notes"] });
      setSelected([]);
    },
  });

  const tagMutation = useMutation({
    mutationFn: async ({
      ids,
      tags,
      action,
    }: {
      ids: string[];
      tags: string[];
      action: string;
    }) => api.patch("/notes/v1/notes/bulk/tags/", { ids, tags, action }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notes"] }),
  });

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        BULK <span className="text-neon-blue">NOTES</span>
      </h1>
      <Card>
        <div className="flex gap-2 mb-4">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setSelected(notes.map((n: any) => n.id))}
          >
            Select All
          </Button>
          <Button
            size="sm"
            variant="danger"
            disabled={selected.length === 0}
            onClick={() => deleteMutation.mutate(selected)}
          >
            Delete Selected ({selected.length})
          </Button>
        </div>
        <div className="space-y-2 max-h-[500px] overflow-y-auto scrollbar-thin">
          {notes.map((note: any) => (
            <div
              key={note.id}
              className="flex items-center gap-3 bg-cyber-800 p-3 rounded"
            >
              <input
                type="checkbox"
                checked={selected.includes(note.id)}
                onChange={() =>
                  setSelected((prev) =>
                    prev.includes(note.id)
                      ? prev.filter((id) => id !== note.id)
                      : [...prev, note.id],
                  )
                }
              />
              <span className="flex-1 text-sm text-white">
                {note.title || "Untitled"}
              </span>
              <Badge status="note">{note.tags.join(", ")}</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
