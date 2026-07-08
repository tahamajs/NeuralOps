import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Webhook, Plus, Trash2, Play } from "lucide-react";

export default function WebhookManagerPage() {
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");
  const [events, setEvents] = useState("note.created,note.updated");

  const { data: webhooks = [], isLoading } = useQuery({
    queryKey: ["webhooks"],
    queryFn: async () => (await api.get("/api/notes/v1/webhooks/")).data,
  });

  const createMut = useMutation({
    mutationFn: async () => {
      const eventsArray = events
        .split(",")
        .map((e) => e.trim())
        .filter(Boolean);
      await api.post("/api/notes/v1/webhooks/", { url, events: eventsArray });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
      setUrl("");
      setEvents("");
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.delete(`/api/notes/v1/webhooks/${id}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["webhooks"] }),
  });

  const testMut = useMutation({
    mutationFn: (id: string) => api.post(`/api/notes/v1/webhooks/${id}/test/`),
    onSuccess: () => alert("Webhook test triggered"),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl mx-auto">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <Webhook size={24} className="text-neon-blue" /> Webhooks
      </h1>

      <Card>
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <Plus size={16} /> Add Webhook
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Input
            placeholder="URL"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <Input
            placeholder="Events (comma)"
            value={events}
            onChange={(e) => setEvents(e.target.value)}
          />
          <Button
            onClick={() => createMut.mutate()}
            disabled={!url.trim() || createMut.isPending}
          >
            Add
          </Button>
        </div>
      </Card>

      {webhooks.length === 0 ? (
        <EmptyState message="No webhooks configured." />
      ) : (
        webhooks.map((w: any) => (
          <Card key={w.id} className="flex items-center justify-between">
            <div className="text-sm">
              <p className="text-white">{w.url}</p>
              <p className="text-xs text-slate-400">{w.events?.join(", ")}</p>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => testMut.mutate(w.id)}
              >
                <Play size={14} className="mr-1" /> Test
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => deleteMut.mutate(w.id)}
              >
                <Trash2 size={14} className="text-red-400" />
              </Button>
            </div>
          </Card>
        ))
      )}
    </div>
  );
}
