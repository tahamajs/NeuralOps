import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Bot, Plus, Trash2, MessageSquare } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

export default function AgentSessionsListPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: sessions = [], isLoading } = useQuery({
    queryKey: ["agentSessions"],
    queryFn: async () => (await api.get("/api/agents/chats/")).data,
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.delete(`/api/agents/chats/${id}/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["agentSessions"] }),
  });

  const handleNewChat = async () => {
    const res = await api.post("/api/agents/chats/", { title: "" });
    navigate(`/agent/${res.data.id}`);
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <Bot size={24} className="text-neon-blue" /> AI Assistant Chats
        </h1>
        <Button onClick={handleNewChat}>
          <Plus size={16} className="mr-1" /> New Chat
        </Button>
      </div>

      {sessions.length === 0 ? (
        <EmptyState title="No chats yet. Start a new conversation with the AI assistant!" />
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {sessions.map((session: any) => (
            <Card
              key={session.id}
              hover
              className="flex items-center justify-between"
            >
              <Link
                to={`/agent/${session.id}`}
                className="flex-1 flex items-center gap-4 p-2"
              >
                <div className="w-10 h-10 rounded-full bg-neon-blue/10 flex items-center justify-center">
                  <MessageSquare size={20} className="text-neon-blue" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    {session.title || "Untitled Chat"}
                  </h3>
                  <p className="text-xs text-slate-400">
                    {session.message_count || 0} messages · Last active{" "}
                    {formatDistanceToNow(new Date(session.updated_at), {
                      addSuffix: true,
                    })}
                  </p>
                </div>
              </Link>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => deleteMut.mutate(session.id)}
                className="ml-2"
              >
                <Trash2
                  size={14}
                  className="text-slate-400 hover:text-red-400"
                />
              </Button>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
