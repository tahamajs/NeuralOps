import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { ArrowLeft, Send } from "lucide-react";

export default function NoteRepliesPage() {
  const { noteId } = useParams<{ noteId: string }>();
  const queryClient = useQueryClient();
  const [replyContent, setReplyContent] = useState("");

  const { data: replies = [], isLoading } = useQuery({
    queryKey: ["noteReplies", noteId],
    queryFn: async () =>
      (await api.get(`/api/notes/v1/notes/${noteId}/replies/`)).data,
    enabled: !!noteId,
  });

  const postReplyMut = useMutation({
    mutationFn: async () => {
      await api.post(`/api/notes/v1/notes/${noteId}/replies/`, {
        content: replyContent,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["noteReplies", noteId] });
      setReplyContent("");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex items-center gap-2">
        <Link
          to={`/notes/${noteId}`}
          className="text-slate-400 hover:text-white"
        >
          <ArrowLeft size={18} />
        </Link>
        <h1 className="text-2xl font-display font-bold text-white">
          Replies ({replies.length})
        </h1>
      </div>

      {/* Reply input */}
      <Card>
        <textarea
          className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[80px] focus:border-neon-blue/50 resize-none"
          placeholder="Write a reply..."
          value={replyContent}
          onChange={(e) => setReplyContent(e.target.value)}
        />
        <Button
          size="sm"
          className="mt-2"
          onClick={() => postReplyMut.mutate()}
          disabled={!replyContent.trim() || postReplyMut.isPending}
        >
          <Send size={14} className="mr-1" /> Post Reply
        </Button>
      </Card>

      {replies.length === 0 ? (
        <EmptyState title="No replies to this note yet." />
      ) : (
        <div className="space-y-3">
          {replies.map((reply: any) => (
            <Card key={reply.id} hover>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span className="font-semibold">{reply.author_username}</span>
                <span>{new Date(reply.created_at).toLocaleString()}</span>
              </div>
              <div className="prose prose-invert max-w-none text-sm text-slate-300 whitespace-pre-wrap">
                {reply.content}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
