// src/features/social/pages/GroupDiscussionPage.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { socialAPI } from "../services/api";
import { DiscussionPost } from "../components/DiscussionPost";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Textarea } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { MessageSquare, Plus, MessageCircle, Heart } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function GroupDiscussionPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { groupId } = useParams();
  const effectiveGroupId = groupId || "general";
  const [selectedDiscussionId, setSelectedDiscussionId] = useState<
    string | null
  >(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newContent, setNewContent] = useState("");
  const [replyContent, setReplyContent] = useState("");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: discussions, isLoading: threadsLoading } = useQuery({
    queryKey: ["discussions", effectiveGroupId],
    queryFn: () => socialAPI.getDiscussionThreads(effectiveGroupId),
    enabled: !!effectiveGroupId,
  });

  const { data: discussion, isLoading: discussionLoading } = useQuery({
    queryKey: ["discussion", selectedDiscussionId],
    queryFn: () => socialAPI.getDiscussion(selectedDiscussionId!),
    enabled: !!selectedDiscussionId,
  });

  const createDiscussion = useMutation({
    mutationFn: () =>
      socialAPI.createDiscussion(effectiveGroupId, newTitle, newContent),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["discussions", effectiveGroupId] });
      setShowCreateModal(false);
      setNewTitle("");
      setNewContent("");
      success("Discussion created");
    },
    onError: () => error("Failed to create discussion"),
  });

  const postReply = useMutation({
    mutationFn: (content: string) =>
      socialAPI.postReply(selectedDiscussionId!, content),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["discussion", selectedDiscussionId],
      });
      setReplyContent("");
      success("Reply posted");
    },
    onError: () => error("Failed to post reply"),
  });

  const likeDiscussion = useMutation({
    mutationFn: () => socialAPI.likeDiscussion(selectedDiscussionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["discussion", selectedDiscussionId],
      });
    },
  });

  const handleSendReply = () => {
    if (!replyContent.trim() || !selectedDiscussionId) return;
    postReply.mutate(replyContent.trim());
  };

  if (threadsLoading) return <Spinner />;

  const discussionList = Array.isArray(discussions) ? discussions : [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white">Group Discussions</h1>
          <p className="text-xs text-slate-400">Collaborate with lab members</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Discussion
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4 h-full">
          <div className="space-y-2 overflow-auto">
            {discussionList.length === 0 ? (
              <div className="text-center text-slate-500 py-8 text-sm">
                No discussion threads yet. Start one!
              </div>
            ) : (
              discussionList.map((d) => (
                <TerminalCard
                  key={d.id}
                  className={`p-3 cursor-pointer ${selectedDiscussionId === d.id ? "border-neon-blue" : ""}`}
                  onClick={() => setSelectedDiscussionId(d.id)}
                >
                  <div className="flex justify-between">
                    <h3 className="font-bold text-white">{d.title || "Untitled Discussion"}</h3>
                    <span className="text-xs text-slate-400">
                      {d.created_at
                        ? new Date(d.created_at).toLocaleDateString()
                        : "-"}
                    </span>
                  </div>
                  <div className="flex gap-3 mt-1 text-xs text-slate-500">
                    <span>
                      <MessageCircle size={12} className="inline mr-0.5" />{" "}
                      {d.reply_count || 0}
                    </span>
                    <span>
                      <Heart size={12} className="inline mr-0.5" /> {d.like_count || 0}
                    </span>
                    <span>by {d.author || "Anonymous"}</span>
                  </div>
                </TerminalCard>
              ))
            )}
          </div>
          <div className="lg:col-span-2 overflow-auto">
            {discussionLoading ? (
              <Spinner />
            ) : discussion ? (
              <div className="space-y-4">
                <TerminalCard className="p-4">
                  <h2 className="text-xl font-bold text-white">
                    {discussion.title || "Untitled"}
                  </h2>
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-sm text-slate-400">
                      Posted by {discussion.author || "Anonymous"} ·{" "}
                      {discussion.created_at
                        ? new Date(discussion.created_at).toLocaleString()
                        : "-"}
                    </span>
                    <button
                      onClick={() => likeDiscussion.mutate()}
                      className={`flex items-center gap-1 text-sm ${discussion.user_liked ? "text-red-400" : "text-slate-400 hover:text-red-400"}`}
                    >
                      <Heart size={14} /> {discussion.like_count || 0}
                    </button>
                  </div>
                  <div className="mt-3 text-slate-300 whitespace-pre-wrap">
                    {discussion.content || ""}
                  </div>
                </TerminalCard>
                <div className="space-y-2">
                  {(discussion.replies || []).map((reply) => (
                    <DiscussionPost key={reply.id} reply={reply} />
                  ))}
                </div>
                <TerminalCard className="p-4">
                  <Textarea
                    rows={3}
                    placeholder="Write a reply... (Press Enter to post)"
                    value={replyContent}
                    onChange={(e) => setReplyContent(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendReply();
                      }
                    }}
                  />
                  <Button
                    size="sm"
                    className="mt-2"
                    disabled={!replyContent.trim() || postReply.isPending}
                    onClick={handleSendReply}
                  >
                    {postReply.isPending ? "Posting..." : "Post Reply"}
                  </Button>
                </TerminalCard>
              </div>
            ) : (
              <div className="text-center text-slate-500 py-12">
                Select a discussion to view
              </div>
            )}
          </div>
        </div>
      </div>
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Start New Discussion"
      >
        <div className="space-y-4">
          <Input
            label="Title"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="Discussion topic..."
          />
          <Textarea
            label="Content"
            rows={6}
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="Share your thoughts, findings, or questions..."
          />
          <Button
            onClick={() => createDiscussion.mutate()}
            disabled={!newTitle.trim() || createDiscussion.isPending}
          >
            {createDiscussion.isPending ? "Posting..." : "Post"}
          </Button>
        </div>
      </Modal>
    </div>
  );
}
