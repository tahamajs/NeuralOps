import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Avatar } from "../../../components/ui/Avatar";
import { Heart } from "lucide-react";
import { DiscussionReply } from "../types";

export function DiscussionPost({ reply }: { reply: DiscussionReply | any }) {
  if (!reply) return null;

  return (
    <TerminalCard className="p-3">
      <div className="flex gap-3">
        <Avatar src={reply.author_avatar} size="sm" />
        <div className="flex-1">
          <div className="flex justify-between">
            <span className="font-medium text-white">{reply.author || "Anonymous"}</span>
            <span className="text-xs text-slate-400">
              {reply.created_at
                ? new Date(reply.created_at).toLocaleString()
                : "-"}
            </span>
          </div>
          <div className="text-sm text-slate-300 mt-1 whitespace-pre-wrap">
            {reply.content || ""}
          </div>
          <button className="mt-1 text-xs text-slate-400">
            <Heart size={12} className="inline mr-0.5" /> {reply.like_count || 0}
          </button>
        </div>
      </div>
    </TerminalCard>
  );
}
