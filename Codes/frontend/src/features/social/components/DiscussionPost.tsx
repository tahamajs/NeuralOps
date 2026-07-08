import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Avatar } from "../../../components/ui/Avatar";
import { Heart } from "lucide-react";

export function DiscussionPost({ reply }: any) {
  return (
    <TerminalCard className="p-3">
      <div className="flex gap-3">
        <Avatar src={reply.author_avatar} size="sm" />
        <div className="flex-1">
          <div className="flex justify-between">
            <span className="font-medium text-white">{reply.author}</span>
            <span className="text-xs text-slate-400">
              {new Date(reply.created_at).toLocaleString()}
            </span>
          </div>
          <div className="text-sm text-slate-300 mt-1 whitespace-pre-wrap">
            {reply.content}
          </div>
          <button className="mt-1 text-xs text-slate-400">
            <Heart size={12} className="inline mr-0.5" /> {reply.like_count}
          </button>
        </div>
      </div>
    </TerminalCard>
  );
}
