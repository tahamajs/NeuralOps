import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";

export const NoteReactions = ({
  noteId,
  reactions,
}: {
  noteId: string;
  reactions: Record<string, string[]>;
}) => {
  const handleReact = async (reaction: string) => {
    await api.post(ENDPOINTS.NOTES.REACT(noteId), { reaction });
  };

  return (
    <div className="flex gap-2">
      {["like", "important", "dislike"].map((r) => (
        <button
          key={r}
          onClick={() => handleReact(r)}
          className="text-xs bg-white/5 px-2 py-1 rounded hover:bg-white/10 transition-colors"
        >
          {r} {reactions[r]?.length || 0}
        </button>
      ))}
    </div>
  );
};
