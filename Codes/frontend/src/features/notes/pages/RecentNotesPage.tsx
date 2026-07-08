import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Link } from "react-router-dom";

export default function RecentNotesPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["recentNotes"],
    queryFn: async () =>
      (await api.get("/api/notes/v1/notes/recent/?days=7")).data,
  });

  if (isLoading) return <Spinner />;
  const notes = data?.results || [];

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        RECENT <span className="text-neon-blue">NOTES</span>
      </h1>
      {notes.length === 0 ? (
        <p className="text-slate-500">No recent notes.</p>
      ) : (
        notes.map((note: any) => (
          <Link key={note.id} to={`/notes/${note.id}`}>
            <Card hover>
              <h3 className="text-white font-semibold">
                {note.title || "Untitled"}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                {note.content.slice(0, 100)}...
              </p>
              <div className="flex justify-between text-[10px] text-slate-500 mt-2">
                <span>{note.author_username}</span>
                <span>{new Date(note.updated_at).toLocaleDateString()}</span>
              </div>
            </Card>
          </Link>
        ))
      )}
    </div>
  );
}
