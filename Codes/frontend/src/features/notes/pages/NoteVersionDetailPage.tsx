import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { ArrowLeft } from "lucide-react";

export default function NoteVersionDetailPage() {
  const { noteId, versionNumber } = useParams<{
    noteId: string;
    versionNumber: string;
  }>();

  const { data, isLoading } = useQuery({
    queryKey: ["noteVersion", noteId, versionNumber],
    queryFn: async () =>
      (
        await api.get(
          `/notes/v1/notes/${noteId}/versions/${versionNumber}/`,
        )
      ).data,
    enabled: !!noteId && !!versionNumber,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex items-center gap-2">
        <Link
          to={`/notes/${noteId}/versions`}
          className="text-slate-400 hover:text-white"
        >
          <ArrowLeft size={18} />
        </Link>
        <h1 className="text-2xl font-display font-bold text-white">
          Version <span className="text-neon-blue">{versionNumber}</span>
        </h1>
      </div>

      <Card>
        <h2 className="text-sm font-bold text-white mb-2">{data?.title}</h2>
        <div className="text-xs text-slate-400 mb-4">
          Changed by {data?.changed_by || "unknown"} ·{" "}
          {new Date(data?.created_at).toLocaleString()}
        </div>
        <div className="prose prose-invert max-w-none text-slate-300 whitespace-pre-wrap">
          {data?.content}
        </div>
      </Card>
    </div>
  );
}
