import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { BarChart3 } from "lucide-react";

export default function NoteStatisticsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["noteStats"],
    queryFn: async () =>
      (await api.get("/notes/v1/notes/statistics/")).data,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <BarChart3 size={24} className="text-neon-blue" /> Note Statistics
      </h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center p-4">
          <div className="text-3xl font-bold text-white">
            {data?.total_notes || 0}
          </div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-1">
            Total Notes
          </div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-3xl font-bold text-green-400">
            {data?.public_notes || 0}
          </div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-1">
            Public
          </div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-3xl font-bold text-slate-400">
            {data?.private_notes || 0}
          </div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-1">
            Private
          </div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-3xl font-bold text-neon-blue">
            {data?.top_authors?.length || 0}
          </div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-1">
            Active Authors
          </div>
        </Card>
      </div>

      {data?.top_authors?.length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-2">Top Authors</h3>
          <div className="space-y-2">
            {data.top_authors.map((a: any) => (
              <div
                key={a.author__username}
                className="flex justify-between text-xs py-1 border-b border-white/5"
              >
                <span className="text-white">{a.author__username}</span>
                <span className="text-slate-400">{a.count} notes</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
