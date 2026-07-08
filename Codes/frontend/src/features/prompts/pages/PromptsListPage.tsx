import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";

export default function PromptsListPage() {
  const { data: prompts = [], isLoading } = useQuery({
    queryKey: ["prompts"],
    queryFn: async () => (await api.get("/api/prompts/")).data,
  });

  if (isLoading) return <Spinner />;
  if (!prompts.length) return <EmptyState title="No prompts registered" />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        PROMPT <span className="text-neon-blue">REGISTRY</span>
      </h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {prompts.map((p: any) => (
          <Link key={p.name} to={`/prompts/${p.name}`}>
            <Card hover className="h-full flex flex-col justify-between">
              <h3 className="text-white font-semibold text-lg">{p.name}</h3>
              <p className="text-xs text-slate-400 mt-2 line-clamp-3">
                {p.template}
              </p>
              <span className="text-[10px] text-neon-blue mt-4 font-bold uppercase tracking-wider">
                View Details →
              </span>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
