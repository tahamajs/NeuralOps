import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";

export default function ReputationPage() {
  const { data } = useQuery({
    queryKey: ["marketplace-reputation"],
    queryFn: () => api.get("/marketplace/reputation/").then((r) => r.data),
  });

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Reputation</h1>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Consumer Score</p>
          <p className="text-2xl font-bold">{data?.consumer_score}</p>
        </div>
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Provider Score</p>
          <p className="text-2xl font-bold">{data?.provider_score}</p>
        </div>
      </div>
      <h2 className="font-bold mb-2">Badges</h2>
      <div className="flex flex-wrap gap-2">
        {data?.badges?.map((b: any, i: number) => (
          <span key={i} className="badge-pill bg-white/10 text-white">
            {b.icon} {b.name}{" "}
            <span className="text-xs text-slate-400">({b.tier})</span>
          </span>
        ))}
      </div>
    </div>
  );
}
