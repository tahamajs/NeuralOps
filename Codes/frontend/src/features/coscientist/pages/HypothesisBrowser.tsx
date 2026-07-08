// src/features/coscientist/pages/HypothesisBrowser.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { coscientistAPI } from "../services/api";
import { HypothesisCard } from "../components/HypothesisCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";
import { Search } from "lucide-react";

export default function HypothesisBrowser({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("total_score");
  const { data: sessions, isLoading } = useQuery({
    queryKey: ["coscientist-sessions"],
    queryFn: () => coscientistAPI.listSessions(),
  });
  if (isLoading) return <Spinner />;
  const allHypotheses =
    sessions?.data.results
      .flatMap((s) => s.hypotheses)
      .filter((h) => h.title.toLowerCase().includes(search.toLowerCase())) ||
    [];
  const sorted = [...allHypotheses].sort(
    (a, b) => (b[sortBy] || 0) - (a[sortBy] || 0),
  );
  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white">Hypothesis Browser</h1>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search hypotheses..."
              className="pl-9"
            />
          </div>
          <Select
            options={[
              { value: "total_score", label: "Total Score" },
              { value: "scientific_merit", label: "Scientific Merit" },
              { value: "novelty_score", label: "Novelty" },
              { value: "impact_score", label: "Impact" },
            ]}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="w-40"
          />
        </div>
        <div className="grid md:grid-cols-2 gap-3">
          {sorted.map((h) => (
            <HypothesisCard key={h.id} hypothesis={h} />
          ))}
        </div>
      </div>
    </div>
  );
}
