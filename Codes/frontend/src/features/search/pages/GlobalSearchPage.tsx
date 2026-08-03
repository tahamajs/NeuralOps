import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Input } from "../../../components/ui/Input";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Link } from "react-router-dom";

export default function GlobalSearchPage() {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const { data, isLoading } = useQuery({
    queryKey: ["globalSearch", debouncedQuery],
    queryFn: async () => (await api.get(`/search/?q=${debouncedQuery}`)).data,
    enabled: debouncedQuery.length > 0,
  });

  // Simple debounce
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    const timer = setTimeout(() => setDebouncedQuery(val), 300);
    return () => clearTimeout(timer);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        GLOBAL <span className="text-neon-blue">SEARCH</span>
      </h1>
      <Input
        value={query}
        onChange={handleChange}
        placeholder="Search across experiments, notes, models..."
        className="text-lg py-4"
      />
      {isLoading && <Spinner />}
      {data?.results && (
        <div className="space-y-4">
          <p className="text-slate-400 text-xs">
            {data.total} results for "{debouncedQuery}"
          </p>
          {data.results.map((item: any) => (
            <Link
              key={`${item.type}-${item.id}`}
              to={`/${item.type}s/${item.id}`}
            >
              <Card hover className="flex justify-between items-center">
                <div>
                  <h3 className="text-white font-semibold">
                    {item.title || "Untitled"}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">{item.subtitle}</p>
                </div>
                <Badge status={item.type}>{item.type}</Badge>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
