import { useState, useMemo } from "react";
import { useQueries } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Search } from "lucide-react";

const configEndpoints = [
  { key: "attentions", path: "/api/config/attentions/", label: "Attentions" },
  {
    key: "compressors",
    path: "/api/config/compressors/",
    label: "Compressors",
  },
  { key: "trainers", path: "/api/config/trainers/", label: "Trainers" },
  { key: "methods", path: "/api/config/methods/", label: "Methods" },
  { key: "models", path: "/api/config/models/", label: "Models" },
  { key: "tasks", path: "/api/config/tasks/", label: "Tasks" },
  { key: "losses", path: "/api/config/losses/", label: "Losses" },
  { key: "optimizers", path: "/api/config/optimizers/", label: "Optimizers" },
  { key: "schedulers", path: "/api/config/schedulers/", label: "Schedulers" },
  { key: "callbacks", path: "/api/config/callbacks/", label: "Callbacks" },
  {
    key: "augmentations",
    path: "/api/config/augmentations/",
    label: "Augmentations",
  },
];

export default function ConfigPage() {
  const [search, setSearch] = useState("");

  const results = useQueries({
    queries: configEndpoints.map(({ key, path }) => ({
      queryKey: ["config", key],
      queryFn: async () => (await api.get(path)).data,
      staleTime: 5 * 60 * 1000, // cache for 5 minutes
    })),
  });

  const isLoading = results.some((r) => r.isLoading);

  // Build a flat list of all config items for filtering
  const allItems = useMemo(() => {
    if (isLoading) return [];
    return configEndpoints.flatMap((ep, idx) => {
      const data = results[idx].data;
      const values = data
        ? data[ep.key] || data.models || data.tasks || []
        : [];
      if (Array.isArray(values)) {
        return values.map((v) => ({ group: ep.label, key: v, value: v }));
      }
      return Object.entries(values).map(([k, v]) => ({
        group: ep.label,
        key: k,
        value: typeof v === "string" ? v : JSON.stringify(v),
      }));
    });
  }, [results, isLoading]);

  const filtered = useMemo(() => {
    if (!search.trim()) return null;
    const q = search.toLowerCase();
    return allItems.filter(
      (item) =>
        item.key.toLowerCase().includes(q) ||
        String(item.value).toLowerCase().includes(q) ||
        item.group.toLowerCase().includes(q),
    );
  }, [allItems, search]);

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <h1 className="text-2xl font-display font-bold text-white">
          Configuration <span className="text-neon-blue">Reference</span>
        </h1>
        <div className="w-full sm:w-72">
          <Input
            placeholder="Search config..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            icon={<Search size={14} />}
          />
        </div>
      </div>

      {search.trim() && (
        <div className="text-xs text-slate-400">
          {filtered?.length ?? 0} result
          {(filtered?.length ?? 0) !== 1 ? "s" : ""} for "{search}"
        </div>
      )}

      {search.trim() ? (
        // Search results view
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered?.map((item, i) => (
            <Card key={i} hover className="flex flex-col gap-1">
              <p className="text-xs font-bold text-neon-blue uppercase tracking-wider">
                {item.group}
              </p>
              <p className="text-sm font-medium text-white">{item.key}</p>
              <p className="text-xs text-slate-400 truncate">
                {String(item.value)}
              </p>
            </Card>
          ))}
          {filtered?.length === 0 && (
            <p className="text-slate-500 col-span-full text-center py-8">
              No matching configuration found.
            </p>
          )}
        </div>
      ) : (
        // Full grid view
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {configEndpoints.map((ep, idx) => {
            const data = results[idx].data;
            const values = data
              ? data[ep.key] || data.models || data.tasks || []
              : [];
            return (
              <Card key={ep.key}>
                <h3 className="text-sm font-display font-bold text-white capitalize mb-2">
                  {ep.label}
                </h3>
                <ul className="text-xs text-slate-400 max-h-40 overflow-y-auto scrollbar-thin space-y-0.5">
                  {Array.isArray(values) ? (
                    values.length > 0 ? (
                      values.map((v: string, i: number) => (
                        <li key={i} className="flex items-center gap-2">
                          <div className="w-1 h-1 rounded-full bg-neon-blue/60 flex-shrink-0" />
                          <span className="truncate">{v}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic">No data</li>
                    )
                  ) : (
                    Object.entries(values).map(([k, v], i) => (
                      <li key={i} className="flex items-center gap-2">
                        <div className="w-1 h-1 rounded-full bg-neon-purple/60 flex-shrink-0" />
                        <span className="font-mono text-neon-blue">{k}:</span>
                        <span className="truncate">
                          {typeof v === "string" ? v : JSON.stringify(v)}
                        </span>
                      </li>
                    ))
                  )}
                </ul>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
