import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Layers, Plus, Search } from "lucide-react";

export default function TemplatesListPage() {
  const [search, setSearch] = useState("");

  const { data: templates = [], isLoading } = useQuery({
    queryKey: ["templates"],
    queryFn: async () => (await api.get("/api/experiments/templates/")).data,
  });

  const filtered = useMemo(() => {
    if (!search.trim()) return templates;
    const q = search.toLowerCase();
    return templates.filter(
      (t: any) =>
        (t.name || "").toLowerCase().includes(q) ||
        (t.description || "").toLowerCase().includes(q) ||
        (t.category || "").toLowerCase().includes(q),
    );
  }, [templates, search]);

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <Layers size={24} className="text-neon-blue" />
          Experiment <span className="text-neon-blue">Templates</span>
        </h1>
        <Link to="/templates/new">
          <Button size="sm">
            <Plus size={14} className="mr-1" /> New Template
          </Button>
        </Link>
      </div>

      {/* Search */}
      <div className="w-full max-w-sm">
        <Input
          placeholder="Search templates..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          icon={<Search size={14} />}
        />
      </div>

      {/* Templates Grid */}
      {filtered.length === 0 ? (
        <EmptyState
          title="No Presets Found"
          description={
            search
              ? `No templates match "${search}".`
              : "No experiment templates yet. Create your first preset!"
          }
          action={
            !search && (
              <Link to="/templates/new">
                <Button size="sm">Create Template</Button>
              </Link>
            )
          }
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((t: any) => (
            <Link key={t.id} to={`/templates/${t.id}`}>
              <Card hover className="h-full flex flex-col justify-between">
                <div>
                  <h3 className="text-white font-semibold text-sm">{t.name}</h3>
                  {t.description && (
                    <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                      {t.description}
                    </p>
                  )}
                </div>
                <div className="flex items-center justify-between mt-4 pt-3 border-t border-white/5">
                  <span className="px-2 py-0.5 rounded-full bg-white/5 text-[10px] text-slate-400 uppercase tracking-wider">
                    {t.category || "General"}
                  </span>
                  <Button variant="ghost" size="sm">
                    Edit →
                  </Button>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
