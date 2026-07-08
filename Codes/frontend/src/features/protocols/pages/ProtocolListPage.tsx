// src/features/protocols/pages/ProtocolListPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { protocolsAPI } from "../services/api";
import { ProtocolCard } from "../components/ProtocolCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Search, Plus, Filter, BookOpen } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ProtocolListPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["protocols", { search, status: statusFilter, page }],
    queryFn: () =>
      protocolsAPI.getProtocols({
        search: search || undefined,
        status: statusFilter || undefined,
      }),
  });

  const deleteProtocol = useMutation({
    mutationFn: (id: string) => protocolsAPI.deleteProtocol(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["protocols"] });
      success("Protocol deleted");
    },
  });

  const protocols = data?.results || [];
  const totalPages = Math.ceil((data?.count || 0) / 20);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <BookOpen size={20} className="text-neon-mint" /> Research Protocols
          </h1>
          <p className="text-xs text-slate-400">
            Reusable, versioned experimental procedures
          </p>
        </div>
        <div className="flex gap-2">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search protocols..."
              className="pl-9 w-64"
            />
          </div>
          <Select
            options={[
              { value: "", label: "All" },
              { value: "draft", label: "Draft" },
              { value: "published", label: "Published" },
              { value: "archived", label: "Archived" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-32"
          />
          <Link to="/protocols/new">
            <Button size="sm">
              <Plus size={14} className="mr-1" /> New Protocol
            </Button>
          </Link>
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <RefreshCw size={14} />
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          {protocols.map((p) => (
            <ProtocolCard
              key={p.id}
              protocol={p}
              onDelete={() => deleteProtocol.mutate(p.id)}
            />
          ))}
        </div>
        {totalPages > 1 && (
          <div className="flex justify-center gap-2 mt-4">
            <Button
              size="sm"
              variant="ghost"
              disabled={page === 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm text-slate-400">
              Page {page} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={page === totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
