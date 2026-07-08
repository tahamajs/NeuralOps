import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { literatureAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { GitBranch, Copy, Save } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export function PathFinder() {
  const [fromId, setFromId] = useState("");
  const [toId, setToId] = useState("");
  const [result, setResult] = useState<{
    path: string[];
    length: number;
  } | null>(null);
  const { success, error } = useToast();

  const findMutation = useMutation({
    mutationFn: () => literatureAPI.findPath(fromId, toId),
    onSuccess: (res) => setResult(res.data),
    onError: () => error("Failed to find path"),
  });

  const saveMutation = useMutation({
    mutationFn: () => literatureAPI.getSavedPaths(),
    onSuccess: () => success("Path saved"),
  });

  return (
    <TerminalCard className="p-4">
      <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-3">
        <GitBranch size={16} className="text-neon-mint" /> Citation Pathfinder
      </h3>
      <div className="grid grid-cols-2 gap-3 mb-3">
        <Input
          placeholder="Start paper ID"
          value={fromId}
          onChange={(e) => setFromId(e.target.value)}
        />
        <Input
          placeholder="Target paper ID"
          value={toId}
          onChange={(e) => setToId(e.target.value)}
        />
      </div>
      <Button
        size="sm"
        onClick={() => findMutation.mutate()}
        disabled={!fromId || !toId || findMutation.isPending}
        className="w-full"
      >
        {findMutation.isPending ? <Spinner size="sm" /> : "Find Path"}
      </Button>
      {result && (
        <div className="mt-4 p-3 bg-white/5 rounded">
          <div className="text-xs text-slate-400 mb-1">
            Path length: {result.length}
          </div>
          <div className="text-sm text-white break-all">
            {result.path.join(" → ")}
          </div>
          <div className="flex gap-2 mt-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => saveMutation.mutate()}
            >
              <Save size={12} /> Save
            </Button>
            <Button size="sm" variant="ghost">
              <Copy size={12} /> Copy
            </Button>
          </div>
        </div>
      )}
    </TerminalCard>
  );
}
