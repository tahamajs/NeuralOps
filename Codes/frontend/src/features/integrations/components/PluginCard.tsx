import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Star, Download, Trash2, ExternalLink } from "lucide-react";
import { Plugin } from "../types";

export function PluginCard({
  plugin,
  onInstall,
  onUninstall,
  installed,
}: {
  plugin: Plugin;
  onInstall?: () => void;
  onUninstall?: () => void;
  installed?: boolean;
}) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div className="flex gap-3">
          <div className="w-12 h-12 rounded-lg bg-white/10 flex items-center justify-center text-2xl">
            {plugin.icon || "🔌"}
          </div>
          <div>
            <h3 className="font-bold text-white">{plugin.name}</h3>
            <p className="text-sm text-slate-400">{plugin.description}</p>
            <div className="flex gap-2 mt-1 text-xs">
              <span>v{plugin.version}</span>
              <span>by {plugin.author}</span>
              <span className="flex items-center gap-1">
                <Star size={10} className="fill-yellow-400 text-yellow-400" />{" "}
                {plugin.rating}
              </span>
              <span>
                <Download size={10} className="inline mr-0.5" />{" "}
                {plugin.downloads.toLocaleString()}
              </span>
            </div>
            <div className="flex flex-wrap gap-1 mt-2">
              {plugin.permissions.slice(0, 3).map((p) => (
                <Badge key={p} variant="info">
                  {p}
                </Badge>
              ))}
            </div>
          </div>
        </div>
        <div>
          <div className="flex gap-1">
            {installed ? (
              <Button size="sm" variant="ghost" onClick={onUninstall}>
                <Trash2 size={14} className="text-red-400" />
              </Button>
            ) : (
              <Button size="sm" variant="ghost" onClick={onInstall}>
                <Download size={14} /> Install
              </Button>
            )}
            <Button size="sm" variant="ghost">
              <ExternalLink size={14} />
            </Button>
          </div>
        </div>
      </div>
    </TerminalCard>
  );
}
