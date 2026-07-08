import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { Switch } from "../../../components/ui/Switch";

export function CTCDCard({ trigger, onRefresh }: any) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between">
        <div>
          <h3 className="font-bold text-white">{trigger.name}</h3>
          <div className="text-xs text-slate-400">Type: {trigger.type}</div>
        </div>
        <Switch checked={trigger.enabled} onChange={() => onRefresh()} />
      </div>
      <div className="text-xs text-slate-500 mt-2">
        Last trigger:{" "}
        {trigger.last_triggered_at
          ? new Date(trigger.last_triggered_at).toLocaleString()
          : "Never"}
      </div>
    </TerminalCard>
  );
}
