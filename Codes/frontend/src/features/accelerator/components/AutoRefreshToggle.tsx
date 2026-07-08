import { useState } from "react";
import { Switch } from "../../../components/ui/Switch";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../../../components/ui/Popover";
import { Settings } from "lucide-react";

export function AutoRefreshToggle({
  enabled,
  interval,
  onSave,
}: {
  enabled: boolean;
  interval: number;
  onSave: (enabled: boolean, interval: number) => void;
}) {
  const [localEnabled, setLocalEnabled] = useState(enabled);
  const [localInterval, setLocalInterval] = useState(interval);

  return (
    <Popover>
      <PopoverTrigger>
        <button
          className="p-2 rounded hover:bg-white/10"
          title="Auto‑refresh settings"
        >
          <Settings size={16} className="text-slate-400" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="glass-terminal p-3 w-64">
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-white">Auto‑refresh</span>
            <Switch checked={localEnabled} onChange={setLocalEnabled} />
          </div>
          {localEnabled && (
            <div>
              <label className="text-xs text-slate-400">
                Interval (seconds)
              </label>
              <select
                value={localInterval}
                onChange={(e) => setLocalInterval(parseInt(e.target.value))}
                className="w-full mt-1 bg-white/5 border border-white/10 rounded px-2 py-1 text-sm"
              >
                <option value={15}>15 seconds</option>
                <option value={30}>30 seconds</option>
                <option value={60}>1 minute</option>
                <option value={300}>5 minutes</option>
              </select>
            </div>
          )}
          <Button
            size="sm"
            className="w-full"
            onClick={() => onSave(localEnabled, localInterval)}
          >
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
