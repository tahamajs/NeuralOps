import { useState } from "react";
import { MarketplaceListing } from "../types";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Eye, Zap, CheckCircle2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface DynamicComparisonTableProps {
  listings: MarketplaceListing[];
}

export function DynamicComparisonTable({
  listings,
}: DynamicComparisonTableProps) {
  const [highlightDifferences, setHighlightDifferences] = useState(false);
  const { success } = useToast();

  if (!listings || !listings.length) {
    return (
      <div className="text-center text-slate-500 py-8 text-xs">
        No hardware targets to compare.
      </div>
    );
  }

  const allFields = [
    { key: "model", label: "GPU Model" },
    { key: "vram", label: "VRAM Memory" },
    { key: "architecture", label: "Architecture" },
    { key: "interconnect", label: "Interconnect" },
    { key: "price", label: "Spot Rate" },
    { key: "region", label: "Datacenter Region" },
    { key: "reliability", label: "Attested SLA" },
    { key: "availability", label: "Available / Total Units" },
    { key: "compliance", label: "Audited Compliance" },
  ];

  const getCellValue = (listing: MarketplaceListing, field: string) => {
    switch (field) {
      case "model":
        return listing.gpu_specs?.model || "N/A";
      case "vram":
        return `${listing.gpu_specs?.vram_gb || 0} GB`;
      case "architecture":
        return listing.gpu_specs?.architecture || "N/A";
      case "interconnect":
        return (listing.gpu_specs?.interconnects || []).join(", ") || "PCIe 4.0";
      case "price":
        return `$${listing.price_per_hour?.toFixed(4)}/hr`;
      case "region":
        return listing.region || "Global";
      case "reliability":
        return `${listing.reliability_score || 99.5}%`;
      case "availability":
        return `${listing.available_units || 0} / ${listing.total_units || 0} units`;
      case "compliance":
        return (listing.compliance_badges || []).join(", ") || "Standard";
      default:
        return "";
    }
  };

  const isDifferent = (field: string, idx: number) => {
    const values = listings.map((l) => getCellValue(l, field));
    return values[0] !== values[idx];
  };

  const handleDeploy = (listing: MarketplaceListing) => {
    success("Provisioning Node", `Allocating ${listing.gpu_specs?.model} from ${listing.provider}.`);
  };

  return (
    <div className="overflow-x-auto space-y-3">
      <table className="w-full text-xs font-sans">
        <thead>
          <tr className="border-b border-white/10">
            <th className="p-3 text-left font-semibold text-slate-400">Spec / Metric</th>
            {listings.map((l) => (
              <th key={l.id} className="p-3 text-left">
                <div className="font-bold text-white text-sm">{l.gpu_specs?.model}</div>
                <div className="text-[11px] text-slate-400 font-normal">{l.provider}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {allFields.map(({ key, label }) => (
            <tr key={key} className="border-t border-white/5 hover:bg-white/5 transition-colors">
              <td className="p-3 font-semibold text-slate-400">{label}</td>
              {listings.map((l, idx) => (
                <td
                  key={l.id}
                  className={`p-3 font-mono ${
                    highlightDifferences && isDifferent(key, idx)
                      ? "bg-yellow-500/10 text-yellow-300 font-bold"
                      : "text-slate-200"
                  }`}
                >
                  {getCellValue(l, key)}
                </td>
              ))}
            </tr>
          ))}
          <tr className="border-t border-white/10 bg-black/30">
            <td className="p-3 font-bold text-white">Deploy Node</td>
            {listings.map((l) => (
              <td key={l.id} className="p-3">
                <Button
                  size="sm"
                  onClick={() => handleDeploy(l)}
                  className="bg-gradient-to-r from-cyan-500 to-blue-600 text-black font-bold text-xs py-1.5 px-3 rounded-lg"
                >
                  <Zap size={12} className="mr-1" /> Deploy Now
                </Button>
              </td>
            ))}
          </tr>
        </tbody>
      </table>

      <div className="flex justify-end pt-1">
        <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={highlightDifferences}
            onChange={(e) => setHighlightDifferences(e.target.checked)}
            className="accent-cyan-400"
          />
          <span>Highlight specification differences</span>
        </label>
      </div>
    </div>
  );
}
