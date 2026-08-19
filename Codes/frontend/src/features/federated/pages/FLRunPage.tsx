import { useState } from "react";
import { useParams } from "react-router-dom";
import { FLDashboard } from "../components/FLDashboard";
import { Select } from "../../../components/ui/Select";

export default function FLRunPage({ onClose }: { onClose?: () => void }) {
  const { runId: paramRunId } = useParams();
  const queryRunId = new URLSearchParams(window.location.search).get("runId") ||
    new URLSearchParams(window.location.search).get("id");
  const [selectedRun, setSelectedRun] = useState(paramRunId || queryRunId || "fl-svd-cluster-01");

  const runOptions = [
    { value: "fl-svd-cluster-01", label: "FL Cluster 01 (SVD Low-Rank Attention)" },
    { value: "fl-edge-dist-02", label: "FL Edge 02 (Decentralized BERT)" },
    { value: "fl-priv-fedavg", label: "FL Private 03 (FedAvg + Differential Privacy)" },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 bg-slate-900/60 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">FL Run:</span>
          <Select
            options={runOptions}
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="w-64 text-xs"
          />
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <FLDashboard runId={selectedRun} onClose={onClose} />
      </div>
    </div>
  );
}
