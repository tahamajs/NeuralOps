import { useState } from "react";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Button } from "../../../components/ui/Button";
import { Modal } from "../../../components/ui/Modal";
import { Badge } from "../../../components/ui/Badge";
import { PerformanceSnapshot } from "../types";
import { ArrowUpDown, GitCompare, BarChart2, CheckCircle2, Clock } from "lucide-react";

export function SnapshotTable({
  snapshots,
}: {
  snapshots: PerformanceSnapshot[];
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [inspectSnapshot, setInspectSnapshot] = useState<PerformanceSnapshot | null>(null);

  const toggleSelect = (id: string) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter((item) => item !== id));
    } else {
      if (selectedIds.length >= 2) {
        setSelectedIds([selectedIds[1], id]);
      } else {
        setSelectedIds([...selectedIds, id]);
      }
    }
  };

  const snapA = snapshots.find((s) => s.id === selectedIds[0]);
  const snapB = snapshots.find((s) => s.id === selectedIds[1]);

  const columns = [
    {
      id: "select",
      header: "",
      cell: ({ row }: any) => {
        const isChecked = selectedIds.includes(row.original.id);
        return (
          <input
            type="checkbox"
            checked={isChecked}
            onChange={() => toggleSelect(row.original.id)}
            className="rounded border-slate-700 bg-slate-900 text-neon-blue focus:ring-0 cursor-pointer"
            title="Select for comparison"
          />
        );
      },
    },
    {
      accessorKey: "timestamp",
      header: "Timestamp",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-slate-300">
          {new Date(row.original.timestamp).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      ),
    },
    {
      accessorKey: "metrics.accuracy",
      header: "Accuracy",
      cell: ({ row }: any) => (
        <span className="font-mono font-bold text-white">
          {(row.original.metrics.accuracy * 100).toFixed(2)}%
        </span>
      ),
    },
    {
      accessorKey: "metrics.f1",
      header: "F1 Score",
      cell: ({ row }: any) => (
        <span className="font-mono text-neon-mint">
          {row.original.metrics.f1.toFixed(4)}
        </span>
      ),
    },
    {
      accessorKey: "metrics.precision",
      header: "Precision",
      cell: ({ row }: any) => (
        <span className="font-mono text-slate-300">
          {row.original.metrics.precision.toFixed(4)}
        </span>
      ),
    },
    {
      accessorKey: "metrics.recall",
      header: "Recall",
      cell: ({ row }: any) => (
        <span className="font-mono text-slate-300">
          {row.original.metrics.recall.toFixed(4)}
        </span>
      ),
    },
    {
      accessorKey: "metrics.latency_p99",
      header: "Latency P50/P99",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-yellow-300">
          {row.original.metrics.latency_p50}ms / {row.original.metrics.latency_p99}ms
        </span>
      ),
    },
    {
      accessorKey: "metrics.throughput",
      header: "Throughput",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-slate-300">
          {row.original.metrics.throughput?.toLocaleString() || 1450} req/s
        </span>
      ),
    },
    {
      accessorKey: "data_drift_score",
      header: "Data Drift",
      cell: ({ row }: any) => {
        const drift = row.original.data_drift_score;
        return (
          <span
            className={`font-mono text-xs font-bold ${
              drift > 0.1 ? "text-red-400" : drift > 0.05 ? "text-yellow-400" : "text-green-400"
            }`}
          >
            {drift.toFixed(4)}
          </span>
        );
      },
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }: any) => (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setInspectSnapshot(row.original)}
          className="text-xs h-6 px-2 text-neon-blue"
        >
          Details
        </Button>
      ),
    },
  ];

  return (
    <div className="space-y-3">
      {/* Comparison Toolbar when items selected */}
      {selectedIds.length > 0 && (
        <div className="p-2.5 bg-neon-blue/10 border border-neon-blue/30 rounded flex justify-between items-center text-xs">
          <span className="text-white flex items-center gap-1.5 font-medium">
            <GitCompare size={14} className="text-neon-mint" />
            {selectedIds.length} snapshot(s) selected for comparison
          </span>
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={() => setShowCompareModal(true)}
              disabled={selectedIds.length !== 2}
              className="bg-neon-blue text-white h-7 text-xs"
            >
              Compare Snapshots (2)
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSelectedIds([])}
              className="h-7 text-xs"
            >
              Clear
            </Button>
          </div>
        </div>
      )}

      <DataGrid data={snapshots} columns={columns} />

      {/* Snapshot Comparison Modal */}
      <Modal
        open={showCompareModal}
        onClose={() => setShowCompareModal(false)}
        title="Side-by-Side Snapshot Comparison"
      >
        {snapA && snapB ? (
          <div className="space-y-4 max-h-[70vh] overflow-y-auto">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-white/5 rounded border border-white/10">
                <div className="font-bold text-white">Snapshot A</div>
                <div className="text-slate-400 text-[11px] font-mono mt-0.5">
                  {new Date(snapA.timestamp).toLocaleString()}
                </div>
              </div>
              <div className="p-3 bg-white/5 rounded border border-white/10">
                <div className="font-bold text-white">Snapshot B</div>
                <div className="text-slate-400 text-[11px] font-mono mt-0.5">
                  {new Date(snapB.timestamp).toLocaleString()}
                </div>
              </div>
            </div>

            {/* Metrics Comparison Table */}
            <div className="p-3 bg-slate-900 rounded border border-white/10 space-y-2 text-xs">
              <h4 className="font-bold text-white text-xs border-b border-white/10 pb-1.5">
                Metric Performance Deltas
              </h4>
              {[
                { label: "Accuracy", a: snapA.metrics.accuracy, b: snapB.metrics.accuracy, format: (v: number) => `${(v * 100).toFixed(2)}%`, higherBetter: true },
                { label: "F1 Score", a: snapA.metrics.f1, b: snapB.metrics.f1, format: (v: number) => v.toFixed(4), higherBetter: true },
                { label: "Precision", a: snapA.metrics.precision, b: snapB.metrics.precision, format: (v: number) => v.toFixed(4), higherBetter: true },
                { label: "Recall", a: snapA.metrics.recall, b: snapB.metrics.recall, format: (v: number) => v.toFixed(4), higherBetter: true },
                { label: "Latency P50", a: snapA.metrics.latency_p50, b: snapB.metrics.latency_p50, format: (v: number) => `${v}ms`, higherBetter: false },
                { label: "Latency P99", a: snapA.metrics.latency_p99, b: snapB.metrics.latency_p99, format: (v: number) => `${v}ms`, higherBetter: false },
                { label: "Data Drift Score", a: snapA.data_drift_score, b: snapB.data_drift_score, format: (v: number) => v.toFixed(4), higherBetter: false },
              ].map((row) => {
                const delta = row.b - row.a;
                const isPositive = row.higherBetter ? delta >= 0 : delta <= 0;
                return (
                  <div key={row.label} className="grid grid-cols-12 items-center py-1 border-b border-white/5 font-mono">
                    <span className="col-span-5 text-slate-300 font-sans font-medium">{row.label}</span>
                    <span className="col-span-2 text-center text-slate-300">{row.format(row.a)}</span>
                    <span className="col-span-2 text-center text-slate-300">{row.format(row.b)}</span>
                    <span className={`col-span-3 text-right font-bold ${delta === 0 ? "text-slate-400" : isPositive ? "text-green-400" : "text-red-400"}`}>
                      {delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
      </Modal>

      {/* Snapshot Inspect Modal */}
      <Modal
        open={!!inspectSnapshot}
        onClose={() => setInspectSnapshot(null)}
        title={`Snapshot Profile: ${inspectSnapshot?.id || ""}`}
      >
        {inspectSnapshot && (
          <div className="space-y-4 text-xs">
            <div className="p-3 bg-white/5 rounded border border-white/10 space-y-1">
              <div className="font-bold text-white text-sm">{inspectSnapshot.model_name}</div>
              <div className="text-slate-400 font-mono">
                Captured: {new Date(inspectSnapshot.timestamp).toLocaleString()}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center font-mono">
              <div className="p-2 bg-slate-900 rounded border border-white/5">
                <span className="text-slate-500 text-[10px]">ACCURACY</span>
                <div className="font-bold text-white text-base mt-0.5">
                  {(inspectSnapshot.metrics.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div className="p-2 bg-slate-900 rounded border border-white/5">
                <span className="text-slate-500 text-[10px]">F1 SCORE</span>
                <div className="font-bold text-neon-mint text-base mt-0.5">
                  {inspectSnapshot.metrics.f1.toFixed(4)}
                </div>
              </div>
              <div className="p-2 bg-slate-900 rounded border border-white/5">
                <span className="text-slate-500 text-[10px]">LATENCY P99</span>
                <div className="font-bold text-yellow-400 text-base mt-0.5">
                  {inspectSnapshot.metrics.latency_p99}ms
                </div>
              </div>
            </div>

            <div className="space-y-1">
              <span className="font-semibold text-slate-300">Feature Statistics Metadata</span>
              <pre className="p-2 bg-slate-950 rounded border border-white/10 text-[11px] font-mono text-neon-blue max-h-40 overflow-auto">
                {JSON.stringify(inspectSnapshot.feature_stats, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

