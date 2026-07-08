import { DataGrid } from "../../../components/ui/DataGrid";
import { PerformanceSnapshot } from "../types";

export function SnapshotTable({
  snapshots,
}: {
  snapshots: PerformanceSnapshot[];
}) {
  const columns = [
    {
      accessorKey: "timestamp",
      header: "Timestamp",
      cell: ({ row }: any) => new Date(row.original.timestamp).toLocaleString(),
    },
    {
      accessorKey: "metrics.accuracy",
      header: "Accuracy",
      cell: ({ row }: any) => row.original.metrics.accuracy.toFixed(4),
    },
    {
      accessorKey: "metrics.precision",
      header: "Precision",
      cell: ({ row }: any) => row.original.metrics.precision.toFixed(4),
    },
    {
      accessorKey: "metrics.recall",
      header: "Recall",
      cell: ({ row }: any) => row.original.metrics.recall.toFixed(4),
    },
    {
      accessorKey: "metrics.latency_p99",
      header: "P99 Latency",
      cell: ({ row }: any) => `${row.original.metrics.latency_p99}ms`,
    },
    {
      accessorKey: "data_drift_score",
      header: "Data Drift",
      cell: ({ row }: any) => row.original.data_drift_score.toFixed(4),
    },
  ];
  return <DataGrid data={snapshots} columns={columns} />;
}
