import { useQuery } from "@tanstack/react-query";
import { hpoAPI } from "../services/api";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";

export function TrialTable({ sweepId }: { sweepId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["trials", sweepId],
    queryFn: () => hpoAPI.getTrials(sweepId),
  });

  if (isLoading) return <Spinner size="sm" />;

  const columns = [
    { accessorKey: "number", header: "#" },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }: any) => (
        <Badge
          variant={
            row.original.status === "completed"
              ? "success"
              : row.original.status === "running"
                ? "info"
                : "failed"
          }
        >
          {row.original.status}
        </Badge>
      ),
    },
    {
      accessorKey: "objective_value",
      header: "Value",
      cell: ({ row }: any) => row.original.objective_value?.toFixed(4) || "-",
    },
    {
      accessorKey: "params",
      header: "Parameters",
      cell: ({ row }: any) =>
        Object.entries(row.original.params)
          .slice(0, 3)
          .map(([k, v]) => `${k}=${v}`)
          .join(", "),
    },
    {
      accessorKey: "duration_seconds",
      header: "Duration",
      cell: ({ row }: any) => `${row.original.duration_seconds}s`,
    },
    {
      accessorKey: "created_at",
      header: "Created",
      cell: ({ row }: any) =>
        new Date(row.original.created_at).toLocaleString(),
    },
  ];

  return <DataGrid data={data?.results || []} columns={columns} />;
}
