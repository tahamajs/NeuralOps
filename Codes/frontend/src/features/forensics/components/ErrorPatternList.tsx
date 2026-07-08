import { useQuery } from "@tanstack/react-query";
import { forensicsAPI } from "../services/api";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Spinner } from "../../../components/ui/Spinner";

export function ErrorPatternList() {
  const { data, isLoading } = useQuery({
    queryKey: ["error-patterns"],
    queryFn: () => forensicsAPI.getErrorPatterns(),
  });
  if (isLoading) return <Spinner size="sm" />;
  const columns = [
    { accessorKey: "pattern_name", header: "Pattern" },
    { accessorKey: "occurrences", header: "Occurrences" },
    {
      accessorKey: "last_seen",
      header: "Last Seen",
      cell: ({ row }: any) => new Date(row.original.last_seen).toLocaleString(),
    },
    { accessorKey: "common_fix", header: "Common Fix" },
  ];
  return <DataGrid data={data || []} columns={columns} />;
}
