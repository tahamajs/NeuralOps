import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  TerminalCard,
  Button,
  Spinner,
  DataGrid,
} from "../../../components/ui";
import { Link } from "react-router-dom";

export default function HPORunsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["hpo-runs"],
    queryFn: () => api.get("/anomaly-hpo/hpo/").then((res) => res.data),
  });
  if (isLoading) return <Spinner />;
  const columns = [
    {
      accessorKey: "id",
      header: "ID",
      cell: ({ row }: any) => (
        <Link to={`/hpo/${row.original.id}`} className="text-neon-blue">
          {row.original.id.slice(0, 8)}
        </Link>
      ),
    },
    { accessorKey: "algorithm", header: "Algorithm" },
    { accessorKey: "total_trials", header: "Trials" },
    { accessorKey: "status", header: "Status" },
  ];
  return (
    <TerminalCard noPadding>
      <DataGrid data={data?.results || []} columns={columns} />
    </TerminalCard>
  );
}
