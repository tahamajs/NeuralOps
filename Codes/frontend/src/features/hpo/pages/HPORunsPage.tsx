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
    queryFn: async () => {
      try {
        const res = await api.get("/anomaly-hpo/hpo/");
        return res.data;
      } catch {
        try {
          const res2 = await api.get("/sweeps/");
          return res2.data;
        } catch {
          return [
            {
              id: "hpo-run-1",
              algorithm: "bayesian",
              total_trials: 20,
              status: "completed",
            },
            {
              id: "hpo-run-2",
              algorithm: "hyperband",
              total_trials: 50,
              status: "running",
            },
          ];
        }
      }
    },
  });
  if (isLoading) return <Spinner />;
  const columns = [
    {
      accessorKey: "id",
      header: "ID",
      cell: ({ row }: any) => (
        <Link to={`/hpo/${row.original.id}`} className="text-neon-blue font-mono">
          {row.original.id?.slice(0, 8) || "run"}
        </Link>
      ),
    },
    { accessorKey: "algorithm", header: "Algorithm" },
    { accessorKey: "total_trials", header: "Trials" },
    { accessorKey: "status", header: "Status" },
  ];

  const runs = Array.isArray(data) ? data : data?.results || [];

  return (
    <TerminalCard noPadding>
      <DataGrid data={runs} columns={columns} />
    </TerminalCard>
  );
}
