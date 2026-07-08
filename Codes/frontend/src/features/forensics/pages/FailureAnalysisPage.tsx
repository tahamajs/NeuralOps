import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { forensicsAPI } from "../services/api";
import { FixApplicationPanel } from "../components/FixApplicationPanel";
import { ErrorPatternList } from "../components/ErrorPatternList";
import { FailureStatsWidget } from "../components/FailureStatsWidget";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import {
  AlertTriangle,
  CheckCircle,
  Eye,
  RefreshCw,
  Bug,
  Shield,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function FailureAnalysisPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [statusFilter, setStatusFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState("");
  const [runIdSearch, setRunIdSearch] = useState("");
  const [selectedAnalysis, setSelectedAnalysis] =
    useState<FailureAnalysis | null>(null);
  const [showFixPanel, setShowFixPanel] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: [
      "failures",
      { status: statusFilter, severity: severityFilter, run_id: runIdSearch },
    ],
    queryFn: () =>
      forensicsAPI.getFailures({
        status: statusFilter || undefined,
        severity: severityFilter || undefined,
        run_id: runIdSearch || undefined,
      }),
  });

  const analyzeMutation = useMutation({
    mutationFn: (runId: string) => forensicsAPI.analyzeRun(runId),
    onSuccess: () => {
      refetch();
      success("Analysis started", "Check back in a moment.");
    },
    onError: () => error("Failed to trigger analysis"),
  });

  const resolveMutation = useMutation({
    mutationFn: (id: string) => forensicsAPI.resolveFailure(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["failures"] });
      success("Failure marked as resolved");
    },
  });

  const columns = [
    {
      accessorKey: "run_name",
      header: "Run",
      cell: ({ row }: any) => (
        <Link
          to={`/runs/${row.original.run_id}`}
          className="text-neon-blue hover:underline"
        >
          {row.original.run_name || row.original.run_id.slice(0, 8)}
        </Link>
      ),
    },
    {
      accessorKey: "error_type",
      header: "Error Type",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs">{row.original.error_type}</span>
      ),
    },
    {
      accessorKey: "severity",
      header: "Severity",
      cell: ({ row }: any) => (
        <Badge
          variant={
            row.original.severity === "high"
              ? "danger"
              : row.original.severity === "medium"
                ? "warning"
                : "info"
          }
        >
          {row.original.severity}
        </Badge>
      ),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }: any) => (
        <Badge
          variant={row.original.status === "resolved" ? "success" : "warning"}
        >
          {row.original.status}
        </Badge>
      ),
    },
    {
      accessorKey: "created_at",
      header: "Detected",
      cell: ({ row }: any) =>
        new Date(row.original.created_at).toLocaleString(),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }: any) => (
        <div className="flex gap-2">
          <button
            onClick={() => {
              setSelectedAnalysis(row.original);
              setShowFixPanel(true);
            }}
          >
            <Bug size={16} className="text-yellow-400" />
          </button>
          {row.original.status !== "resolved" && (
            <button onClick={() => resolveMutation.mutate(row.original.id)}>
              <CheckCircle size={16} className="text-green-400" />
            </button>
          )}
        </div>
      ),
    },
  ];

  const failures = data?.results || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <Shield size={20} className="text-red-400" /> Failure Forensics
        </h1>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              analyzeMutation.mutate(prompt("Enter Run ID to analyze:") || "")
            }
          >
            Analyze Run
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <FailureStatsWidget />
        <div className="flex gap-3 items-end">
          <Select
            label="Status"
            options={[
              { value: "", label: "All" },
              { value: "open", label: "Open" },
              { value: "analyzing", label: "Analyzing" },
              { value: "resolved", label: "Resolved" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-36"
          />
          <Select
            label="Severity"
            options={[
              { value: "", label: "All" },
              { value: "low", label: "Low" },
              { value: "medium", label: "Medium" },
              { value: "high", label: "High" },
            ]}
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="w-36"
          />
          <Input
            placeholder="Run ID"
            value={runIdSearch}
            onChange={(e) => setRunIdSearch(e.target.value)}
            className="w-48"
          />
          <Button size="sm" onClick={() => refetch()}>
            Search
          </Button>
        </div>
        <TerminalCard noPadding>
          <DataGrid data={failures} columns={columns} />
        </TerminalCard>
        <Tabs defaultValue="patterns">
          <TabList>
            <Tab value="patterns">Error Patterns</Tab>
          </TabList>
          <TabPanel value="patterns">
            <ErrorPatternList />
          </TabPanel>
        </Tabs>
      </div>
      {selectedAnalysis && (
        <FixApplicationPanel
          analysis={selectedAnalysis}
          open={showFixPanel}
          onClose={() => {
            setShowFixPanel(false);
            setSelectedAnalysis(null);
          }}
        />
      )}
    </div>
  );
}
