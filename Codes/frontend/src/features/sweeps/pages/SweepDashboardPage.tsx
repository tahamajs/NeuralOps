import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  Card,
  Button,
  Spinner,
  Badge,
  Table,
  Tabs,
  TabList,
  Tab,
  TabPanel,
} from "../../../components/ui";
import { BarChart3, TrendingUp, GitBranch, Download } from "lucide-react";
import Plot from "react-plotly.js";

export default function SweepDashboardPage() {
  const { sweepId } = useParams();
  const queryClient = useQueryClient();
  const { data: sweep, isLoading } = useQuery({
    queryKey: ["sweep", sweepId],
    queryFn: async () => (await api.get(`/sweeps/${sweepId}/`)).data,
    enabled: !!sweepId,
  });
  const { data: trials } = useQuery({
    queryKey: ["sweep-trials", sweepId],
    queryFn: async () => (await api.get(`/sweeps/${sweepId}/trials/`)).data,
    enabled: !!sweepId,
  });
  const { data: importance } = useQuery({
    queryKey: ["sweep-importance", sweepId],
    queryFn: async () => (await api.get(`/sweeps/${sweepId}/importance/`)).data,
    enabled: !!sweepId,
  });
  const { data: pareto } = useQuery({
    queryKey: ["sweep-pareto", sweepId],
    queryFn: async () => (await api.get(`/sweeps/${sweepId}/pareto/`)).data,
    enabled: !!sweepId,
  });

  const stopSweep = useMutation({
    mutationFn: () => api.post(`/sweeps/${sweepId}/stop/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["sweep", sweepId] }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-white">
            {sweep?.name || "Sweep"}
          </h1>
          <p className="text-sm text-slate-400">
            {sweep?.algorithm} · {sweep?.objective}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="danger" size="sm" onClick={() => stopSweep.mutate()}>
            Stop Sweep
          </Button>
          <Link to={`/sweeps/${sweepId}/export`}>
            <Button size="sm" variant="ghost">
              <Download size={14} />
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <div className="text-2xl">{sweep?.total_trials || 0}</div>
          <div className="text-xs">Total Trials</div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl text-green-400">
            {sweep?.completed_trials || 0}
          </div>
          <div className="text-xs">Completed</div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl text-blue-400">
            {sweep?.running_trials || 0}
          </div>
          <div className="text-xs">Running</div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl text-yellow-400">
            {sweep?.best_score?.toFixed(4) || "-"}
          </div>
          <div className="text-xs">Best Score</div>
        </Card>
      </div>

      <Tabs defaultValue="trials">
        <TabList>
          <Tab value="trials">Trials</Tab>
          <Tab value="importance">HP Importance</Tab>
          <Tab value="pareto">Pareto Front</Tab>
          <Tab value="parallel">Parallel Coords</Tab>
        </TabList>
        <TabPanel value="trials">
          <Table>
            <thead>
              <tr>
                <th>#</th>
                <th>State</th>
                <th>Score</th>
                <th>Params</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {trials?.map((t: any) => (
                <tr key={t.number}>
                  <td>{t.number}</td>
                  <td>
                    <Badge variant={t.state}>{t.state}</Badge>
                  </td>
                  <td>{t.score?.toFixed(4) || "-"}</td>
                  <td className="font-mono text-xs">
                    {Object.entries(t.params || {})
                      .slice(0, 3)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}
                  </td>
                  <td>{t.duration_seconds}s</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </TabPanel>
        <TabPanel value="importance">
          {importance?.map((hp: any) => (
            <div key={hp.name} className="mb-3">
              <div className="flex justify-between text-sm">
                <span>{hp.name}</span>
                <span>{hp.importance.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-white/10 h-2 rounded">
                <div
                  className="bg-neon-blue h-2 rounded"
                  style={{ width: `${hp.importance}%` }}
                />
              </div>
            </div>
          ))}
        </TabPanel>
        <TabPanel value="pareto">
          {pareto && (
            <Plot
              data={[
                {
                  x: pareto.x,
                  y: pareto.y,
                  type: "scatter",
                  mode: "markers",
                  marker: { color: "cyan" },
                },
              ]}
              layout={{ height: 400 }}
            />
          )}
        </TabPanel>
        <TabPanel value="parallel">
          {trials && trials.length > 0 ? (
            <Plot
              data={[
                {
                  type: "parcoords",
                  line: {
                    color: trials.map((t: any) => t.score || 0),
                    colorscale: "Jet",
                  },
                  dimensions: Object.keys(trials[0]?.params || {})
                    .slice(0, 5)
                    .map((key) => ({
                      label: key,
                      values: trials.map(
                        (t: any) => parseFloat(t.params[key]) || 0,
                      ),
                    }))
                    .concat([
                      {
                        label: "Score",
                        values: trials.map((t: any) => t.score || 0),
                      },
                    ]),
                } as any,
              ]}
              layout={{
                height: 400,
                autosize: true,
                paper_bgcolor: "rgba(0,0,0,0)",
                font: { color: "#94a3b8" },
              }}
              style={{ width: "100%" }}
            />
          ) : (
            <div className="text-center text-slate-500 py-10 border border-dashed border-white/10 rounded-xl">
              Not enough trial data to render parallel coordinates.
            </div>
          )}
        </TabPanel>
      </Tabs>
    </div>
  );
}
