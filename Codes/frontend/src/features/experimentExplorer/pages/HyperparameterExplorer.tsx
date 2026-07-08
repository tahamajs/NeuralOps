// src/features/experimentExplorer/pages/HyperparameterExplorer.tsx
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { experimentExplorerAPI } from "../services/api";
import { ParallelCoordinatesPlot } from "../components/ParallelCoordinatesPlot";
import { ThreeDScatterPlot } from "../components/ThreeDScatterPlot";
import { CorrelationMatrix } from "../components/CorrelationMatrix";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { BarChart3, GitBranch, Maximize2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function HyperparameterExplorer({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedMetric, setSelectedMetric] = useState("accuracy");
  const [xParam, setXParam] = useState("learning_rate");
  const [yParam, setYParam] = useState("batch_size");
  const [zParam, setZParam] = useState("dropout");
  const [colorMetric, setColorMetric] = useState("accuracy");
  const [activeTab, setActiveTab] = useState("parallel");
  const { success, error } = useToast();

  const {
    data: experiments,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["explorer-experiments"],
    queryFn: () => experimentExplorerAPI.getExperiments({ limit: 500 }),
  });

  const { data: correlations } = useQuery({
    queryKey: ["parameter-correlations", selectedMetric],
    queryFn: () =>
      experimentExplorerAPI.getParameterCorrelations(selectedMetric),
    enabled: !!selectedMetric,
  });

  const numericParams = useMemo(() => {
    if (!experiments?.length) return [];
    const firstExp = experiments[0];
    return Object.keys(firstExp.parameters).filter(
      (key) => typeof firstExp.parameters[key] === "number",
    );
  }, [experiments]);

  const numericMetrics = useMemo(() => {
    if (!experiments?.length) return [];
    const firstExp = experiments[0];
    return Object.keys(firstExp.metrics).filter(
      (key) => typeof firstExp.metrics[key] === "number",
    );
  }, [experiments]);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <BarChart3 size={20} className="text-neon-mint" /> Hyperparameter
            Explorer
          </h1>
          <p className="text-xs text-slate-400">
            Interactive 3D visualisation of experiment parameters
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <GitBranch size={14} className="mr-1" /> Refresh
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-3 gap-3">
          <Select
            label="Target Metric"
            options={numericMetrics.map((m) => ({ value: m, label: m }))}
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          />
          <Select
            label="3D X‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={xParam}
            onChange={(e) => setXParam(e.target.value)}
          />
          <Select
            label="3D Y‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={yParam}
            onChange={(e) => setYParam(e.target.value)}
          />
          <Select
            label="3D Z‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={zParam}
            onChange={(e) => setZParam(e.target.value)}
          />
          <Select
            label="Color by"
            options={numericMetrics.map((m) => ({ value: m, label: m }))}
            value={colorMetric}
            onChange={(e) => setColorMetric(e.target.value)}
          />
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabList>
            <Tab value="parallel">Parallel Coordinates</Tab>
            <Tab value="scatter3d">3D Scatter Plot</Tab>
            <Tab value="correlation">Correlation Matrix</Tab>
          </TabList>
          <TabPanel value="parallel">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              <ParallelCoordinatesPlot
                experiments={experiments || []}
                targetMetric={selectedMetric}
              />
            </TerminalCard>
          </TabPanel>
          <TabPanel value="scatter3d">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              <ThreeDScatterPlot
                experiments={experiments || []}
                xParam={xParam}
                yParam={yParam}
                zParam={zParam}
                colorMetric={colorMetric}
              />
            </TerminalCard>
          </TabPanel>
          <TabPanel value="correlation">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              <CorrelationMatrix
                correlations={correlations || []}
                targetMetric={selectedMetric}
              />
            </TerminalCard>
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
