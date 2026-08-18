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

  const experimentList = Array.isArray(experiments) ? experiments : [];

  const numericParams = useMemo(() => {
    if (!experimentList.length) return ["learning_rate", "batch_size", "dropout"];
    const paramSet = new Set<string>();
    experimentList.forEach((exp) => {
      if (exp?.parameters && typeof exp.parameters === "object") {
        Object.entries(exp.parameters).forEach(([key, val]) => {
          if (typeof val === "number" && !isNaN(val)) paramSet.add(key);
        });
      }
    });
    const arr = Array.from(paramSet);
    return arr.length > 0 ? arr : ["learning_rate", "batch_size", "dropout"];
  }, [experimentList]);

  const numericMetrics = useMemo(() => {
    if (!experimentList.length) return ["accuracy", "loss", "f1"];
    const metricSet = new Set<string>();
    experimentList.forEach((exp) => {
      if (exp?.metrics && typeof exp.metrics === "object") {
        Object.entries(exp.metrics).forEach(([key, val]) => {
          if (typeof val === "number" && !isNaN(val)) metricSet.add(key);
        });
      }
    });
    const arr = Array.from(metricSet);
    return arr.length > 0 ? arr : ["accuracy", "loss", "f1"];
  }, [experimentList]);

  const activeMetric = numericMetrics.includes(selectedMetric) ? selectedMetric : numericMetrics[0];
  const activeX = numericParams.includes(xParam) ? xParam : numericParams[0];
  const activeY = numericParams.includes(yParam) ? yParam : (numericParams[1] || numericParams[0]);
  const activeZ = numericParams.includes(zParam) ? zParam : (numericParams[2] || numericParams[0]);
  const activeColor = numericMetrics.includes(colorMetric) ? colorMetric : numericMetrics[0];

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
            value={activeMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          />
          <Select
            label="3D X‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={activeX}
            onChange={(e) => setXParam(e.target.value)}
          />
          <Select
            label="3D Y‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={activeY}
            onChange={(e) => setYParam(e.target.value)}
          />
          <Select
            label="3D Z‑Axis"
            options={numericParams.map((p) => ({ value: p, label: p }))}
            value={activeZ}
            onChange={(e) => setZParam(e.target.value)}
          />
          <Select
            label="Color by"
            options={numericMetrics.map((m) => ({ value: m, label: m }))}
            value={activeColor}
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
                experiments={experimentList}
                targetMetric={activeMetric}
              />
            </TerminalCard>
          </TabPanel>
          <TabPanel value="scatter3d">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              <ThreeDScatterPlot
                experiments={experimentList}
                xParam={activeX}
                yParam={activeY}
                zParam={activeZ}
                colorMetric={activeColor}
              />
            </TerminalCard>
          </TabPanel>
          <TabPanel value="correlation">
            <TerminalCard className="p-2" style={{ height: "500px" }}>
              <CorrelationMatrix
                correlations={(correlations as any) || []}
                targetMetric={activeMetric}
              />
            </TerminalCard>
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
