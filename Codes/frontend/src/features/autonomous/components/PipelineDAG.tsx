import { useEffect, useCallback } from "react";
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";
import { PipelineStep } from "../types";

interface PipelineDAGProps {
  steps: PipelineStep[];
  onNodeClick: (nodeId: string) => void;
}

const getStatusColor = (status: string) => {
  switch (status) {
    case "running":
      return "#38bdf8"; // neon blue
    case "completed":
      return "#00ff9d"; // neon mint
    case "failed":
      return "#f43f5e"; // rose
    case "skipped":
      return "#64748b"; // slate
    default:
      return "#334155";
  }
};

const getStatusBorder = (status: string) => {
  switch (status) {
    case "running":
      return "2px solid #38bdf8";
    case "completed":
      return "2px solid #00ff9d";
    case "failed":
      return "2px solid #f43f5e";
    default:
      return "1px solid rgba(255, 255, 255, 0.15)";
  }
};

const layoutGraph = (nodes: Node[], edges: Edge[]) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: "TB", ranksep: 60, nodesep: 40 });

  nodes.forEach((node) =>
    dagreGraph.setNode(node.id, { width: 240, height: 75 })
  );
  edges.forEach((edge) => dagreGraph.setEdge(edge.source, edge.target));

  try {
    dagre.layout(dagreGraph);
  } catch (e) {
    console.warn("Dagre layout error:", e);
  }

  return nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id) || { x: 100, y: 100 };
    return {
      ...node,
      position: {
        x: (nodeWithPosition.x || 100) - 120,
        y: (nodeWithPosition.y || 100) - 37,
      },
    };
  });
};

export function PipelineDAG({ steps = [], onNodeClick }: PipelineDAGProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    const safeSteps = Array.isArray(steps) ? steps : [];
    if (safeSteps.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const newNodes: Node[] = safeSteps.map((step) => ({
      id: step.id,
      type: "default",
      data: {
        label: (
          <div className="text-left font-sans select-none">
            <div className="font-bold text-white text-xs truncate">
              {step.name || "Unnamed Step"}
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono mt-1 text-slate-300">
              <span className="capitalize">{step.status}</span>
              <span>{step.duration !== undefined ? `${step.duration}s` : ""}</span>
            </div>
          </div>
        ),
      },
      position: { x: 0, y: 0 },
      style: {
        background: "rgba(15, 23, 42, 0.85)",
        border: getStatusBorder(step.status),
        borderRadius: "12px",
        padding: "10px 14px",
        width: "240px",
        boxShadow:
          step.status === "running"
            ? "0 0 15px rgba(56, 189, 248, 0.3)"
            : "0 4px 12px rgba(0, 0, 0, 0.4)",
      },
    }));

    const newEdges: Edge[] = safeSteps.flatMap((step) =>
      (step.dependencies || []).map((depId) => ({
        id: `${depId}-${step.id}`,
        source: depId,
        target: step.id,
        type: "smoothstep",
        animated: step.status === "running",
        style: {
          stroke: step.status === "running" ? "#38bdf8" : "#475569",
          strokeWidth: 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: step.status === "running" ? "#38bdf8" : "#475569",
        },
      }))
    );

    const layoutedNodes = layoutGraph(newNodes, newEdges);
    setNodes(layoutedNodes);
    setEdges(newEdges);
  }, [steps, setNodes, setEdges]);

  const onNodeClickHandler = useCallback(
    (_: any, node: Node) => onNodeClick?.(node.id),
    [onNodeClick]
  );

  return (
    <div className="w-full h-full min-h-[450px]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClickHandler}
        fitView
      >
        <Background color="#334155" gap={16} size={1} />
        <Controls className="bg-slate-900 border-white/10 fill-white" />
        <MiniMap
          nodeColor={(n: any) => getStatusColor(n.data?.status || "idle")}
          className="bg-slate-950 border border-white/10 rounded-xl"
        />
      </ReactFlow>
    </div>
  );
}
