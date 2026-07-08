import { useEffect, useRef, useCallback } from "react";
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Position,
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
      return "#3b82f6";
    case "completed":
      return "#10b981";
    case "failed":
      return "#ef4444";
    case "skipped":
      return "#94a3b8";
    default:
      return "#475569";
  }
};

const layoutGraph = (nodes: Node[], edges: Edge[]) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: "TB", ranksep: 50, nodesep: 30 });
  nodes.forEach((node) =>
    dagreGraph.setNode(node.id, { width: 250, height: 80 }),
  );
  edges.forEach((edge) => dagreGraph.setEdge(edge.source, edge.target));
  dagre.layout(dagreGraph);
  return nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: { x: nodeWithPosition.x - 125, y: nodeWithPosition.y - 40 },
    };
  });
};

export function PipelineDAG({ steps, onNodeClick }: PipelineDAGProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    const newNodes: Node[] = steps.map((step) => ({
      id: step.id,
      type: "default",
      data: { label: step.name, status: step.status, duration: step.duration },
      position: { x: 0, y: 0 },
      style: {
        background: getStatusColor(step.status),
        color: "#fff",
        borderRadius: "8px",
        padding: "8px",
        width: "250px",
      },
    }));
    const newEdges: Edge[] = steps.flatMap((step) =>
      step.dependencies.map((depId) => ({
        id: `${depId}-${step.id}`,
        source: depId,
        target: step.id,
        type: "smoothstep",
        animated: step.status === "running",
      })),
    );
    const layoutedNodes = layoutGraph(newNodes, newEdges);
    setNodes(layoutedNodes);
    setEdges(newEdges);
  }, [steps]);

  const onNodeClickHandler = useCallback(
    (_, node) => onNodeClick(node.id),
    [onNodeClick],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClickHandler}
      fitView
    >
      <Background />
      <Controls />
      <MiniMap />
    </ReactFlow>
  );
}
