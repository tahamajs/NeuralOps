import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";

export function CitationGraph({ nodes = [], links = [] }: any) {
  const safeNodes = Array.isArray(nodes) ? nodes : [];
  const safeLinks = Array.isArray(links) ? links : [];

  const flowNodes = safeNodes.map((n: any, i: number) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const title = n?.title || `Paper #${i + 1}`;
    return {
      id: String(n?.id || i),
      data: {
        label: (
          <div className="text-left font-sans text-xs">
            <div className="font-bold text-white truncate max-w-[200px]" title={title}>
              {title}
            </div>
            <div className="text-[10px] text-slate-300 font-mono mt-0.5">
              {n?.year || 2024} · {n?.citations || 0} citations
            </div>
          </div>
        ),
      },
      position: { x: col * 260 + 50, y: row * 120 + 50 },
      style: {
        background: "rgba(15, 23, 42, 0.9)",
        border: "1px solid #38bdf8",
        borderRadius: "10px",
        padding: "10px 12px",
        width: "230px",
        boxShadow: "0 0 12px rgba(56, 189, 248, 0.2)",
      },
    };
  });

  const flowEdges = safeLinks.map((l: any, i: number) => ({
    id: `e-${l.source}-${l.target}-${i}`,
    source: String(l.source),
    target: String(l.target),
    type: "smoothstep",
    animated: true,
    style: { stroke: "#38bdf8", strokeWidth: 1.5 },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#38bdf8",
    },
  }));

  return (
    <div className="w-full h-[450px] bg-slate-950/60 rounded-xl overflow-hidden border border-white/10">
      <ReactFlow nodes={flowNodes} edges={flowEdges} fitView>
        <Background color="#334155" gap={16} size={1} />
        <Controls className="bg-slate-900 border-white/10 fill-white" />
        <MiniMap
          nodeColor="#38bdf8"
          className="bg-slate-950 border border-white/10 rounded-xl"
        />
      </ReactFlow>
    </div>
  );
}
