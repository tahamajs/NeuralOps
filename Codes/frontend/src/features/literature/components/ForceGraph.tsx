import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { PaperNode, CitationLink } from "../types";

interface ForceGraphProps {
  nodes: PaperNode[];
  links: CitationLink[];
  onNodeClick?: (node: PaperNode) => void;
  height?: number;
}

export function ForceGraph({
  nodes = [],
  links = [],
  onNodeClick,
  height = 500,
}: ForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 700, height });

  // Measure parent container dynamically
  useEffect(() => {
    if (!containerRef.current) return;
    const updateSize = () => {
      if (containerRef.current) {
        const { clientWidth } = containerRef.current;
        setDimensions({
          width: clientWidth || 700,
          height: height || 500,
        });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [height]);

  // Make defensive shallow copies for D3 mutation
  const safeNodes = (nodes || []).map((n) => ({
    ...n,
    id: String(n.id || Math.random()),
    title: n.title || "Untitled Paper",
    citations: Number(n.citations) || 0,
    group: n.group || "core",
  }));

  const safeLinks = (links || []).map((l) => ({
    ...l,
    source: String(l.source),
    target: String(l.target),
    type: l.type || "references",
  }));

  useEffect(() => {
    if (graphRef.current) {
      try {
        if (typeof graphRef.current.d3Force === "function") {
          graphRef.current.d3Force("charge")?.strength?.(-250);
          graphRef.current.d3Force("link")?.distance?.(90);
        }
        if (typeof graphRef.current.zoomToFit === "function") {
          setTimeout(() => {
            try {
              graphRef.current?.zoomToFit(400);
            } catch (e) {}
          }, 300);
        }
      } catch (e) {
        console.warn("Force graph force configuration notice:", e);
      }
    }
  }, [nodes, links]);

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-xl bg-slate-950/80">
      <ForceGraph2D
        ref={graphRef}
        graphData={{ nodes: safeNodes, links: safeLinks }}
        nodeLabel={(node: any) => `${node.title || "Paper"} (${node.year || 2024}) · ${node.citations || 0} cites`}
        nodeColor={(node: any) => {
          if (node.group === "core" || node.type === "source") return "#38bdf8"; // neon blue
          if (node.group === "influential" || node.citations > 500) return "#00ff9d"; // neon mint
          if (node.group === "emerging") return "#fbbf24"; // yellow
          return "#94a3b8"; // slate
        }}
        nodeVal={(node: any) => Math.max(3, Math.min(18, Math.log((node.citations || 1) + 10) * 3))}
        linkWidth={(link: any) => (link.type === "references" ? 1 : 2)}
        linkColor={(link: any) => (link.type === "references" ? "rgba(100, 116, 139, 0.4)" : "rgba(56, 189, 248, 0.6)")}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        onNodeClick={(node: any) => onNodeClick?.(node as PaperNode)}
        cooldownTicks={80}
        width={dimensions.width}
        height={dimensions.height}
      />
    </div>
  );
}
