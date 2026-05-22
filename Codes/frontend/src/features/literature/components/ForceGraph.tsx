import { useEffect, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { PaperNode, CitationLink } from '../types'

interface ForceGraphProps {
  nodes: PaperNode[]
  links: CitationLink[]
  onNodeClick?: (node: PaperNode) => void
  height?: number
}

export function ForceGraph({ nodes, links, onNodeClick, height = 500 }: ForceGraphProps) {
  const graphRef = useRef<any>()

  useEffect(() => {
    if (graphRef.current) {
      graphRef.current.d3Force('charge').strength(-300)
      graphRef.current.d3Force('link').distance(100)
      graphRef.current.zoomToFit(400)
    }
  }, [nodes, links])

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={{ nodes, links }}
      nodeLabel="title"
      nodeColor={(node: any) => node.group === 'core' ? '#3b82f6' : node.group === 'influential' ? '#00ff9d' : '#94a3b8'}
      nodeVal={(node: any) => Math.log(node.citations + 10) * 3}
      linkWidth={(link: any) => link.type === 'references' ? 1 : 2}
      linkColor={(link: any) => link.type === 'references' ? '#64748b' : '#3b82f6'}
      onNodeClick={onNodeClick}
      cooldownTicks={100}
      width={window.innerWidth - 100}
      height={height}
    />
  )
}