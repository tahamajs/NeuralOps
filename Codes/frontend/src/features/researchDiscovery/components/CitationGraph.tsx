import ReactFlow, { Background, Controls, MiniMap, useNodesState, useEdgesState } from 'reactflow'
import 'reactflow/dist/style.css'

export function CitationGraph({ nodes, links }: any) {
  const flowNodes = nodes.map((n: any, i: number) => ({ id: n.id, data: { label: n.title.slice(0, 30) + '...' }, position: { x: i * 200, y: 100 }, style: { background: '#3b82f6', color: '#fff', padding: '8px', borderRadius: '8px' } }))
  const flowEdges = links.map((l: any) => ({ id: `${l.source}-${l.target}`, source: l.source, target: l.target, animated: true }))
  return <div className="h-[400px]"><ReactFlow nodes={flowNodes} edges={flowEdges} fitView><Background /><Controls /><MiniMap /></ReactFlow></div>
}