// src/features/communityLab/components/WorkflowBuilder.tsx
import { useState, useCallback } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  Connection,
} from "reactflow";
import "reactflow/dist/style.css";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Save, Play, Download } from "lucide-react";
import { communityLabAPI } from "../services/api";
import { useToast } from "../../../components/ui/Toast";

const nodeTypes = {
  default: ({ data }: any) => (
    <div className="px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-white text-sm">
      <div className="font-bold">{data.label}</div>
      {data.config && (
        <div className="text-xs text-slate-400">{data.config.command}</div>
      )}
    </div>
  ),
};

export function WorkflowBuilder({ labId }: { labId: string }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [config, setConfig] = useState({ name: "", command: "", image: "" });
  const { success, error } = useToast();

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );
  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node);
    setConfig(
      node.data.config || { name: node.data.label, command: "", image: "" },
    );
    setShowConfigModal(true);
  }, []);
  const addNode = () => {
    const id = `node-${Date.now()}`;
    setNodes((nds) => [
      ...nds,
      {
        id,
        type: "default",
        data: {
          label: "New Task",
          config: { name: "New Task", command: "", image: "" },
        },
        position: { x: Math.random() * 400, y: Math.random() * 400 },
      },
    ]);
  };
  const updateNode = () => {
    if (selectedNode) {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === selectedNode.id
            ? { ...n, data: { ...n.data, label: config.name, config } }
            : n,
        ),
      );
      setShowConfigModal(false);
    }
  };
  const exportYAML = () => {
    const workflow = {
      nodes: nodes.map((n) => ({ name: n.data.label, config: n.data.config })),
      edges: edges.map((e) => ({ from: e.source, to: e.target })),
    };
    const yaml = `# Workflow exported from NeuralOps\n${JSON.stringify(workflow, null, 2)}`;
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workflow.yaml";
    a.click();
    URL.revokeObjectURL(url);
    success("Workflow exported");
  };

  return (
    <TerminalCard className="h-[500px] flex flex-col overflow-hidden">
      <div className="flex justify-between items-center p-2 border-b border-white/10">
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={addNode}>
            Add Node
          </Button>
          <Button size="sm" variant="ghost" onClick={exportYAML}>
            <Download size={14} className="mr-1" /> Export
          </Button>
        </div>
      </div>
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
      <Modal
        open={showConfigModal}
        onClose={() => setShowConfigModal(false)}
        title="Configure Task"
      >
        <div className="space-y-3">
          <Input
            label="Task Name"
            value={config.name}
            onChange={(e) => setConfig({ ...config, name: e.target.value })}
          />
          <Input
            label="Command"
            value={config.command}
            onChange={(e) => setConfig({ ...config, command: e.target.value })}
            placeholder="python train.py"
          />
          <Input
            label="Docker Image"
            value={config.image}
            onChange={(e) => setConfig({ ...config, image: e.target.value })}
            placeholder="pytorch/pytorch:latest"
          />
          <Button onClick={updateNode}>Save</Button>
        </div>
      </Modal>
    </TerminalCard>
  );
}
