import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  Card,
  Button,
  Spinner,
  Table,
  Badge,
  Input,
  Modal,
} from "../../../components/ui";
import { useState } from "react";
import { Plus, Trash2, Heartbeat } from "lucide-react";

export default function FederationDashboardPage() {
  const queryClient = useQueryClient();
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  const [showAddNode, setShowAddNode] = useState(false);
  const [nodeName, setNodeName] = useState("");

  const { data: clusters, isLoading } = useQuery({
    queryKey: ["federation-clusters"],
    queryFn: async () => (await api.get("/clusters/")).data,
  });
  const { data: tasks } = useQuery({
    queryKey: ["federation-tasks"],
    queryFn: async () => (await api.get("/tasks/")).data,
  });

  const addNode = useMutation({
    mutationFn: () =>
      api.post(`/clusters/${selectedCluster}/nodes/`, { name: nodeName }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["federation-clusters"] });
      setShowAddNode(false);
      setNodeName("");
    },
  });
  const removeNode = useMutation({
    mutationFn: (nodeId: string) =>
      api.delete(`/clusters/${selectedCluster}/nodes/${nodeId}/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["federation-clusters"] }),
  });
  const launchTask = useMutation({
    mutationFn: (taskId: string) => api.post(`/tasks/${taskId}/launch/`),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">
        GPU Federation Dashboard
      </h1>

      {/* Clusters & Nodes */}
      <Card>
        <div className="flex justify-between mb-3">
          <h2 className="text-sm font-bold text-white">Clusters</h2>
          <Button size="sm" variant="ghost">
            Register New Cluster
          </Button>
        </div>
        {clusters?.map((cluster: any) => (
          <div
            key={cluster.id}
            className="mb-4 border border-white/10 rounded-lg p-3"
          >
            <div className="flex justify-between items-center">
              <h3 className="font-bold text-white">{cluster.name}</h3>
              <Button size="sm" onClick={() => setSelectedCluster(cluster.id)}>
                Manage
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
              <span>Nodes: {cluster.nodes?.length}</span>
              <span>
                Status: <Badge variant={cluster.status}>{cluster.status}</Badge>
              </span>
            </div>
            {selectedCluster === cluster.id && (
              <div className="mt-3 pl-3 border-l-2 border-neon-blue">
                <div className="flex gap-2 mb-2">
                  <Button size="sm" onClick={() => setShowAddNode(true)}>
                    <Plus size={12} /> Add Node
                  </Button>
                </div>
                {cluster.nodes?.map((node: any) => (
                  <div
                    key={node.id}
                    className="flex justify-between items-center py-1"
                  >
                    <span>
                      {node.name} – {node.status}
                    </span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => removeNode.mutate(node.id)}
                    >
                      <Trash2 size={12} />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </Card>

      {/* Tasks */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Federated Tasks</h2>
        <Table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Assigned Node</th>
              <th>Result</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {tasks?.map((task: any) => (
              <tr key={task.id}>
                <td className="font-mono">{task.id.slice(0, 8)}</td>
                <td>{task.task_type}</td>
                <td>
                  <Badge variant={task.status}>{task.status}</Badge>
                </td>
                <td>{task.assigned_node || "-"}</td>
                <td>{task.result ? "✅" : "⏳"}</td>
                <td>
                  {task.status === "pending" && (
                    <Button
                      size="sm"
                      onClick={() => launchTask.mutate(task.id)}
                    >
                      Launch
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      <Modal
        open={showAddNode}
        onClose={() => setShowAddNode(false)}
        title="Add Node"
      >
        <Input
          label="Node Name"
          value={nodeName}
          onChange={(e) => setNodeName(e.target.value)}
        />
        <Button className="mt-3" onClick={() => addNode.mutate()}>
          Add
        </Button>
      </Modal>
    </div>
  );
}
