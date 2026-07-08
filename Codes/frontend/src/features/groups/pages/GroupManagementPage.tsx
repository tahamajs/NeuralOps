import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  Card,
  Button,
  Input,
  Spinner,
  Table,
  Badge,
  Modal,
} from "../../../components/ui";
import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";

export default function GroupManagementPage() {
  const queryClient = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [newGroupName, setNewGroupName] = useState("");
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);
  const [userToAdd, setUserToAdd] = useState("");

  const { data: groups, isLoading } = useQuery({
    queryKey: ["groups"],
    queryFn: async () => (await api.get("/groups/")).data,
  });
  const { data: groupDetail } = useQuery({
    queryKey: ["group", selectedGroup],
    queryFn: async () => (await api.get(`/groups/${selectedGroup}/`)).data,
    enabled: !!selectedGroup,
  });

  const createGroup = useMutation({
    mutationFn: () => api.post("/groups/", { name: newGroupName }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      setShowCreate(false);
      setNewGroupName("");
    },
  });
  const addUser = useMutation({
    mutationFn: () =>
      api.post(`/groups/${selectedGroup}/users/`, { username: userToAdd }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["group", selectedGroup] });
      setUserToAdd("");
    },
  });
  const removeUser = useMutation({
    mutationFn: (userId: number) =>
      api.delete(`/groups/${selectedGroup}/users/${userId}/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["group", selectedGroup] }),
  });
  const deleteGroup = useMutation({
    mutationFn: () => api.delete(`/groups/${selectedGroup}/`),
    onSuccess: () => {
      setSelectedGroup(null);
      queryClient.invalidateQueries({ queryKey: ["groups"] });
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <div className="flex justify-between">
        <h1 className="text-2xl font-bold text-white">Fair‑Share Groups</h1>
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus size={14} /> Create Group
        </Button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <h2 className="text-sm font-bold text-white mb-3">All Groups</h2>
          {groups?.map((group: any) => (
            <div
              key={group.name}
              className={`p-2 rounded cursor-pointer ${selectedGroup === group.name ? "bg-neon-blue/20" : "hover:bg-white/5"}`}
              onClick={() => setSelectedGroup(group.name)}
            >
              <div className="font-bold text-white">{group.name}</div>
              <div className="text-xs text-slate-400">
                {group.user_count} users · {group.active_jobs} jobs
              </div>
            </div>
          ))}
        </Card>

        {selectedGroup && groupDetail && (
          <Card>
            <div className="flex justify-between mb-3">
              <h2 className="text-sm font-bold text-white">{selectedGroup}</h2>
              <Button
                size="sm"
                variant="danger"
                onClick={() => deleteGroup.mutate()}
              >
                Delete Group
              </Button>
            </div>
            <div className="space-y-2">
              <div className="flex gap-2">
                <Input
                  placeholder="Username"
                  value={userToAdd}
                  onChange={(e) => setUserToAdd(e.target.value)}
                />
                <Button size="sm" onClick={() => addUser.mutate()}>
                  Add User
                </Button>
              </div>
              <Table>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Jobs</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {groupDetail.users?.map((u: any) => (
                    <tr key={u.id}>
                      <td>{u.username}</td>
                      <td>{u.job_count}</td>
                      <td>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeUser.mutate(u.id)}
                        >
                          <Trash2 size={12} />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="mt-3 text-xs text-slate-400">
                Quota: {groupDetail.quota_cpu} CPU, {groupDetail.quota_gpu} GPU
              </div>
            </div>
          </Card>
        )}
      </div>

      <Modal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="Create Group"
      >
        <Input
          label="Group Name"
          value={newGroupName}
          onChange={(e) => setNewGroupName(e.target.value)}
        />
        <Button className="mt-3" onClick={() => createGroup.mutate()}>
          Create
        </Button>
      </Modal>
    </div>
  );
}
