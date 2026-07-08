// src/features/organizations/pages/OrganizationDetailPage.tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  Card,
  Button,
  Input,
  Modal,
  Select,
  Spinner,
  Avatar,
  Tabs,
  TabList,
  Tab,
  TabPanel,
} from "../../../components/ui";
import {
  Users,
  CreditCard,
  Settings,
  ClipboardList,
  FolderTree,
  UserPlus,
  Trash2,
  Crown,
  Mail,
} from "lucide-react";

export default function OrganizationDetailPage() {
  const { orgId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("members");
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState("");

  const { data: org, isLoading: orgLoading } = useQuery({
    queryKey: ["organization", orgId],
    queryFn: () => api.get(`/orgs/${orgId}/`).then((res) => res.data),
    enabled: !!orgId,
  });

  const { data: members, isLoading: membersLoading } = useQuery({
    queryKey: ["org-members", orgId],
    queryFn: () => api.get(`/orgs/${orgId}/members/`).then((res) => res.data),
    enabled: !!orgId,
  });

  const { data: teams } = useQuery({
    queryKey: ["org-teams", orgId],
    queryFn: () => api.get(`/orgs/${orgId}/teams/`).then((res) => res.data),
    enabled: !!orgId,
  });

  const { data: auditLog } = useQuery({
    queryKey: ["org-audit", orgId],
    queryFn: () => api.get(`/orgs/${orgId}/audit-log/`).then((res) => res.data),
    enabled: !!orgId && activeTab === "audit",
  });

  const inviteMember = useMutation({
    mutationFn: () =>
      api.post(`/orgs/${orgId}/invite/`, {
        email: inviteEmail,
        role: inviteRole,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["org-members", orgId] });
      setShowInviteModal(false);
      setInviteEmail("");
    },
  });

  const changeRole = useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: string }) =>
      api.post(`/orgs/${orgId}/members/${userId}/role/`, { role }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["org-members", orgId] }),
  });

  const removeMember = useMutation({
    mutationFn: (userId: string) =>
      api.delete(`/orgs/${orgId}/members/${userId}/remove/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["org-members", orgId] }),
  });

  const deleteOrg = useMutation({
    mutationFn: () => api.delete(`/orgs/${orgId}/`),
    onSuccess: () => navigate("/organizations"),
  });

  if (orgLoading || membersLoading) return <Spinner />;
  if (!org) return <div className="text-red-400">Organization not found</div>;

  const isOwner = org.current_user_role === "owner";

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        {org.avatar_url ? (
          <img
            src={org.avatar_url}
            className="w-16 h-16 rounded-xl object-cover"
          />
        ) : (
          <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-2xl text-white">
            {org.name[0]}
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold text-white">{org.name}</h1>
          <p className="text-slate-400">
            {org.description || "No description"}
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabList>
          <Tab value="members">Members</Tab>
          <Tab value="teams">Teams</Tab>
          <Tab value="billing">Billing</Tab>
          <Tab value="audit">Audit Log</Tab>
          <Tab value="settings">Settings</Tab>
        </TabList>

        <TabPanel value="members">
          <div className="flex justify-between mb-4">
            <h2 className="text-lg font-bold text-white">
              Members ({members?.length || 0})
            </h2>
            {isOwner && (
              <Button size="sm" onClick={() => setShowInviteModal(true)}>
                <UserPlus size={14} className="mr-1" /> Invite Member
              </Button>
            )}
          </div>
          <Card>
            <div className="divide-y divide-white/5">
              {members?.map((member) => (
                <div
                  key={member.id}
                  className="flex items-center justify-between py-3"
                >
                  <div className="flex items-center gap-3">
                    <Avatar src={member.avatar_url} size="sm" />
                    <div>
                      <div className="text-white">{member.username}</div>
                      <div className="text-xs text-slate-400">
                        {member.email}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {isOwner && member.role !== "owner" ? (
                      <select
                        value={member.role}
                        onChange={(e) =>
                          changeRole.mutate({
                            userId: member.user_id,
                            role: e.target.value,
                          })
                        }
                        className="bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
                      >
                        <option value="member">Member</option>
                        <option value="billing_contact">Billing Contact</option>
                      </select>
                    ) : (
                      <span className="text-xs capitalize">{member.role}</span>
                    )}
                    {isOwner && member.role !== "owner" && (
                      <button
                        onClick={() => removeMember.mutate(member.user_id)}
                        className="p-1 rounded hover:bg-white/10"
                      >
                        <Trash2 size={14} className="text-red-400" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </TabPanel>

        <TabPanel value="teams">
          <Button size="sm" className="mb-4">
            + Create Team
          </Button>
          <div className="grid md:grid-cols-2 gap-4">
            {teams?.map((team) => (
              <Card key={team.id}>
                <div className="font-bold text-white">{team.name}</div>
                <div className="text-slate-400 text-sm">
                  {team.member_count} members
                </div>
              </Card>
            ))}
          </div>
        </TabPanel>

        <TabPanel value="billing">
          <Card>
            <h3 className="text-sm font-bold text-white mb-2">
              Subscription Plan: {org.subscription_plan || "Free"}
            </h3>
            <Button>Upgrade Plan</Button>
          </Card>
        </TabPanel>

        <TabPanel value="audit">
          <Card>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {auditLog?.map((entry) => (
                <div
                  key={entry.id}
                  className="flex justify-between text-sm border-b border-white/5 py-2"
                >
                  <span>
                    {entry.user} {entry.action}
                  </span>
                  <span className="text-xs text-slate-500">
                    {new Date(entry.created_at).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </TabPanel>

        <TabPanel value="settings">
          <Card>
            <h3 className="text-sm font-bold text-red-400 mb-2">Danger Zone</h3>
            <p className="text-sm text-slate-400 mb-3">
              Permanently delete organization
            </p>
            <Button variant="danger" onClick={() => setShowDeleteModal(true)}>
              Delete Organization
            </Button>
          </Card>
        </TabPanel>
      </Tabs>

      <Modal
        open={showInviteModal}
        onClose={() => setShowInviteModal(false)}
        title="Invite Member"
      >
        <Input
          label="Email or Username"
          value={inviteEmail}
          onChange={(e) => setInviteEmail(e.target.value)}
        />
        <Select
          label="Role"
          options={[
            { value: "member", label: "Member" },
            { value: "billing_contact", label: "Billing Contact" },
          ]}
          value={inviteRole}
          onChange={(e) => setInviteRole(e.target.value)}
          className="mt-2"
        />
        <Button className="mt-4 w-full" onClick={() => inviteMember.mutate()}>
          Send Invitation
        </Button>
      </Modal>

      <Modal
        open={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Organization"
      >
        <p>
          Type <strong>{org.name}</strong> to confirm:
        </p>
        <Input
          value={deleteConfirm}
          onChange={(e) => setDeleteConfirm(e.target.value)}
          className="mt-2"
        />
        <div className="flex gap-2 mt-4">
          <Button variant="ghost" onClick={() => setShowDeleteModal(false)}>
            Cancel
          </Button>
          <Button
            variant="danger"
            disabled={deleteConfirm !== org.name}
            onClick={() => deleteOrg.mutate()}
          >
            Delete
          </Button>
        </div>
      </Modal>
    </div>
  );
}
