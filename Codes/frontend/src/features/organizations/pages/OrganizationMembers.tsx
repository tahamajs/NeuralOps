import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../api/client'
import { Card, Button, Input, Modal, Select, Spinner, Avatar } from '../../components/ui'
import { Trash2, Crown, Mail, UserPlus } from 'lucide-react'

export function OrganizationMembers({ orgId }: { orgId: string }) {
  const [showInviteModal, setShowInviteModal] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole, setInviteRole] = useState('member')
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['org-members', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/members/`)).data,
  })

  const inviteMember = useMutation({
    mutationFn: () => api.post(`/orgs/${orgId}/invite/`, { email: inviteEmail, role: inviteRole }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['org-members', orgId] })
      setShowInviteModal(false)
      setInviteEmail('')
      setInviteRole('member')
    },
  })

  const changeRole = useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: string }) =>
      api.post(`/orgs/${orgId}/members/${userId}/role/`, { role }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['org-members', orgId] }),
  })

  const removeMember = useMutation({
    mutationFn: (userId: string) => api.delete(`/orgs/${orgId}/members/${userId}/remove/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['org-members', orgId] }),
  })

  if (isLoading) return <Spinner />

  const members = data || []

  return (
    <div className="space-y-4">
      <div className="flex justify-between">
        <h2 className="text-lg font-bold text-white">Members ({members.length})</h2>
        <Button size="sm" onClick={() => setShowInviteModal(true)}>
          <UserPlus size={14} className="mr-1" /> Invite Member
        </Button>
      </div>

      <Card>
        <div className="space-y-3">
          {members.map((member: any) => (
            <div key={member.id} className="flex items-center justify-between py-2 border-b border-white/5">
              <div className="flex items-center gap-3">
                <Avatar src={member.avatar_url} size="sm" />
                <div>
                  <p className="text-white font-medium">{member.username}</p>
                  <p className="text-xs text-slate-400">{member.email}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={member.role}
                  onChange={(e) => changeRole.mutate({ userId: member.user_id, role: e.target.value })}
                  className="bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
                  disabled={member.role === 'owner'}
                >
                  <option value="owner">👑 Owner</option>
                  <option value="billing_contact">💳 Billing Contact</option>
                  <option value="member">👤 Member</option>
                </select>
                {member.role !== 'owner' && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => removeMember.mutate(member.user_id)}
                    className="text-red-400 hover:text-red-300"
                  >
                    <Trash2 size={14} />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Invite Modal */}
      <Modal open={showInviteModal} onClose={() => setShowInviteModal(false)} title="Invite Member">
        <div className="space-y-4">
          <Input
            label="Email or Username"
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            placeholder="user@example.com"
          />
          <Select
            label="Role"
            options={[
              { value: 'member', label: 'Member (read/write experiments)' },
              { value: 'billing_contact', label: 'Billing Contact (manage billing)' },
            ]}
            value={inviteRole}
            onChange={(e) => setInviteRole(e.target.value)}
          />
          <Button onClick={() => inviteMember.mutate()} disabled={!inviteEmail} className="w-full">
            Send Invitation
          </Button>
        </div>
      </Modal>
    </div>
  )
}