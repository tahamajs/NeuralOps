import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../api/client'
import { Card, Button, Input, Modal, Spinner } from '../../components/ui'
import { Plus, Users, Trash2 } from 'lucide-react'

export function OrganizationTeams({ orgId }: { orgId: string }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [teamName, setTeamName] = useState('')
  const [teamDesc, setTeamDesc] = useState('')
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['org-teams', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/teams/`)).data,
  })

  const createTeam = useMutation({
    mutationFn: () => api.post(`/orgs/${orgId}/teams/`, { name: teamName, description: teamDesc }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['org-teams', orgId] })
      setShowCreateModal(false)
      setTeamName('')
      setTeamDesc('')
    },
  })

  const deleteTeam = useMutation({
    mutationFn: (teamId: string) => api.delete(`/orgs/${orgId}/teams/${teamId}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['org-teams', orgId] }),
  })

  if (isLoading) return <Spinner />

  const teams = data || []

  return (
    <div className="space-y-4">
      <div className="flex justify-between">
        <h2 className="text-lg font-bold text-white">Teams</h2>
        <Button size="sm" onClick={() => setShowCreateModal(true)}>
          <Plus size={14} className="mr-1" /> Create Team
        </Button>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {teams.map((team: any) => (
          <Card key={team.id}>
            <div className="flex justify-between">
              <div>
                <h3 className="font-bold text-white">{team.name}</h3>
                <p className="text-sm text-slate-400">{team.description || 'No description'}</p>
                <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
                  <Users size={12} /> {team.member_count} members
                </div>
              </div>
              <Button size="sm" variant="ghost" onClick={() => deleteTeam.mutate(team.id)}>
                <Trash2 size={14} className="text-red-400" />
              </Button>
            </div>
          </Card>
        ))}
      </div>

      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Team">
        <div className="space-y-4">
          <Input label="Team Name" value={teamName} onChange={(e) => setTeamName(e.target.value)} />
          <Input label="Description" value={teamDesc} onChange={(e) => setTeamDesc(e.target.value)} />
          <Button onClick={() => createTeam.mutate()} disabled={!teamName}>Create Team</Button>
        </div>
      </Modal>
    </div>
  )
}