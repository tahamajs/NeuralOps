import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { Card, Button, Input, Modal, Spinner, EmptyState } from '../../../components/ui'
import { Plus, Search, Users, CreditCard, Settings } from 'lucide-react'

export default function OrganizationListPage() {
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newOrgName, setNewOrgName] = useState('')
  const [newOrgSlug, setNewOrgSlug] = useState('')
  const [newOrgDesc, setNewOrgDesc] = useState('')
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['organizations', search],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (search) params.set('search', search)
      const res = await api.get(`/orgs/?${params.toString()}`)
      return res.data
    },
  })

  const createOrg = useMutation({
    mutationFn: async () => {
      const res = await api.post('/orgs/', {
        name: newOrgName,
        slug: newOrgSlug,
        description: newOrgDesc,
      })
      return res.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['organizations'] })
      setShowCreateModal(false)
      setNewOrgName('')
      setNewOrgSlug('')
      setNewOrgDesc('')
    },
  })

  if (isLoading) return <Spinner />

  const organizations = data?.results || []

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap justify-between items-center gap-4">
        <h1 className="text-2xl font-bold text-white">Organizations</h1>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus size={16} className="mr-1" /> Create Organization
        </Button>
      </div>

      <div className="relative">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <Input
          placeholder="Search organizations..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {organizations.length === 0 ? (
        <EmptyState
          title="No organizations yet"
          description="Create your first organization to get started"
          action={
            <Button onClick={() => setShowCreateModal(true)}>
              <Plus size={16} className="mr-1" /> Create Organization
            </Button>
          }
        />
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {organizations.map((org: any) => (
            <Link key={org.id} to={`/orgs/${org.id}`}>
              <Card hover className="group">
                <div className="flex items-start gap-3">
                  {org.avatar_url ? (
                    <img
                      src={org.avatar_url}
                      alt={org.name}
                      className="w-12 h-12 rounded-xl object-cover"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center">
                      <span className="text-white font-bold text-lg">
                        {org.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                  )}
                  <div className="flex-1">
                    <h3 className="font-bold text-white group-hover:text-neon-blue transition-colors">
                      {org.name}
                    </h3>
                    <p className="text-sm text-slate-400 mt-0.5">
                      {org.description || 'No description'}
                    </p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <Users size={12} /> {org.member_count} members
                      </span>
                      {org.is_owner && (
                        <span className="flex items-center gap-1 text-yellow-400">
                          👑 Owner
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}

      {/* Create Organization Modal */}
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Organization">
        <div className="space-y-4">
          <Input
            label="Organization Name"
            value={newOrgName}
            onChange={(e) => {
              setNewOrgName(e.target.value)
              setNewOrgSlug(e.target.value.toLowerCase().replace(/\s+/g, '-'))
            }}
            placeholder="Acme Inc."
          />
          <Input
            label="Slug (URL identifier)"
            value={newOrgSlug}
            onChange={(e) => setNewOrgSlug(e.target.value)}
            placeholder="acme-inc"
          />
          <Input
            label="Description (optional)"
            value={newOrgDesc}
            onChange={(e) => setNewOrgDesc(e.target.value)}
            placeholder="Our awesome organization"
          />
          <Button
            onClick={() => createOrg.mutate()}
            disabled={!newOrgName || !newOrgSlug || createOrg.isPending}
            className="w-full"
          >
            {createOrg.isPending ? 'Creating...' : 'Create Organization'}
          </Button>
        </div>
      </Modal>
    </div>
  )
}