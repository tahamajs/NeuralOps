import { useParams, Outlet, NavLink } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Tabs, TabList, Tab, TabPanel } from '../../../components/ui'
import { Users, CreditCard, Settings, ClipboardList, FolderTree, Activity } from 'lucide-react'

export default function OrganizationDetailPage() {
  const { orgId } = useParams()
  const { data: org, isLoading } = useQuery({
    queryKey: ['organization', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/`)).data,
    enabled: !!orgId,
  })

  if (isLoading) return <Spinner />

  const tabs = [
    { value: 'dashboard', label: 'Dashboard', icon: <Activity size={14} /> },
    { value: 'members', label: 'Members', icon: <Users size={14} /> },
    { value: 'teams', label: 'Teams', icon: <FolderTree size={14} /> },
    { value: 'billing', label: 'Billing', icon: <CreditCard size={14} /> },
    { value: 'settings', label: 'Settings', icon: <Settings size={14} /> },
    { value: 'audit', label: 'Audit Log', icon: <ClipboardList size={14} /> },
  ]

  return (
    <div className="space-y-6">
      {/* Organization header */}
      <div className="flex items-center gap-4">
        {org.avatar_url ? (
          <img src={org.avatar_url} alt={org.name} className="w-16 h-16 rounded-xl object-cover" />
        ) : (
          <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center">
            <span className="text-white font-bold text-2xl">{org.name.charAt(0).toUpperCase()}</span>
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold text-white">{org.name}</h1>
          <p className="text-sm text-slate-400">{org.description || 'No description'}</p>
          <div className="flex gap-2 mt-1">
            {org.is_owner && <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">Owner</span>}
            {org.is_billing_contact && <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded">Billing Contact</span>}
          </div>
        </div>
      </div>

      {/* Navigation tabs */}
      <Tabs defaultValue="dashboard">
        <TabList>
          {tabs.map((tab) => (
            <Tab key={tab.value} value={tab.value}>
              <span className="flex items-center gap-2">{tab.icon} {tab.label}</span>
            </Tab>
          ))}
        </TabList>
        <TabPanel value="dashboard"><OrganizationDashboard orgId={orgId!} /></TabPanel>
        <TabPanel value="members"><OrganizationMembers orgId={orgId!} /></TabPanel>
        <TabPanel value="teams"><OrganizationTeams orgId={orgId!} /></TabPanel>
        <TabPanel value="billing"><OrganizationBilling orgId={orgId!} /></TabPanel>
        <TabPanel value="settings"><OrganizationSettings orgId={orgId!} /></TabPanel>
        <TabPanel value="audit"><OrganizationAuditLog orgId={orgId!} /></TabPanel>
      </Tabs>
    </div>
  )
}