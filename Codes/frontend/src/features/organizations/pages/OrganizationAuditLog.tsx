import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'
import { Card, Spinner } from '../../components/ui'

export function OrganizationAuditLog({ orgId }: { orgId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['org-audit', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/audit-log/`)).data,
  })

  if (isLoading) return <Spinner />

  return (
    <Card>
      <h3 className="text-sm font-bold text-white mb-3">Audit Log</h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {data?.map((entry: any) => (
          <div key={entry.id} className="flex justify-between text-sm border-b border-white/5 py-2">
            <div>
              <span className="text-neon-blue">{entry.user}</span>
              <span className="text-slate-300"> {entry.action}</span>
            </div>
            <span className="text-xs text-slate-500">{new Date(entry.created_at).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}