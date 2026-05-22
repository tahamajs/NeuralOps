import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Spinner } from '../../../components/ui'
import { CheckCircle, XCircle } from 'lucide-react'

export default function InviteAcceptPage() {
  const { token } = useParams()
  const navigate = useNavigate()
  const { data, isLoading, error } = useQuery({
    queryKey: ['invitation', token],
    queryFn: async () => (await api.get(`/invitations/${token}/`)).data,
    enabled: !!token,
  })

  const acceptInvite = useMutation({
    mutationFn: () => api.post(`/invitations/${token}/accept/`, {}),
    onSuccess: () => navigate('/organizations'),
  })

  if (isLoading) return <Spinner />
  if (error) return <div className="text-center text-red-400">Invalid or expired invitation</div>

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface">
      <Card className="max-w-md text-center">
        {data?.status === 'pending' ? (
          <>
            <CheckCircle size={48} className="text-green-400 mx-auto mb-4" />
            <h1 className="text-xl font-bold text-white">You've been invited!</h1>
            <p className="text-slate-400 mt-2">
              {data.invited_by} has invited you to join <strong>{data.organization_name}</strong> as a <strong>{data.role}</strong>.
            </p>
            <Button className="mt-4 w-full" onClick={() => acceptInvite.mutate()}>Accept Invitation</Button>
          </>
        ) : (
          <>
            <XCircle size={48} className="text-red-400 mx-auto mb-4" />
            <h1 className="text-xl font-bold text-white">Invitation Expired</h1>
            <p className="text-slate-400 mt-2">This invitation is no longer valid.</p>
            <Button variant="ghost" className="mt-4" onClick={() => navigate('/organizations')}>Back to Organizations</Button>
          </>
        )}
      </Card>
    </div>
  )
}