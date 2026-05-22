import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Table, Badge } from '../../../components/ui'

export default function ValidatorDashboardPage() {
  const { data: myValidator, isLoading } = useQuery({
    queryKey: ['my-validator'],
    queryFn: async () => (await api.get('/validators/me/')).data,
  })
  const { data: tasks } = useQuery({
    queryKey: ['verification-tasks'],
    queryFn: async () => (await api.get('/tasks/')).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="text-sm font-bold text-white">My Validator Node</h2>
        <div>Status: <Badge variant={myValidator?.status}>{myValidator?.status}</Badge></div>
        <div>Stake: {myValidator?.staked_amount} tokens</div>
        <div>Verified tasks: {myValidator?.verified_count}</div>
      </Card>
      <Card>
        <h2 className="text-sm font-bold text-white mb-2">Pending Verification Tasks</h2>
        <Table>
          <thead><tr><th>Job ID</th><th>Provider</th><th>Deadline</th><th>Action</th></tr></thead>
          <tbody>
            {tasks?.map((task: any) => (
              <tr key={task.id}>
                <td>{task.job_id.slice(0,8)}</td>
                <td>{task.provider_username}</td>
                <td>{new Date(task.deadline).toLocaleString()}</td>
                <td><button>Verify</button></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  )
}