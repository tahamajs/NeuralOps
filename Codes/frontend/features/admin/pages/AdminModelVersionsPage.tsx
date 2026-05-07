import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'

export default function AdminModelVersionsPage() {
  const { data: experiments = [], isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get('/api/experiments/')).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white">Model Versions</h1>
      <p className="text-slate-400 text-sm">Select an experiment to manage its model versions.</p>
      <div className="grid gap-4">
        {experiments.map((exp: any) => (
          <Card key={exp.id} hover className="flex justify-between items-center">
            <div>
              <span className="text-white font-semibold">{exp.name || exp.id.slice(0,8)}</span>
              <p className="text-xs text-slate-500">{exp.job_type} – {exp.status}</p>
            </div>
            <Button variant="ghost" size="sm" onClick={() => window.open(`/runs/${exp.id}/model-versions`, '_blank')}>
              Manage Versions
            </Button>
          </Card>
        ))}
      </div>
    </div>
  )
}