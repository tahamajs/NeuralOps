import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'

export default function TemplatesListPage() {
  const { data: templates = [], isLoading } = useQuery({
    queryKey: ['templates'],
    queryFn: async () => (await api.get('/api/experiments/templates/')).data,
  })

  if (isLoading) return <Spinner />
  if (!templates.length) return <EmptyState message="No templates saved" />

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-black text-white tracking-tighter">
          EXPERIMENT <span className="text-neon-blue">TEMPLATES</span>
        </h1>
        <Link to="/templates/new"><Button size="sm">+ New Template</Button></Link>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {templates.map((t: any) => (
          <Card key={t.id} hover>
            <h3 className="text-white font-semibold">{t.name}</h3>
            <p className="text-xs text-slate-400 mt-1">{t.category}</p>
            <div className="flex justify-end mt-4">
              <Link to={`/templates/${t.id}`}><Button variant="ghost" size="sm">Edit</Button></Link>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}