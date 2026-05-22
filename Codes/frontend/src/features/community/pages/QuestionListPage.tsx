import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Badge, Spinner, Link } from '../../../components/ui'
import { Plus } from 'lucide-react'

export default function QuestionListPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['questions'],
    queryFn: async () => (await api.get('/questions/')).data,
  })
  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6">
      <div className="flex justify-between"><h1 className="text-2xl font-bold text-white">Community Q&A</h1><Link to="/questions/ask"><Button size="sm"><Plus size={14} /> Ask Question</Button></Link></div>
      {data?.results?.map((q: any) => (
        <Card key={q.id} hover>
          <h3 className="font-bold text-white">{q.title}</h3>
          <p className="text-sm text-slate-400 line-clamp-2">{q.body}</p>
          <div className="flex justify-between mt-2">
            <div className="flex gap-1">{q.tags?.map((t: string) => <Badge key={t}>{t}</Badge>)}</div>
            <div className="text-xs text-slate-500">{q.answer_count} answers · {q.vote_score} votes</div>
          </div>
          <Link to={`/questions/${q.id}`}><Button size="sm" variant="ghost" className="mt-2">View →</Button></Link>
        </Card>
      ))}
    </div>
  )
}