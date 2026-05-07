import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { X, Search, Bookmark, ExternalLink } from 'lucide-react'

interface Props {
  onClose: () => void
  onPaperSaved?: () => void
}

export default function PaperSearchModal({ onClose, onPaperSaved }: Props) {
  const [query, setQuery] = useState('')
  const [source, setSource] = useState('all')
  const [results, setResults] = useState<any[]>([])

  const searchMut = useMutation({
    mutationFn: async () => {
      const res = await api.post('/api/agents/papers/search/', { query, source, limit: 20 })
      return res.data
    },
    onSuccess: (data) => setResults(data.results || []),
  })

  const saveMut = useMutation({
    mutationFn: async (paper: any) => {
      await api.post('/api/agents/papers/save/', {
        title: paper.title,
        authors: paper.authors,
        abstract: paper.abstract,
        url: paper.url,
        source: paper.source,
        source_id: paper.source_id,
        year: paper.year,
        citation_count: paper.citation_count,
        pdf_url: paper.pdf_url,
      })
    },
    onSuccess: () => onPaperSaved?.(),
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 animate-fade-in">
      <Card className="w-full max-w-3xl max-h-[80vh] overflow-y-auto p-6 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
            <Search size={20} className="text-neon-blue" /> Search Papers
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
        </div>

        <div className="flex gap-3">
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && searchMut.mutate()}
            placeholder="Search papers (e.g., 'attention compression transformers')"
            className="flex-1"
          />
          <Select
            options={[
              {value:'all',label:'All Sources'},
              {value:'semantic_scholar',label:'Semantic Scholar'},
              {value:'arxiv',label:'arXiv'},
              {value:'openalex',label:'OpenAlex'},
            ]}
            value={source}
            onChange={e => setSource(e.target.value)}
            className="w-40"
          />
          <Button onClick={() => searchMut.mutate()} disabled={searchMut.isPending}>
            {searchMut.isPending ? <Spinner className="py-0" /> : <Search size={16} />}
          </Button>
        </div>

        {results.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400">{results.length} papers found</p>
            {results.map((paper, i) => (
              <div key={i} className="glass border border-white/10 p-3 rounded-xl">
                <h3 className="text-sm font-semibold text-white">{paper.title}</h3>
                <p className="text-xs text-slate-400 mt-1">
                  {paper.authors?.slice(0, 4).join(', ')} ({paper.year})
                  {paper.journal_quality && ` · ${paper.journal_quality}`}
                </p>
                <p className="text-xs text-slate-500 mt-1 line-clamp-2">{paper.abstract}</p>
                <div className="flex gap-2 mt-2">
                  <Button size="sm" variant="ghost" onClick={() => window.open(paper.url, '_blank')}>
                    <ExternalLink size={12} className="mr-1" /> View
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => saveMut.mutate(paper)}>
                    <Bookmark size={12} className="mr-1" /> Save
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}