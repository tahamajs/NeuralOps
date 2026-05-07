import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Input } from '../../../components/ui/Input'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Search, Copy, ExternalLink, FlaskConical } from 'lucide-react'

export default function ModelSearchPage() {
  const [query, setQuery] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const { data, isLoading, isFetched } = useQuery({
    queryKey: ['hfModels', searchQuery],
    queryFn: async () =>
      (await api.get(`/api/config/search-models/?q=${searchQuery}`)).data,
    enabled: searchQuery.length > 0,
  })

  const handleSearch = () => setSearchQuery(query.trim())

  const handleCopy = (modelId: string) => {
    navigator.clipboard.writeText(modelId)
    setCopiedId(modelId)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center gap-3">
        <Search size={24} className="text-neon-blue" />
        <h1 className="text-2xl font-display font-bold text-white">
          HuggingFace <span className="text-neon-blue">Model Search</span>
        </h1>
      </div>

      <Card>
        <p className="text-sm text-slate-400 mb-4">
          Discover models on the HuggingFace Hub. Click the copy icon to grab a model ID for your experiment launcher.
        </p>

        <div className="flex gap-3">
          <Input
            placeholder="Search models (e.g., bert-base-uncased, Llama-2-7b, mistral)..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            icon={<Search size={14} />}
            className="flex-1"
          />
          <Button onClick={handleSearch} loading={isLoading} disabled={!query.trim()}>
            Search
          </Button>
        </div>

        <div className="mt-6">
          {isLoading && <Spinner className="py-8" />}

          {isFetched && !data?.results?.length && searchQuery && (
            <p className="text-slate-500 text-sm text-center py-8">
              No models found for "{searchQuery}".
            </p>
          )}

          {isFetched && data?.results?.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-slate-400 mb-3">
                {data.results.length} model{data.results.length !== 1 ? 's' : ''} found
              </p>
              <div className="grid md:grid-cols-2 gap-3 max-h-[600px] overflow-y-auto scrollbar-thin">
                {data.results.map((modelId: string) => (
                  <div
                    key={modelId}
                    className="bg-cyber-800 p-4 rounded-xl border border-white/5 flex items-center justify-between group hover:border-white/10 transition-all"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">{modelId}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">HF Hub</p>
                    </div>
                    <div className="flex gap-1 ml-3">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleCopy(modelId)}
                        title={copiedId === modelId ? 'Copied!' : 'Copy model ID'}
                      >
                        {copiedId === modelId ? (
                          <span className="text-green-400 text-xs">Copied!</span>
                        ) : (
                          <Copy size={14} />
                        )}
                      </Button>
                      <a
                        href={`https://huggingface.co/${modelId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Button size="sm" variant="ghost" title="View on HuggingFace">
                          <ExternalLink size={14} />
                        </Button>
                      </a>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          // Navigate to launcher with model pre‑filled
                          window.location.href = `/launcher?model=${encodeURIComponent(modelId)}`
                        }}
                        title="Use in experiment"
                      >
                        <FlaskConical size={14} />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}