import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Input } from '../../../components/ui/Input'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'

export default function ModelSearchPage() {
  const [query, setQuery] = useState('')
  const [searchQuery, setSearchQuery] = useState('')

  const { data, isLoading, isFetched } = useQuery({
    queryKey: ['hfModels', searchQuery],
    queryFn: async () => (await api.get(`/api/config/search-models/?q=${searchQuery}`)).data,
    enabled: searchQuery.length > 0,
  })

  const handleSearch = () => setSearchQuery(query)

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        HUGGINGFACE <span className="text-neon-blue">MODELS</span>
      </h1>
      <Card>
        <div className="flex gap-3">
          <Input
            placeholder="Search models on HuggingFace Hub..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1"
          />
          <Button onClick={handleSearch}>Search</Button>
        </div>
        <div className="mt-4">
          {isLoading && <Spinner />}
          {isFetched && data?.results && (
            <div className="grid md:grid-cols-2 gap-3 max-h-[500px] overflow-y-auto scrollbar-thin">
              {data.results.map((model: string) => (
                <div key={model} className="bg-cyber-800 p-3 rounded text-sm text-white flex justify-between items-center">
                  <span className="truncate">{model}</span>
                  <Button size="sm" variant="ghost" onClick={() => {
                    // copy to clipboard or navigate to launcher with prefill
                    navigator.clipboard.writeText(model)
                    alert(`Copied "${model}" to clipboard`)
                  }}>Copy</Button>
                </div>
              ))}
            </div>
          )}
          {isFetched && !data?.results?.length && searchQuery && (
            <p className="text-slate-500 text-sm">No models found for "{searchQuery}".</p>
          )}
        </div>
      </Card>
    </div>
  )
}