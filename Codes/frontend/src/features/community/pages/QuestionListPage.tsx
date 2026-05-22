import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Button, Input, Spinner, Badge, PaginationControls } from '../../../components/ui'
import { Search, Plus, Clock, TrendingUp, Users } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

type SortOption = 'newest' | 'votes' | 'unanswered' | 'trending'

export default function QuestionListPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [search, setSearch] = useState(searchParams.get('q') || '')
  const [sortBy, setSortBy] = useState<SortOption>((searchParams.get('sort') as SortOption) || 'newest')
  const [currentPage, setCurrentPage] = useState(parseInt(searchParams.get('page') || '1'))
  const [selectedTag, setSelectedTag] = useState(searchParams.get('tag') || '')

  const { data: questionsData, isLoading } = useQuery({
    queryKey: ['questions', { sort: sortBy, tag: selectedTag, page: currentPage, search }],
    queryFn: () =>
      communityAPI.getQuestions({
        sort: sortBy,
        tag: selectedTag || undefined,
        page: currentPage,
        q: search || undefined,
      }),
  })

  const { data: tagsData } = useQuery({
    queryKey: ['tags'],
    queryFn: () => communityAPI.getTags(),
  })

  const questions = questionsData?.results || []
  const totalPages = Math.ceil(questionsData?.count / 20) || 0

  const sortOptions: { value: SortOption; label: string; icon: React.ReactNode }[] = [
    { value: 'newest', label: 'Newest', icon: <Clock size={14} /> },
    { value: 'votes', label: 'Most Votes', icon: <TrendingUp size={14} /> },
    { value: 'unanswered', label: 'Unanswered', icon: <Users size={14} /> },
  ]

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setSearchParams({ q: search, sort: sortBy, tag: selectedTag, page: '1' })
    setCurrentPage(1)
  }

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4">
        <h1 className="text-2xl font-bold text-white">Community Q&A</h1>
        <Link to="/questions/ask">
          <Button><Plus size={16} className="mr-1" /> Ask Question</Button>
        </Link>
      </div>

      {/* Search & Filters */}
      <div className="grid gap-4 lg:grid-cols-4">
        <div className="lg:col-span-3">
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search questions..."
                className="pl-9"
              />
            </div>
            <Button type="submit" size="sm">Search</Button>
          </form>
        </div>

        <select
          value={sortBy}
          onChange={(e) => {
            setSortBy(e.target.value as SortOption)
            setSearchParams({ ...searchParams, sort: e.target.value, page: '1' })
            setCurrentPage(1)
          }}
          className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white"
        >
          {sortOptions.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {/* Tags filter */}
      {tagsData?.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => {
              setSelectedTag('')
              setSearchParams({ sort: sortBy, page: '1' })
            }}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${!selectedTag ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
          >
            All
          </button>
          {tagsData.map((tag: any) => (
            <button
              key={tag.name}
              onClick={() => {
                setSelectedTag(tag.name)
                setSearchParams({ tag: tag.name, sort: sortBy, page: '1' })
                setCurrentPage(1)
              }}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${selectedTag === tag.name ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
            >
              {tag.name} <span className="text-slate-500">{tag.count}</span>
            </button>
          ))}
        </div>
      )}

      {/* Question list */}
      {questions.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-slate-400">No questions found. Be the first to ask!</p>
          <Link to="/questions/ask"><Button variant="secondary" className="mt-4">Ask Question</Button></Link>
        </Card>
      ) : (
        <div className="space-y-3">
          {questions.map((question: any) => (
            <Link key={question.id} to={`/questions/${question.id}`}>
              <Card hover className="group">
                <div className="flex flex-wrap gap-6">
                  {/* Stats */}
                  <div className="flex gap-4 text-center text-sm">
                    <div><div className="font-bold text-white">{question.vote_score}</div><div className="text-[10px] text-slate-500">votes</div></div>
                    <div><div className={`font-bold ${question.answer_count > 0 ? 'text-green-400' : 'text-slate-400'}`}>{question.answer_count}</div><div className="text-[10px] text-slate-500">answers</div></div>
                    <div><div className="font-bold text-slate-400">{question.view_count}</div><div className="text-[10px] text-slate-500">views</div></div>
                  </div>
                  {/* Content */}
                  <div className="flex-1">
                    <h3 className="font-semibold text-white group-hover:text-neon-blue transition-colors">
                      {question.title}
                    </h3>
                    <div className="text-sm text-slate-400 line-clamp-2 mt-1" dangerouslySetInnerHTML={{ __html: question.body?.slice(0, 200) }} />
                    <div className="flex flex-wrap gap-1 mt-2">
                      {question.tags?.map((tag: string) => (
                        <span key={tag} className="px-2 py-0.5 rounded bg-neon-blue/10 text-neon-blue text-[10px] font-medium">#{tag}</span>
                      ))}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-slate-500">
                      <span>asked {formatDistanceToNow(new Date(question.created_at), { addSuffix: true })}</span>
                      <span>by {question.author?.username}</span>
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex justify-center mt-4">
          <PaginationControls currentPage={currentPage} totalPages={totalPages} onPageChange={(page) => {
            setCurrentPage(page)
            setSearchParams({ ...searchParams, page: page.toString() })
          }} />
        </div>
      )}
    </div>
  )
}