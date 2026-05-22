import { useState } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Avatar } from '../../../components/ui/Avatar'
import { Button } from '../../../components/ui/Button'
import { Mail, ExternalLink, Linkedin, BookOpen, Award, MapPin, Star, StarOff } from 'lucide-react'
import { ExpertProfile } from '../types'

interface ExpertCardProps {
  expert: ExpertProfile
  variant?: 'card' | 'list'
}

export function ExpertCard({ expert, variant = 'card' }: ExpertCardProps) {
  const [isStarred, setIsStarred] = useState(false)

  const toggleStar = () => setIsStarred(!isStarred)

  if (variant === 'list') {
    return (
      <TerminalCard className="p-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3"><Avatar src={expert.avatar_url} size="md" /><div><div className="font-medium text-white">{expert.name}</div><div className="text-xs text-slate-400">{expert.title} · {expert.institution}</div></div></div>
        <div className="flex gap-2"><Badge variant={expert.availability === 'available' ? 'success' : expert.availability === 'looking' ? 'info' : 'secondary'}>{expert.availability}</Badge><Button size="sm" variant="ghost"><Mail size={14} /></Button></div>
      </TerminalCard>
    )
  }

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start"><div className="flex items-center gap-3"><Avatar src={expert.avatar_url} size="lg" /><div><h3 className="font-bold text-white">{expert.name}</h3><div className="text-sm text-slate-400">{expert.title}</div><div className="text-xs text-slate-500">{expert.institution}{expert.department ? ` · ${expert.department}` : ''}</div></div></div><button onClick={toggleStar} className="text-slate-400 hover:text-yellow-400">{isStarred ? <Star size={18} className="fill-yellow-400 text-yellow-400" /> : <StarOff size={18} />}</button></div>
      {expert.location && <div className="flex items-center gap-1 text-xs text-slate-500 mt-2"><MapPin size={12} /> {expert.location}</div>}
      <div className="flex flex-wrap gap-1 mt-3">{expert.expertise.slice(0, 5).map(tag => <Badge key={tag}>{tag}</Badge>)}</div>
      <div className="flex gap-4 mt-3 text-sm"><div className="flex items-center gap-1"><Award size={14} className="text-slate-400" /> h‑index: {expert.h_index}</div><div><span className="text-slate-400">📄</span> {expert.publications_count}</div><div><span className="text-slate-400">📊</span> {expert.citations_count.toLocaleString()}</div></div>
      <div className="flex gap-2 mt-3"><Badge variant={expert.availability === 'available' ? 'success' : expert.availability === 'looking' ? 'info' : 'secondary'}>{expert.availability}</Badge><Button size="sm" variant="ghost"><Mail size={14} className="mr-1" /> Contact</Button><Button size="sm" variant="ghost"><ExternalLink size={14} className="mr-1" /> Profile</Button></div>
    </TerminalCard>
  )
}