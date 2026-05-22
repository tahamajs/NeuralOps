import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { gamificationAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Trophy, Lock, Award, TrendingUp } from 'lucide-react'
import type { Badge as BadgeType, BadgeCategory } from '../types'

const categoryIcons: Record<BadgeCategory, string> = {
  experiments: '🔬',
  notes: '📝',
  collaboration: '🤝',
  streak: '🔥',
  community: '💬',
}

const rarityColors: Record<BadgeRarity, string> = {
  common: 'text-slate-400 border-slate-500',
  rare: 'text-blue-400 border-blue-500',
  epic: 'text-purple-400 border-purple-500',
  legendary: 'text-yellow-400 border-yellow-500',
}

export default function BadgeWallPage({ onClose }: { onClose?: () => void }) {
  const [category, setCategory] = useState<BadgeCategory | 'all'>('all')
  const [showLocked, setShowLocked] = useState(false)

  const { data: allBadges, isLoading: badgesLoading } = useQuery({
    queryKey: ['badges-all'],
    queryFn: () => gamificationAPI.getBadges(),
  })

  const { data: myBadges, isLoading: myBadgesLoading } = useQuery({
    queryKey: ['badges-mine'],
    queryFn: () => gamificationAPI.getMyBadges(),
  })

  const { data: stats } = useQuery({
    queryKey: ['badge-stats'],
    queryFn: () => gamificationAPI.getBadgeStats(),
  })

  if (badgesLoading || myBadgesLoading) return <Spinner />

  const earnedIds = new Set(myBadges?.map(b => b.id) || [])
  const filteredBadges = allBadges?.filter(b =>
    (category === 'all' || b.category === category) &&
    (showLocked || earnedIds.has(b.id))
  ) || []

  const categoryOptions = [
    { value: 'all', label: 'All Categories' },
    { value: 'experiments', label: '🔬 Experiments' },
    { value: 'notes', label: '📝 Notes' },
    { value: 'collaboration', label: '🤝 Collaboration' },
    { value: 'streak', label: '🔥 Streaks' },
    { value: 'community', label: '💬 Community' },
  ]

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Award size={20} className="text-yellow-400" /> Badge Wall</h1><p className="text-xs text-slate-400">Earn achievements by contributing to the community</p></div>
        <div className="flex gap-2"><Button size="sm" variant="ghost" onClick={() => setShowLocked(!showLocked)}>{showLocked ? 'Hide locked' : 'Show locked'}</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        {/* Stats summary */}
        {stats && (
          <TerminalCard className="p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div><div className="text-2xl font-bold text-white">{stats.earned}/{stats.total}</div><div className="text-xs text-slate-400">Badges earned</div></div>
              <div><div className="text-2xl font-bold text-purple-400">{stats.by_rarity.legendary}</div><div className="text-xs text-slate-400">Legendary</div></div>
              <div><div className="text-2xl font-bold text-blue-400">{stats.by_rarity.epic}</div><div className="text-xs text-slate-400">Epic</div></div>
              <div><div className="text-2xl font-bold text-green-400">{stats.by_rarity.rare}</div><div className="text-xs text-slate-400">Rare</div></div>
            </div>
          </TerminalCard>
        )}

        {/* Filters */}
        <div className="flex gap-3"><Select options={categoryOptions} value={category} onChange={e => setCategory(e.target.value as BadgeCategory | 'all')} className="w-48" /></div>

        {/* Badge grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredBadges.map(badge => {
            const earned = earnedIds.has(badge.id)
            return (
              <TerminalCard key={badge.id} className={`text-center p-4 transition-all ${earned ? 'opacity-100' : 'opacity-60'}`}>
                <div className="text-4xl mb-2">{badge.icon}</div>
                <div className="font-bold text-white text-sm">{badge.name}</div>
                <div className="text-xs text-slate-400 mt-0.5">{badge.description}</div>
                <div className="flex justify-center gap-1 mt-2"><Badge variant="secondary" className="text-[10px]">{badge.category}</Badge><Badge variant={earned ? 'success' : 'secondary'} className="text-[10px]">{earned ? 'Earned' : 'Locked'}</Badge></div>
                {!earned && badge.progress_current !== undefined && badge.progress_target !== undefined && (<div className="mt-2"><div className="text-[10px] text-slate-400">{badge.progress_current}/{badge.progress_target}</div><div className="w-full bg-white/10 rounded-full h-1"><div className="bg-neon-blue h-1 rounded-full" style={{ width: `${(badge.progress_current / badge.progress_target) * 100}%` }} /></div></div>)}
              </TerminalCard>
            )
          })}
        </div>
      </div>
    </div>
  )
}