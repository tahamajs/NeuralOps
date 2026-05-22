import { useQuery } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Spinner, Badge } from '../../../components/ui'

export default function BadgesPage() {
  const { data: allBadges } = useQuery({
    queryKey: ['badges'],
    queryFn: () => communityAPI.getBadges(),
  })

  const { data: myBadges } = useQuery({
    queryKey: ['my-badges'],
    queryFn: () => communityAPI.getMyBadges(),
  })

  if (!allBadges) return <Spinner />

  const earnedSet = new Set(myBadges?.map((b: any) => b.id) || [])

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Badges</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {allBadges.map((badge: any) => (
          <Card key={badge.id} className={earnedSet.has(badge.id) ? 'border-yellow-500/30 bg-yellow-500/5' : ''}>
            <img src={badge.icon_url} alt={badge.name} className="w-12 h-12 mx-auto mb-2" />
            <div className="text-center">
              <div className="font-bold text-white text-sm">{badge.name}</div>
              <div className="text-xs text-slate-400">{badge.description}</div>
              {earnedSet.has(badge.id) && <Badge variant="success" className="mt-2">Earned</Badge>}
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}