import { Card, Avatar, Button } from '../../../components/ui'
import { MapPin, Link2, Building, Calendar, Github, Twitter, Linkedin, Users } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import type { Portfolio, PortfolioStats } from '../types'

interface ProfileHeaderProps {
  portfolio: Portfolio
  stats?: PortfolioStats
  isOwnProfile: boolean
  onFollow?: () => void
  isFollowing?: boolean
  isFollowPending?: boolean
}

export function ProfileHeader({
  portfolio,
  stats,
  isOwnProfile,
  onFollow,
  isFollowing = false,
  isFollowPending = false,
}: ProfileHeaderProps) {
  const socialLinks = [
    { platform: 'github', url: portfolio.github_username ? `https://github.com/${portfolio.github_username}` : null, icon: Github },
    { platform: 'twitter', url: portfolio.twitter_username ? `https://twitter.com/${portfolio.twitter_username}` : null, icon: Twitter },
    { platform: 'linkedin', url: portfolio.linkedin_url, icon: Linkedin },
    { platform: 'website', url: portfolio.website, icon: Link2 },
  ]

  return (
    <div className="relative border-b border-white/10 bg-white/5 backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row gap-6">
          <Avatar src={portfolio.avatar_url} size="xl" className="shrink-0" />
          <div className="flex-1">
            <div className="flex flex-wrap justify-between items-start gap-4">
              <div>
                <h1 className="text-3xl font-bold text-white">{portfolio.full_name || portfolio.username}</h1>
                <p className="text-slate-400">@{portfolio.username}</p>
                {portfolio.bio && <p className="mt-2 text-slate-300 max-w-2xl">{portfolio.bio}</p>}
              </div>
              <div className="flex gap-3">
                {!isOwnProfile && onFollow && (
                  <Button onClick={onFollow} disabled={isFollowPending}>
                    {isFollowing ? 'Following' : 'Follow'} {!isFollowing && <Users size={14} className="ml-1" />}
                  </Button>
                )}
                {isOwnProfile && <Link to="/profile/settings"><Button variant="ghost">Edit Profile</Button></Link>}
              </div>
            </div>
            <div className="flex flex-wrap gap-4 mt-4 text-sm text-slate-400">
              {portfolio.location && <span className="flex items-center gap-1"><MapPin size={14} />{portfolio.location}</span>}
              {portfolio.company && <span className="flex items-center gap-1"><Building size={14} />{portfolio.company}</span>}
              <span className="flex items-center gap-1"><Calendar size={14} />Joined {new Date(portfolio.created_at).toLocaleDateString()}</span>
              <span className="flex items-center gap-1">
                <Users size={14} />
                <span className="font-medium text-white">{portfolio.followers_count}</span> followers
              </span>
            </div>
            <div className="flex flex-wrap gap-3 mt-3">
              {socialLinks.map((link, i) => link.url && (
                <a key={i} href={link.url} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
                  <link.icon size={16} />
                </a>
              ))}
            </div>
          </div>
        </div>
        <div className="flex gap-8 mt-6 pt-4 border-t border-white/10">
          <StatItem label="Reputation" value={stats?.reputation_score || 0} />
          <StatItem label="Rank" value={`#${stats?.rank || '-'}`} />
          <StatItem label="Contributions" value={stats?.total_contributions || 0} />
          <StatItem label="Current Streak" value={`${stats?.current_streak || 0} days`} />
          <StatItem label="Longest Streak" value={`${stats?.longest_streak || 0} days`} />
        </div>
      </div>
    </div>
  )
}

const StatItem = ({ label, value }: { label: string; value: string | number }) => (
  <div>
    <div className="text-xl font-bold text-white">{value}</div>
    <div className="text-xs text-slate-500 uppercase tracking-wider">{label}</div>
  </div>
)