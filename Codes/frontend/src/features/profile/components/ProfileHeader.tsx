import { useState } from 'react'
import { useAuthStore } from '../../auth/store/authStore'
import { Card, Button, Avatar } from '../../../components/ui'
import { MapPin, Link2, Calendar, Users, Award, Edit2, Camera, Building, Github, Twitter, Linkedin } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

interface ProfileHeaderProps {
  profile: UserProfile
  stats: UserStats
  isOwnProfile: boolean
  onEdit: () => void
  onAvatarUpload: (file: File) => void
  onCoverUpload?: (file: File) => void
  coverImageUrl?: string
}

export function ProfileHeader({
  profile,
  stats,
  isOwnProfile,
  onEdit,
  onAvatarUpload,
  onCoverUpload,
  coverImageUrl,
}: ProfileHeaderProps) {
  const [avatarHover, setAvatarHover] = useState(false)
  const [coverHover, setCoverHover] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const coverInputRef = useRef<HTMLInputElement>(null)

  const handleAvatarClick = () => fileInputRef.current?.click()
  const handleCoverClick = () => coverInputRef.current?.click()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'avatar' | 'cover') => {
    const file = e.target.files?.[0]
    if (file) {
      type === 'avatar' ? onAvatarUpload(file) : onCoverUpload?.(file)
    }
  }

  return (
    <div className="relative">
      {/* Cover Image */}
      <div
        className="relative h-48 rounded-t-2xl overflow-hidden bg-gradient-to-r from-neon-blue/20 to-purple-500/20"
        onMouseEnter={() => setCoverHover(true)}
        onMouseLeave={() => setCoverHover(false)}
      >
        {coverImageUrl && (
          <img src={coverImageUrl} alt="Cover" className="w-full h-full object-cover" />
        )}
        {isOwnProfile && onCoverUpload && coverHover && (
          <button
            onClick={handleCoverClick}
            className="absolute inset-0 flex items-center justify-center bg-black/50 transition-colors"
          >
            <Camera size={24} className="text-white" />
          </button>
        )}
        <input
          ref={coverInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFileChange(e, 'cover')}
        />
      </div>

      {/* Avatar */}
      <div className="absolute -bottom-12 left-6">
        <div
          className="relative"
          onMouseEnter={() => setAvatarHover(true)}
          onMouseLeave={() => setAvatarHover(false)}
        >
          <Avatar src={profile.avatar_url} size="xl" className="border-4 border-surface" />
          {isOwnProfile && avatarHover && (
            <button
              onClick={handleAvatarClick}
              className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full"
            >
              <Camera size={20} className="text-white" />
            </button>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFileChange(e, 'avatar')}
        />
      </div>

      {/* Edit Button */}
      {isOwnProfile && (
        <Button
          onClick={onEdit}
          variant="secondary"
          size="sm"
          className="absolute top-4 right-4"
        >
          <Edit2 size={14} className="mr-1" /> Edit Profile
        </Button>
      )}

      {/* Profile Info */}
      <div className="pt-16 px-6 pb-4">
        <div className="flex flex-wrap justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-white">
              {profile.full_name || profile.username}
            </h1>
            <p className="text-slate-400">@{profile.username}</p>
            {profile.pronouns && (
              <span className="text-xs text-slate-500">{profile.pronouns}</span>
            )}
            {profile.bio && (
              <p className="mt-3 text-slate-300 max-w-2xl">{profile.bio}</p>
            )}
          </div>

          {/* Reputation Score */}
          <div className="flex items-center gap-2 glass px-3 py-1.5 rounded-full">
            <Award size={14} className="text-yellow-400" />
            <span className="text-sm font-bold text-white">{stats.reputation_score}</span>
            <span className="text-xs text-slate-400">reputation</span>
          </div>
        </div>

        {/* Meta Information */}
        <div className="flex flex-wrap gap-4 mt-4 text-sm text-slate-400">
          {profile.location && (
            <span className="flex items-center gap-1"><MapPin size={14} /> {profile.location}</span>
          )}
          {profile.company && (
            <span className="flex items-center gap-1"><Building size={14} /> {profile.company}</span>
          )}
          {profile.affiliation && (
            <span className="flex items-center gap-1"><Users size={14} /> {profile.affiliation}</span>
          )}
          <span className="flex items-center gap-1">
            <Calendar size={14} /> Joined {new Date(profile.created_at).toLocaleDateString()}
          </span>
        </div>

        {/* Social Links */}
        {(profile.website || profile.github_username || profile.twitter_username || profile.linkedin_url) && (
          <div className="flex flex-wrap gap-3 mt-3">
            {profile.website && (
              <a href={profile.website} target="_blank" rel="noopener noreferrer" className="text-neon-blue hover:underline text-sm">
                <Link2 size={14} className="inline mr-1" /> {profile.website}
              </a>
            )}
            {profile.github_username && (
              <a href={`https://github.com/${profile.github_username}`} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white">
                <Github size={16} className="inline mr-1" /> @{profile.github_username}
              </a>
            )}
            {profile.twitter_username && (
              <a href={`https://twitter.com/${profile.twitter_username}`} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white">
                <Twitter size={16} className="inline mr-1" /> @{profile.twitter_username}
              </a>
            )}
            {profile.linkedin_url && (
              <a href={profile.linkedin_url} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white">
                <Linkedin size={16} className="inline mr-1" /> LinkedIn
              </a>
            )}
          </div>
        )}

        {/* Expertise & Research Interests */}
        {(profile.expertise?.length > 0 || profile.research_interests?.length > 0) && (
          <div className="mt-4 space-y-2">
            {profile.expertise?.length > 0 && (
              <div>
                <span className="text-xs font-semibold text-slate-400 uppercase">Expertise</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {profile.expertise.map(skill => (
                    <span key={skill} className="px-2 py-0.5 rounded bg-neon-blue/10 text-neon-blue text-xs">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {profile.research_interests?.length > 0 && (
              <div>
                <span className="text-xs font-semibold text-slate-400 uppercase">Research Interests</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {profile.research_interests.map(interest => (
                    <span key={interest} className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 text-xs">
                      {interest}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
          <StatCard label="Experiments" value={stats.total_experiments} />
          <StatCard label="Total Runs" value={stats.total_runs} />
          <StatCard label="GPU Hours" value={stats.total_gpu_hours} />
          <StatCard label="Models Published" value={stats.total_models_published} />
          <StatCard label="Papers" value={stats.total_papers} />
          <StatCard label="Notes" value={stats.total_notes} />
          <StatCard label="Contributions" value={stats.contribution_days} />
          <StatCard label="Current Streak" value={stats.current_streak} />
        </div>
      </div>
    </div>
  )
}

const StatCard = ({ label, value }: { label: string; value: number }) => (
  <Card className="text-center p-3">
    <div className="text-xl font-bold text-white">{value.toLocaleString()}</div>
    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
  </Card>
)