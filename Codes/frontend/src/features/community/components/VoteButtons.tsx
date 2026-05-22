import { useMutation, useQueryClient } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { ChevronUp, ChevronDown } from 'lucide-react'
import { cn } from '../../../../lib/utils'

interface VoteButtonsProps {
  targetType: 'question' | 'answer'
  targetId: string
  score: number
  userVote?: number  // -1, 0, or 1
  size?: 'sm' | 'md' | 'lg'
}

export function VoteButtons({ targetType, targetId, score, userVote = 0, size = 'md' }: VoteButtonsProps) {
  const queryClient = useQueryClient()
  const voteMutation = useMutation({
    mutationFn: async (value: number) => {
      if (targetType === 'question') {
        return communityAPI.voteQuestion(targetId, value)
      } else {
        return communityAPI.voteAnswer(targetId, value)
      }
    },
    onSuccess: () => {
      if (targetType === 'question') {
        queryClient.invalidateQueries({ queryKey: ['question', targetId] })
      } else {
        queryClient.invalidateQueries({ queryKey: ['answers'] })
      }
    },
  })

  const handleVote = (value: number) => {
    if (userVote === value) {
      voteMutation.mutate(0)  // Remove vote
    } else {
      voteMutation.mutate(value)
    }
  }

  const iconSize = size === 'sm' ? 16 : size === 'lg' ? 24 : 20

  return (
    <div className="flex flex-col items-center gap-1">
      <button
        onClick={() => handleVote(1)}
        className={cn(
          'p-1 rounded transition-colors',
          userVote === 1
            ? 'text-orange-400 bg-orange-500/10'
            : 'text-slate-500 hover:text-orange-400 hover:bg-white/5'
        )}
      >
        <ChevronUp size={iconSize} />
      </button>
      <span className={cn(
        'font-bold text-sm',
        score > 0 ? 'text-green-400' : score < 0 ? 'text-red-400' : 'text-white'
      )}>
        {score}
      </span>
      <button
        onClick={() => handleVote(-1)}
        className={cn(
          'p-1 rounded transition-colors',
          userVote === -1
            ? 'text-blue-400 bg-blue-500/10'
            : 'text-slate-500 hover:text-blue-400 hover:bg-white/5'
        )}
      >
        <ChevronDown size={iconSize} />
      </button>
    </div>
  )
}