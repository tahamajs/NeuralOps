export type VoteType = 'up' | 'down' | null
export type QuestionStatus = 'open' | 'closed' | 'on_hold'

export interface User {
  id: string
  username: string
  avatar_url?: string
  reputation: number
}

export interface Question {
  id: string
  title: string
  body: string
  author: User
  tags: string[]
  status: QuestionStatus
  view_count: number
  vote_score: number
  answer_count: number
  accepted_answer_id?: string
  bounty_amount?: number
  bounty_expires_at?: string
  created_at: string
  updated_at: string
  bookmarked_by_current_user: boolean
  user_vote: VoteType
  is_closed: boolean
  close_reason?: string
}

export interface Answer {
  id: string
  body: string
  author: User
  vote_score: number
  is_accepted: boolean
  created_at: string
  updated_at: string
  user_vote: VoteType
}

export interface Comment {
  id: string
  content: string
  author: User
  parent_id: string
  parent_type: 'question' | 'answer'
  created_at: string
}

export interface Tag {
  name: string
  count: number
  description?: string
}

export interface Bounty {
  id: string
  question_id: string
  amount: number
  expires_at: string
  status: 'active' | 'claimed' | 'expired'
}