// Question types
export interface Question {
  id: string
  title: string
  body: string
  author: User
  tags: string[]
  view_count: number
  vote_score: number
  answer_count: number
  accepted_answer_id?: string
  bounty_amount?: number
  bounty_expires_at?: string
  created_at: string
  updated_at: string
  is_closed: boolean
  close_reason?: string
  bookmarked_by_current_user: boolean
  user_vote?: number  // -1, 0, or 1
}

// Answer types
export interface Answer {
  id: string
  body: string
  author: User
  question_id: string
  vote_score: number
  is_accepted: boolean
  created_at: string
  updated_at: string
  user_vote?: number
}

// Comment types
export interface Comment {
  id: string
  content: string
  author: User
  parent_type: 'question' | 'answer'
  parent_id: string
  created_at: string
}

// Vote types
export interface Vote {
  id: string
  value: number  // -1 or 1
  user_id: string
  target_type: 'question' | 'answer'
  target_id: string
}

// Bookmark
export interface Bookmark {
  id: string
  question_id: string
  user_id: string
  created_at: string
}

// Tag
export interface Tag {
  name: string
  count: number
  description?: string
}

// User profile (simplified)
export interface User {
  id: string
  username: string
  avatar_url?: string
  reputation: number
}