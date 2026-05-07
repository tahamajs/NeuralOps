export interface ResearcherNote {
  id: string
  title: string
  content: string
  author_username: string
  run_id?: string
  tags: string[]
  is_public: boolean
  pinned: boolean
  reactions: Record<string, string[]>
  shared_with: string[]
  version: number
  parent_note_id?: string
  created_at: string
  updated_at: string
}