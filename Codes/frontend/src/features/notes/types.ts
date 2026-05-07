export interface ResearcherNote {
  id: string
  title: string
  content: string
  author_username: string
  run_id?: string
  tags: string[]
  is_public: boolean
  pinned: boolean
  created_at: string
  updated_at: string
  version: number
}