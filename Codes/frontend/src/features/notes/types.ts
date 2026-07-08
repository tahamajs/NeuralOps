export interface ResearcherNote {
  id: string;
  title: string;
  content: string;
  author_username: string;
  run_id?: string;
  tags: string[];
  is_public: boolean;
  pinned: boolean;
  reactions: Record<string, string[]>;
  version: number;
  parent_note_id?: string;
  reply_count: number;
  created_at: string;
  updated_at: string;
  shared_with?: string[];
  metadata?: Record<string, any>;
}
