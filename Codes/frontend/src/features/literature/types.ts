export interface PaperNode {
  id: string
  title: string
  authors: string[]
  year: number
  citations: number
  abstract?: string
  url?: string
  group?: string  // cluster / research group
}

export interface CitationLink {
  source: string
  target: string
  type: 'references' | 'cites'
  year?: number
}

export interface CitationGraph {
  nodes: PaperNode[]
  links: CitationLink[]
}

export interface PathResult {
  path: string[]
  length: number
}

export interface Recommendation {
  paper_id: string
  title: string
  relevance_score: number
  reason: string
}

export interface ResearchLandscape {
  id: string
  name: string
  topics: string[]
  paper_count: number
  trend: 'rising' | 'stable' | 'declining'
}

export interface LandscapeDetail extends ResearchLandscape {
  top_papers: PaperNode[]
  citation_graph: CitationGraph
  summary: string
}

export interface FeedItem {
  id: string
  paper: PaperNode
  activity_type: 'new_publication' | 'new_citation' | 'trending'
  created_at: string
}