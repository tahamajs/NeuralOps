export type DatabaseSource =
  | "arxiv"
  | "pubmed"
  | "semantic_scholar"
  | "crossref"
  | "openalex"
  | "dblp"
  | "doaj";
export type HypothesisStatus = "generated" | "reviewed" | "ranked" | "evolved";
export type GrantStatus = "open" | "closing" | "expired";

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  year: number;
  citations: number;
  url: string;
  pdf_url?: string;
  source: DatabaseSource;
  keywords: string[];
}

export interface CitationLink {
  source: string;
  target: string;
  type: "references" | "cites";
}

export interface LiteratureMap {
  nodes: Paper[];
  links: CitationLink[];
  seed_paper_id: string;
  total_papers: number;
}

export interface SynthesisResult {
  summary: string;
  key_findings: string[];
  research_gaps: string[];
  future_directions: string[];
  top_papers: Paper[];
  timeline: { year: number; count: number }[];
}

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  scientific_merit: number;
  novelty: number;
  feasibility: number;
  impact: number;
  total_score: number;
  status: HypothesisStatus;
  parent_id?: string;
  created_at: string;
  related_papers: string[];
}

export interface Grant {
  id: string;
  title: string;
  funding_body: string;
  amount: number;
  deadline: string;
  status: GrantStatus;
  eligibility: string[];
  description: string;
  application_url: string;
  match_score: number;
  tags: string[];
}
