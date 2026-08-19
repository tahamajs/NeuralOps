// src/features/codingAssistant/types.ts

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string | Date;
  suggestions?: CodeSuggestion[];
}

export interface GeneratedFile {
  id: string;
  name: string;
  content: string;
  language: string;
  created_at?: string;
  updated_at?: string;
}

export interface CodeSuggestion {
  id?: string;
  title: string;
  diff?: string;
  modifiedCode?: string;
  originalCode?: string;
  explanation?: string;
}

export interface TestResult {
  passed: boolean;
  test_name: string;
  duration_ms: number;
  error_message?: string;
}

export interface DocumentationItem {
  id: string;
  title: string;
  markdown: string;
  created_at: string;
}

export interface ReviewFeedback {
  id: string;
  score: number;
  summary: string;
  suggestions: string[];
}
