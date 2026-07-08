export type AnnotationType =
  "bounding_box" | "polygon" | "classification" | "ner" | "segmentation";
export type ProjectStatus = "draft" | "active" | "completed" | "archived";
export type AnnotationStatus =
  "pending" | "submitted" | "approved" | "rejected";
export type ReviewAction = "approve" | "reject" | "requeue";

export interface AnnotationProject {
  id: string;
  name: string;
  description?: string;
  type: AnnotationType;
  status: ProjectStatus;
  total_items: number;
  annotated_items: number;
  reviewed_items: number;
  created_at: string;
  updated_at: string;
  labels: Label[];
  config: ProjectConfig;
}

export interface Label {
  id: string;
  name: string;
  color: string;
  shortcut?: string;
}

export interface ProjectConfig {
  image_max_size?: number;
  allow_overlap?: boolean;
  require_review?: boolean;
  pre_labeling_enabled?: boolean;
  confidence_threshold?: number;
}

export interface AnnotationItem {
  id: string;
  project_id: string;
  data: {
    image_url?: string;
    text?: string;
    audio_url?: string;
    video_url?: string;
  };
  pre_labels?: PreLabel[];
  annotations: Annotation[];
  status: AnnotationStatus;
  reviewed_by?: string;
  reviewed_at?: string;
  created_at: string;
}

export interface Annotation {
  id: string;
  item_id: string;
  label_id: string;
  geometry?: {
    x: number;
    y: number;
    width: number;
    height: number;
    points?: { x: number; y: number }[];
  };
  text?: string;
  confidence?: number;
  created_by: string;
  created_at: string;
}

export interface PreLabel {
  id: string;
  label_id: string;
  geometry?: any;
  text?: string;
  confidence: number;
  source_model: string;
  created_at: string;
}

export interface ReviewQueueItem {
  item: AnnotationItem;
  assigned_reviewer: string;
  assigned_at: string;
  priority: "high" | "medium" | "low";
}

export interface ReviewResult {
  item_id: string;
  action: ReviewAction;
  comments?: string;
  score?: number;
  reviewed_by: string;
  reviewed_at: string;
}
