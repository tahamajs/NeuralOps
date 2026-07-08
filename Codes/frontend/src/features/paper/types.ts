export type PaperProject = {
  id: string;
  name: string;
  description?: string;
  rootFile: string; // main .tex file
  created_at: string;
  updated_at: string;
  gitRepositoryUrl?: string;
  collaborators: string[];
};

export type PaperFile = {
  path: string;
  name: string;
  type: "file" | "directory";
  content?: string; // for text files
  children?: PaperFile[]; // for directories
  size?: number;
  lastModified?: string;
};

export type CompilationStatus = "idle" | "compiling" | "success" | "failed";

export type CompilationResult = {
  status: CompilationStatus;
  pdfUrl?: string;
  logs?: string;
  error?: string;
};
