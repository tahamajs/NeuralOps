import React from "react";
import { Link } from "react-router-dom";
import { AnnotationProject } from "../types";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Folder, Play, Eye, Cpu, BarChart2 } from "lucide-react";

interface ProjectCardProps {
  project: AnnotationProject;
}

export function ProjectCard({ project }: ProjectCardProps) {
  const percentComplete =
    project.total_items > 0
      ? Math.round((project.annotated_items / project.total_items) * 100)
      : 0;

  const typeLabels: Record<string, string> = {
    bounding_box: "Bounding Box",
    polygon: "Polygon Segmentation",
    classification: "Image Classification",
    ner: "Named Entity (NER)",
    segmentation: "Semantic Segmentation",
  };

  return (
    <TerminalCard className="glass p-5 flex flex-col justify-between h-64 hover:scale-[1.01] transition-transform duration-200">
      <div>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2 text-neon-blue">
            <Folder size={18} />
            <span className="text-xs uppercase tracking-wider font-semibold">
              {typeLabels[project.type] || project.type}
            </span>
          </div>
          <span
            className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${
              project.status === "active"
                ? "bg-green-500/10 text-green-400 border-green-500/20"
                : "bg-slate-500/10 text-slate-400 border-slate-500/20"
            }`}
          >
            {project.status}
          </span>
        </div>

        <h3 className="text-base font-bold text-white mt-3 truncate">
          {project.name}
        </h3>
        <p className="text-xs text-slate-400 mt-1 line-clamp-2 h-8">
          {project.description || "No description provided."}
        </p>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
            <span>Progress ({percentComplete}%)</span>
            <span>
              {project.annotated_items} / {project.total_items} items
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-neon-blue rounded-full transition-all duration-300"
              style={{ width: `${percentComplete}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-4">
        <Link to={`/annotation/project/${project.id}/label`}>
          <Button size="xs" className="w-full flex items-center justify-center gap-1">
            <Play size={10} /> Label
          </Button>
        </Link>
        <Link to={`/annotation/project/${project.id}/review`}>
          <Button
            size="xs"
            variant="outline"
            className="w-full flex items-center justify-center gap-1 border-slate-700 hover:bg-slate-800 text-slate-300"
          >
            <Eye size={10} /> Review
          </Button>
        </Link>
        <Link to={`/annotation/project/${project.id}/pre-label`}>
          <Button
            size="xs"
            variant="outline"
            className="w-full flex items-center justify-center gap-1 border-slate-700 hover:bg-slate-800 text-slate-300"
          >
            <Cpu size={10} /> Auto
          </Button>
        </Link>
      </div>
    </TerminalCard>
  );
}
