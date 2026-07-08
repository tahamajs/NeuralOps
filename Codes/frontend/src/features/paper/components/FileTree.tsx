import { useState } from "react";
import { ChevronDown, ChevronRight, FileText, Folder } from "lucide-react";
import type { PaperFile } from "../types";

interface FileTreeProps {
  files: PaperFile[];
  onSelect: (path: string) => void;
  currentFile: string | null;
}

export function FileTree({ files, onSelect, currentFile }: FileTreeProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggleExpand = (path: string) => {
    const newExpanded = new Set(expanded);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpanded(newExpanded);
  };

  const renderTree = (items: PaperFile[], basePath = "") => {
    return items.map((item) => {
      const fullPath = basePath ? `${basePath}/${item.name}` : item.name;
      const isExpanded = expanded.has(fullPath);

      if (item.type === "directory") {
        return (
          <div key={fullPath} className="ml-4">
            <div
              className="flex items-center gap-1 py-1 cursor-pointer hover:bg-white/5 rounded"
              onClick={() => toggleExpand(fullPath)}
            >
              {isExpanded ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <Folder size={14} className="text-neon-blue" />
              <span className="text-sm text-white">{item.name}</span>
            </div>
            {isExpanded && item.children && (
              <div className="ml-4">{renderTree(item.children, fullPath)}</div>
            )}
          </div>
        );
      }

      return (
        <div
          key={fullPath}
          className={`ml-6 flex items-center gap-1 py-1 cursor-pointer hover:bg-white/5 rounded ${currentFile === fullPath ? "bg-neon-blue/10 text-neon-blue" : ""}`}
          onClick={() => onSelect(fullPath)}
        >
          <FileText size={14} className="text-slate-400" />
          <span className="text-sm text-white">{item.name}</span>
        </div>
      );
    });
  };

  return <div className="space-y-1">{renderTree(files)}</div>;
}
