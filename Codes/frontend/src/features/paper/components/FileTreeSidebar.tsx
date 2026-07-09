import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";

interface FileTreeSidebarProps {
  projectId?: string;
  selectedFile: string;
  onFileSelect: (filename: string) => void;
  files: any[];
  onRefresh: () => void;
}

export default function FileTreeSidebar({
  projectId,
  selectedFile,
  onFileSelect,
  files,
  onRefresh,
}: FileTreeSidebarProps) {
  const [newFileName, setNewFileName] = useState("");
  const [showNewFileInput, setShowNewFileInput] = useState(false);

  const createFileMut = useMutation({
    mutationFn: async (filename: string) => {
      await api.post(`/paper/projects/${projectId}/files/`, {
        filename,
        content: "",
      });
    },
    onSuccess: () => {
      onRefresh();
      setShowNewFileInput(false);
      setNewFileName("");
    },
  });

  const deleteFileMut = useMutation({
    mutationFn: async (filename: string) => {
      await api.delete(`/paper/projects/${projectId}/files/${filename}/`);
    },
    onSuccess: (_, filename) => {
      onRefresh();
      if (selectedFile === filename) {
        onFileSelect("main.tex");
      }
    },
  });

  const handleUpload = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.onchange = async (e: any) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async (ev) => {
        const content = ev.target?.result as string;
        await api.post(`/paper/projects/${projectId}/files/`, {
          filename: file.name,
          content,
        });
        onRefresh();
      };
      reader.readAsText(file);
    };
    input.click();
  };

  return (
    <div className="p-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          Files
        </h3>
        <div className="flex gap-1">
          <button
            onClick={() => setShowNewFileInput(!showNewFileInput)}
            className="text-slate-400 hover:text-white text-lg leading-none p-1"
            title="New File"
          >
            +
          </button>
          <button
            onClick={handleUpload}
            className="text-slate-400 hover:text-white text-lg leading-none p-1"
            title="Upload File"
          >
            ↑
          </button>
        </div>
      </div>

      {showNewFileInput && (
        <div className="flex gap-2 mb-2">
          <Input
            placeholder="filename.tex"
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
            className="text-xs py-1"
          />
          <Button
            size="sm"
            onClick={() => createFileMut.mutate(newFileName)}
            disabled={!newFileName}
          >
            Create
          </Button>
        </div>
      )}

      <ul className="space-y-0.5">
        {files.map((file: any) => (
          <li
            key={file.filename}
            className={`flex items-center justify-between px-2 py-1 rounded text-sm cursor-pointer transition-colors ${
              selectedFile === file.filename
                ? "bg-neon-blue/20 text-neon-blue"
                : "text-slate-300 hover:bg-white/5"
            }`}
            onClick={() => onFileSelect(file.filename)}
          >
            <span className="flex items-center gap-2 truncate">
              <span className="text-xs">
                {file.filename.endsWith(".tex")
                  ? "📄"
                  : file.filename.endsWith(".bib")
                    ? "📚"
                    : file.filename.endsWith(".png") ||
                        file.filename.endsWith(".jpg")
                      ? "🖼️"
                      : "📎"}
              </span>
              <span className="truncate text-xs">{file.filename}</span>
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (confirm(`Delete ${file.filename}?`))
                  deleteFileMut.mutate(file.filename);
              }}
              className="text-slate-500 hover:text-red-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity"
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
