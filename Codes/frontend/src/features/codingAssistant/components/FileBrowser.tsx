import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { FileIcon, FolderIcon, Plus, Trash2, Edit2 } from "lucide-react";
import { GeneratedFile } from "../types";

interface FileBrowserProps {
  files: GeneratedFile[];
  currentFileId: string | null;
  onSelectFile: (fileId: string) => void;
  onDeleteFile: (fileId: string) => void;
  onCreateFile: (name: string) => void;
}

export function FileBrowser({
  files,
  currentFileId,
  onSelectFile,
  onDeleteFile,
  onCreateFile,
}: FileBrowserProps) {
  const [newFileName, setNewFileName] = useState("");
  const [showNewInput, setShowNewInput] = useState(false);

  const handleCreate = () => {
    if (newFileName.trim()) {
      onCreateFile(newFileName);
      setNewFileName("");
      setShowNewInput(false);
    }
  };

  return (
    <TerminalCard className="h-full overflow-hidden flex flex-col">
      <div className="flex justify-between items-center p-3 border-b border-white/10">
        <h3 className="text-sm font-bold text-white">Files</h3>
        <Button size="sm" variant="ghost" onClick={() => setShowNewInput(true)}>
          <Plus size={14} />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {showNewInput && (
          <div className="flex gap-1 p-1">
            <Input
              size="sm"
              value={newFileName}
              onChange={(e) => setNewFileName(e.target.value)}
              placeholder="filename.py"
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <Button size="sm" variant="ghost" onClick={handleCreate}>
              ✓
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowNewInput(false)}
            >
              ✕
            </Button>
          </div>
        )}
        {files.map((file) => (
          <div
            key={file.id}
            className={`flex items-center justify-between p-2 rounded cursor-pointer ${currentFileId === file.id ? "bg-neon-blue/20" : "hover:bg-white/5"}`}
            onClick={() => onSelectFile(file.id)}
          >
            <div className="flex items-center gap-2">
              <FileIcon size={14} className="text-slate-400" />
              <span className="text-sm text-white">{file.name}</span>
            </div>
            <div className="flex gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteFile(file.id);
                }}
              >
                <Trash2 size={12} className="text-red-400" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </TerminalCard>
  );
}
