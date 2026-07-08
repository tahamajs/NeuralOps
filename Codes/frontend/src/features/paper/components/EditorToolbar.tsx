import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";

interface EditorToolbarProps {
  projectId?: string;
  compiling: boolean;
  onCompile: () => void;
  onDownloadPDF: () => void;
  onSave: () => void;
  autoCompile: boolean;
  onToggleAutoCompile: () => void;
  pdfUrl: string | null;
}

export default function EditorToolbar({
  projectId,
  compiling,
  onCompile,
  onDownloadPDF,
  onSave,
  autoCompile,
  onToggleAutoCompile,
  pdfUrl,
}: EditorToolbarProps) {
  const [showShare, setShowShare] = useState(false);
  const [shareUsername, setShareUsername] = useState("");
  const queryClient = useQueryClient();

  const shareMut = useMutation({
    mutationFn: async (username: string) => {
      await api.post(`/api/paper/projects/${projectId}/collaborators/add/`, {
        username,
      });
    },
    onSuccess: () => {
      setShowShare(false);
      queryClient.invalidateQueries({ queryKey: ["paperProject", projectId] });
    },
  });

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-cyber-800 border-b border-white/5">
      <Button
        size="sm"
        onClick={onCompile}
        disabled={compiling}
        className="bg-green-600 hover:bg-green-500"
      >
        {compiling ? "Compiling..." : "Compile"}
      </Button>

      <Button variant="ghost" size="sm" onClick={onSave}>
        Save
      </Button>

      <Button
        variant="ghost"
        size="sm"
        onClick={onDownloadPDF}
        disabled={!pdfUrl}
      >
        Download PDF
      </Button>

      <div className="w-px h-6 bg-white/10 mx-1" />

      <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
        <input
          type="checkbox"
          checked={autoCompile}
          onChange={onToggleAutoCompile}
          className="accent-neon-blue"
        />
        Auto-compile
      </label>

      <div className="flex-1" />

      <Button
        variant="ghost"
        size="sm"
        onClick={() => setShowShare(!showShare)}
      >
        Share
      </Button>

      {/* Share popup */}
      {showShare && (
        <div className="absolute top-12 right-4 z-50 glass p-4 rounded-xl border border-white/10 shadow-xl w-72">
          <h4 className="text-sm font-bold text-white mb-2">Share Project</h4>
          <div className="flex gap-2">
            <Input
              value={shareUsername}
              onChange={(e) => setShareUsername(e.target.value)}
              placeholder="Username"
              className="text-xs"
            />
            <Button size="sm" onClick={() => shareMut.mutate(shareUsername)}>
              Add
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
