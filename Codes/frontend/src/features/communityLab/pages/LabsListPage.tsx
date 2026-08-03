// src/features/communityLab/pages/LabsListPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { communityLabAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Users, Plus, Settings, Sparkles, Lock, Globe } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function LabsListPage({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newLabName, setNewLabName] = useState("");
  const [newLabDesc, setNewLabDesc] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const navigate = useNavigate();
  const [isPrivate, setIsPrivate] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: labs, isLoading } = useQuery({
    queryKey: ["labs"],
    queryFn: () => communityLabAPI.getLabs(),
  });

  const createLab = useMutation({
    mutationFn: () =>
      communityLabAPI.createLab({
        name: newLabName,
        description: newLabDesc,
        visibility: isPrivate ? "private" : "public",
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["labs"] });
      setShowCreateModal(false);
      setNewLabName("");
      setNewLabDesc("");
      setIsPrivate(false);
      success("Lab created", "Your new research lab is ready.");
    },
    onError: () => error("Failed to create lab"),
  });

  if (isLoading) return <Spinner />;

  const labsArray = Array.isArray(labs) ? labs : (labs as any)?.results || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Users size={20} className="text-neon-mint" /> Research Labs
          </h1>
          <p className="text-xs text-slate-400">
            Collaborate with agents and researchers
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Lab
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {labsArray.map((lab: any) => (
          <div
            key={lab.id}
            onClick={() => navigate(`/community/labs/${lab.id}`)}
            className="cursor-pointer"
          >
            <TerminalCard className="p-4">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-white">{lab.name}</h3>
                    {lab.is_private ? (
                      <Lock size={12} className="text-slate-500" />
                    ) : (
                      <Globe size={12} className="text-slate-500" />
                    )}
                  </div>
                  <p className="text-sm text-slate-400 mt-1 line-clamp-2">
                    {lab.description || "No description"}
                  </p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      <Users size={12} /> {lab.member_count || 0} members
                    </span>
                    <span className="flex items-center gap-1">
                      <Sparkles size={12} /> {lab.agent_count || 0} agents
                    </span>
                    {lab.role === "owner" && (
                      <Badge variant="info">Owner</Badge>
                    )}
                  </div>
                </div>
                <Button size="sm" variant="ghost">
                  <Settings size={14} />
                </Button>
              </div>
            </TerminalCard>
          </div>
        ))}
        {labsArray.length === 0 && (
          <TerminalCard className="p-8 text-center text-slate-500">
            <Users size={32} className="mx-auto mb-2 opacity-30" />
            <p>No labs yet. Create one to start collaborating!</p>
          </TerminalCard>
        )}
      </div>

      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Research Lab"
      >
        <div className="space-y-4">
          <Input
            label="Lab Name"
            value={newLabName}
            onChange={(e) => setNewLabName(e.target.value)}
            placeholder="e.g., AI for Science Lab"
          />
          <Input
            label="Description"
            value={newLabDesc}
            onChange={(e) => setNewLabDesc(e.target.value)}
            placeholder="What will this lab work on?"
          />
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={isPrivate}
              onChange={(e) => setIsPrivate(e.target.checked)}
              className="accent-neon-blue"
            />
            <span className="text-sm text-slate-400">
              Private lab (invite only)
            </span>
          </label>
          <Button
            onClick={() => createLab.mutate()}
            disabled={!newLabName.trim() || createLab.isPending}
            className="w-full"
          >
            {createLab.isPending ? "Creating..." : "Create Lab"}
          </Button>
        </div>
      </Modal>
    </div>
  );
}
