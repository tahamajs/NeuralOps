// src/features/protocols/pages/ProtocolBuilder.tsx
import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { protocolsAPI } from "../services/api";
import { StepEditor } from "../components/StepEditor";
import { ResourceSelector } from "../components/ResourceSelector";
import { ProtocolPreview } from "../components/ProtocolPreview";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Textarea } from "../../../components/ui/Input";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Spinner } from "../../../components/ui/Spinner";
import { Save, Eye, GitBranch, ArrowLeft, Plus } from "lucide-react";
import { Protocol, ProtocolStep } from "../types";
import { useToast } from "../../../components/ui/Toast";

export default function ProtocolBuilder({ onClose }: { onClose?: () => void }) {
  const { protocolId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("edit");
  const [protocol, setProtocol] = useState<Partial<Protocol>>({
    name: "",
    description: "",
    steps: [],
    tags: [],
    status: "draft",
  });
  const [isSaving, setIsSaving] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: existingProtocol, isLoading } = useQuery({
    queryKey: ["protocol", protocolId],
    queryFn: () => protocolsAPI.getProtocol(protocolId!),
    enabled: !!protocolId,
  });

  useEffect(() => {
    if (existingProtocol) {
      setProtocol(existingProtocol);
    }
  }, [existingProtocol]);

  const saveMutation = useMutation({
    mutationFn: () =>
      protocolId
        ? protocolsAPI.updateProtocol(protocolId, protocol)
        : protocolsAPI.createProtocol(protocol),
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ["protocols"] });
      success(protocolId ? "Protocol updated" : "Protocol created");
      if (!protocolId) navigate(`/protocols/${saved.id}`);
      setIsSaving(false);
    },
    onError: () => {
      error("Failed to save");
      setIsSaving(false);
    },
  });

  const handleSave = () => {
    setIsSaving(true);
    saveMutation.mutate();
  };

  const updateStep = (stepId: string, updates: Partial<ProtocolStep>) => {
    setProtocol((prev) => ({
      ...prev,
      steps: prev.steps.map((s) =>
        s.id === stepId ? { ...s, ...updates } : s,
      ),
    }));
  };

  const addStep = () => {
    const newStep: ProtocolStep = {
      id: crypto.randomUUID(),
      title: "New Step",
      description: "",
      type: "text",
      order: (protocol.steps?.length || 0) + 1,
    };
    setProtocol((prev) => ({
      ...prev,
      steps: [...(prev.steps || []), newStep],
    }));
  };

  const removeStep = (stepId: string) => {
    setProtocol((prev) => ({
      ...prev,
      steps: prev.steps.filter((s) => s.id !== stepId),
    }));
  };

  const reorderSteps = (startIndex: number, endIndex: number) => {
    const steps = [...(protocol.steps || [])];
    const [removed] = steps.splice(startIndex, 1);
    steps.splice(endIndex, 0, removed);
    setProtocol((prev) => ({
      ...prev,
      steps: steps.map((s, idx) => ({ ...s, order: idx + 1 })),
    }));
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          {protocolId && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/protocols")}
            >
              <ArrowLeft size={16} />
            </Button>
          )}
          <div>
            <h1 className="text-xl font-bold text-white">
              {protocolId ? "Edit Protocol" : "New Protocol"}
            </h1>
            <p className="text-xs text-slate-400">
              Build reusable experimental workflows
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setActiveTab(activeTab === "edit" ? "preview" : "edit")
            }
          >
            {activeTab === "edit" ? (
              <Eye size={14} className="mr-1" />
            ) : (
              <GitBranch size={14} className="mr-1" />
            )}{" "}
            {activeTab === "edit" ? "Preview" : "Edit"}
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={isSaving || !protocol.name}
          >
            <Save size={14} className="mr-1" /> Save
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4 space-y-3">
          <Input
            label="Protocol Name"
            value={protocol.name}
            onChange={(e) => setProtocol({ ...protocol, name: e.target.value })}
          />
          <Textarea
            label="Description"
            rows={3}
            value={protocol.description}
            onChange={(e) =>
              setProtocol({ ...protocol, description: e.target.value })
            }
          />
          <div className="flex gap-2">
            <label className="text-xs text-slate-400 block">
              Tags (comma separated)
            </label>
            <Input
              value={protocol.tags?.join(", ") || ""}
              onChange={(e) =>
                setProtocol({
                  ...protocol,
                  tags: e.target.value.split(",").map((t) => t.trim()),
                })
              }
            />
          </div>
        </TerminalCard>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabList>
            <Tab value="edit">Edit Steps</Tab>
            <Tab value="resources">Resources</Tab>
            <Tab value="preview">Preview</Tab>
          </TabList>
          <TabPanel value="edit">
            <div className="space-y-4">
              {protocol.steps?.map((step, idx) => (
                <StepEditor
                  key={step.id}
                  step={step}
                  index={idx}
                  onUpdate={updateStep}
                  onRemove={() => removeStep(step.id)}
                  onMoveUp={
                    idx > 0 ? () => reorderSteps(idx, idx - 1) : undefined
                  }
                  onMoveDown={
                    idx < protocol.steps.length - 1
                      ? () => reorderSteps(idx, idx + 1)
                      : undefined
                  }
                />
              ))}
              <Button variant="ghost" onClick={addStep} className="w-full">
                <Plus size={14} className="mr-1" /> Add Step
              </Button>
            </div>
          </TabPanel>
          <TabPanel value="resources">
            <ResourceSelector protocol={protocol} onChange={setProtocol} />
          </TabPanel>
          <TabPanel value="preview">
            <ProtocolPreview protocol={protocol} />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
