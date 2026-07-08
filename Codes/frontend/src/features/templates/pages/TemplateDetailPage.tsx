import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Textarea } from "../../../components/ui/Input";
import { Layers, Save } from "lucide-react";

export default function TemplateDetailPage() {
  const { templateId } = useParams<{ templateId: string }>();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    description: "",
    category: "",
    configJson: "{}",
  });
  const [parseError, setParseError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  const { data: template, isLoading } = useQuery({
    queryKey: ["template", templateId],
    queryFn: async () =>
      (await api.get(`/api/experiments/templates/${templateId}/`)).data,
    enabled: !!templateId,
  });

  useEffect(() => {
    if (template) {
      setForm({
        name: template.name || "",
        description: template.description || "",
        category: template.category || "",
        configJson: JSON.stringify(template.config || {}, null, 2),
      });
      setDirty(false);
    }
  }, [template]);

  const updateMutation = useMutation({
    mutationFn: async () => {
      let config;
      try {
        config = JSON.parse(form.configJson);
      } catch {
        throw new Error("Invalid JSON in config");
      }
      return api.patch(`/api/experiments/templates/${templateId}/`, {
        name: form.name,
        description: form.description,
        category: form.category,
        config,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["template", templateId] });
      setDirty(false);
      setParseError(null);
    },
    onError: (err: any) => {
      if (err.message === "Invalid JSON in config") {
        setParseError(
          "Config contains invalid JSON. Please fix it before saving.",
        );
      } else {
        setParseError("Failed to save template.");
      }
    },
  });

  const handleChange = (field: string, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    setDirty(true);
    if (field === "configJson") setParseError(null);
  };

  const handleSave = () => {
    // Quick client‑side JSON validation
    try {
      JSON.parse(form.configJson);
      setParseError(null);
    } catch {
      setParseError("Config contains invalid JSON.");
      return;
    }
    updateMutation.mutate();
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <Layers size={24} className="text-neon-blue" />
          Edit Template
        </h1>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={updateMutation.isPending || !dirty}
            loading={updateMutation.isPending}
          >
            <Save size={14} className="mr-1" />
            Save Changes
          </Button>
        </div>
      </div>

      <Card>
        <div className="space-y-5">
          <Input
            label="Template Name"
            value={form.name}
            onChange={(e) => handleChange("name", e.target.value)}
            placeholder="My experiment preset"
          />
          <Input
            label="Description"
            value={form.description}
            onChange={(e) => handleChange("description", e.target.value)}
            placeholder="Brief description of this template"
          />
          <Input
            label="Category"
            value={form.category}
            onChange={(e) => handleChange("category", e.target.value)}
            placeholder="e.g., GLUE, SVD, LoRA"
          />
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1.5">
              Configuration (JSON)
            </label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white font-mono text-xs min-h-[300px] focus:border-neon-blue/50 transition-colors resize-y"
              value={form.configJson}
              onChange={(e) => handleChange("configJson", e.target.value)}
              spellCheck={false}
            />
            {parseError && (
              <p className="text-red-400 text-xs mt-2">{parseError}</p>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
