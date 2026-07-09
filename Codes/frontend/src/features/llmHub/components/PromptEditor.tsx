import React, { useState, useEffect } from "react";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { FileText, Tag, Variable } from "lucide-react";

interface PromptEditorProps {
  initialPrompt?: any;
  onSave: (data: any) => void;
}

export function PromptEditor({ initialPrompt, onSave }: PromptEditorProps) {
  const [name, setName] = useState(initialPrompt?.name || "");
  const [template, setTemplate] = useState(initialPrompt?.template || "");
  const [tags, setTags] = useState(initialPrompt?.tags?.join(", ") || "");
  const [variables, setVariables] = useState<string[]>([]);

  // Automatically extract variables matching {{variable_name}}
  useEffect(() => {
    const matches = template.match(/\{\{([^}]+)\}\}/g);
    if (matches) {
      const vars = Array.from(new Set(matches.map((m) => m.replace(/\{\{|\}\}/g, "").trim())));
      setVariables(vars);
    } else {
      setVariables([]);
    }
  }, [template]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    onSave({
      name,
      template,
      tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      variables,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <Input
        label="Prompt Name *"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. Code Summarizer"
        required
        className="bg-white/5 border-white/10"
      />

      <div>
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
          Prompt Template *
        </label>
        <textarea
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder="You are an expert coder. Summarize the following code: {{code}}"
          rows={6}
          required
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 font-mono focus:border-neon-blue/50 transition-colors"
        />
        <p className="text-[10px] text-slate-500 mt-1">
          Use double curly braces <code className="text-neon-blue font-bold">{"{{variable}}"}</code> to define custom variables.
        </p>
      </div>

      {variables.length > 0 && (
        <div className="p-3 bg-white/5 border border-white/10 rounded-xl space-y-2">
          <div className="text-xs font-bold text-slate-400 flex items-center gap-1.5">
            <Variable size={14} className="text-neon-blue" />
            Detected Template Variables:
          </div>
          <div className="flex flex-wrap gap-1.5">
            {variables.map((v) => (
              <span key={v} className="px-2.5 py-0.5 rounded-md bg-neon-blue/10 border border-neon-blue/20 text-xs font-mono text-neon-blue">
                {v}
              </span>
            ))}
          </div>
        </div>
      )}

      <Input
        label="Tags (comma-separated)"
        value={tags}
        onChange={(e) => setTags(e.target.value)}
        placeholder="summarization, coding, evaluation"
        className="bg-white/5 border-white/10"
      />

      <div className="flex gap-3 justify-end pt-4 border-t border-white/10">
        <Button type="submit" className="bg-neon-blue hover:bg-neon-blue/80 text-white w-full">
          Save Prompt Template
        </Button>
      </div>
    </form>
  );
}
