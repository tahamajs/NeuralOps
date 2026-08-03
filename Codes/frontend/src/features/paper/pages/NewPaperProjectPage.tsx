import React, { useState } from "react";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { FileText, Plus, Users, Beaker, FileBox } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function NewPaperProjectPage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title) return;

    setIsSubmitting(true);
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      navigate("/paper"); // Go back or to the new project
    }, 1000);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in pb-20">
      <div className="text-center mb-10">
        <div className="w-16 h-16 bg-neon-blue/10 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-neon-blue/20">
          <FileText className="text-neon-blue w-8 h-8" />
        </div>
        <h1 className="text-4xl font-display font-bold text-white tracking-tight">
          Create Paper Project
        </h1>
        <p className="text-slate-400 mt-3 text-lg font-light max-w-xl mx-auto">
          Start a new collaborative paper, link experiments, and generate an
          outline with your AI Co-Scientist.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <Card className="glass border-white/10 p-6 sm:p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Paper Title
                </label>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g., Attention-based Compression in Large Language Models"
                  className="text-lg py-3"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Abstract / Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Briefly describe the research goals and key findings..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-neon-blue/50 focus:ring-1 focus:ring-neon-blue/50 transition-colors min-h-[150px] resize-y"
                />
              </div>

              <div className="pt-4 flex items-center justify-end gap-4 border-t border-white/10">
                <Button
                  variant="outline"
                  type="button"
                  onClick={() => navigate(-1)}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={isSubmitting || !title}
                  className="bg-neon-blue text-black hover:bg-neon-blue/80 flex items-center gap-2"
                >
                  <Plus size={18} />
                  {isSubmitting ? "Creating..." : "Create Project"}
                </Button>
              </div>
            </form>
          </Card>
        </div>

        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">
            What happens next?
          </h3>

          <Card className="bg-black/40 border-white/5 p-4 flex items-start gap-4">
            <div className="bg-purple-500/10 p-2 rounded-lg text-purple-400 mt-1">
              <Users size={18} />
            </div>
            <div>
              <h4 className="text-white font-medium text-sm">
                Add Collaborators
              </h4>
              <p className="text-xs text-slate-400 mt-1">
                Invite team members to edit the paper in real-time.
              </p>
            </div>
          </Card>

          <Card className="bg-black/40 border-white/5 p-4 flex items-start gap-4">
            <div className="bg-blue-500/10 p-2 rounded-lg text-blue-400 mt-1">
              <Beaker size={18} />
            </div>
            <div>
              <h4 className="text-white font-medium text-sm">
                Link Experiments
              </h4>
              <p className="text-xs text-slate-400 mt-1">
                Import charts and metrics directly from your runs.
              </p>
            </div>
          </Card>

          <Card className="bg-black/40 border-white/5 p-4 flex items-start gap-4">
            <div className="bg-emerald-500/10 p-2 rounded-lg text-emerald-400 mt-1">
              <FileBox size={18} />
            </div>
            <div>
              <h4 className="text-white font-medium text-sm">
                Generate Outline
              </h4>
              <p className="text-xs text-slate-400 mt-1">
                Let the AI agent draft an initial structure for your paper.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
