// AgentToolsPage — lists all tools the agent can use
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Wrench,
  Search,
  FlaskConical,
  BookOpen,
  Code2,
  Database,
  Cpu,
  GitBranch,
  FileText,
  BarChart2,
  Zap,
} from "lucide-react";
import type { AgentTool } from "../types";

// Fallback icons by keyword
function getIcon(name: string) {
  if (name.includes("search")) return Search;
  if (name.includes("dataset")) return Database;
  if (name.includes("model")) return Cpu;
  if (name.includes("experiment") || name.includes("launch")) return FlaskConical;
  if (name.includes("paper") || name.includes("library")) return BookOpen;
  if (name.includes("code") || name.includes("latex")) return Code2;
  if (name.includes("citation")) return GitBranch;
  if (name.includes("analysis") || name.includes("metric")) return BarChart2;
  if (name.includes("annotate") || name.includes("note")) return FileText;
  if (name.includes("prompt")) return Zap;
  return Wrench;
}

const CATEGORY_COLORS: Record<string, string> = {
  search: "#818cf8",
  experiment: "#34d399",
  paper: "#f472b6",
  dataset: "#60a5fa",
  model: "#a78bfa",
  analysis: "#fbbf24",
  code: "#fb923c",
  default: "#94a3b8",
};

function getColor(name: string): string {
  for (const [key, color] of Object.entries(CATEGORY_COLORS)) {
    if (name.includes(key)) return color;
  }
  return CATEGORY_COLORS.default;
}

// Fallback static tools list when API isn't available
const FALLBACK_TOOLS: AgentTool[] = [
  { name: "search_papers", description: "Search academic papers on Semantic Scholar and ArXiv", parameters: { query: "string", limit: "number" } },
  { name: "search_library", description: "Search your saved paper library", parameters: { query: "string" } },
  { name: "save_paper", description: "Save a paper to your personal library", parameters: { paper_id: "string", source: "string" } },
  { name: "annotate_paper", description: "Add notes or tags to a saved paper", parameters: { paper_id: "string", notes: "string", tags: "string[]" } },
  { name: "get_citation_graph", description: "Get the citation graph for a paper", parameters: { paper_id: "string", depth: "number" } },
  { name: "search_datasets", description: "Search HuggingFace for datasets", parameters: { query: "string", limit: "number" } },
  { name: "search_models", description: "Search HuggingFace for pretrained models", parameters: { query: "string", task: "string" } },
  { name: "get_dataset_info", description: "Get detailed info about a HuggingFace dataset", parameters: { dataset_id: "string" } },
  { name: "configure_experiment", description: "Propose an experiment configuration for review", parameters: { model_name: "string", task_name: "string", attention_type: "string" } },
  { name: "launch_experiment", description: "Launch an approved experiment configuration", parameters: { config: "ExperimentConfig" } },
  { name: "check_experiment_status", description: "Check the status of a running experiment", parameters: { run_id: "string" } },
  { name: "list_attention_types", description: "List all available attention mechanism types", parameters: {} },
  { name: "list_available_tasks", description: "List all supported GLUE/benchmark tasks", parameters: {} },
  { name: "analyze_training", description: "Analyze training metrics for issues and suggestions", parameters: { run_id: "string" } },
  { name: "suggest_next_experiment", description: "Get AI suggestions for the next experiment to run", parameters: { context: "string" } },
  { name: "read_latex_project", description: "Read the LaTeX source of the current paper project", parameters: {} },
  { name: "edit_latex_file", description: "Edit a file in the LaTeX project", parameters: { filename: "string", content: "string" } },
  { name: "compile_latex", description: "Compile the LaTeX project to PDF", parameters: {} },
  { name: "test_prompt_template", description: "Test a prompt template with sample inputs", parameters: { template: "string", inputs: "object" } },
];

export default function AgentToolsPage() {
  const { data: tools, isLoading } = useQuery<AgentTool[]>({
    queryKey: ["agentTools"],
    queryFn: async () => {
      try {
        const data = (await api.get("/agents/tools/")).data;
        return Array.isArray(data) ? data : (data?.results || FALLBACK_TOOLS);
      } catch {
        return FALLBACK_TOOLS;
      }
    },
  });

  const displayTools = tools ?? FALLBACK_TOOLS;

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
            <Wrench size={20} className="text-white" />
          </div>
          Agent Tools
        </h1>
        <p className="text-slate-400 text-sm mt-1 ml-12">
          {displayTools.length} tools available for the AI agent to use
        </p>
      </div>

      {/* Tool grid */}
      {isLoading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {displayTools.map((tool) => {
            const Icon = getIcon(tool.name);
            const color = tool.category ? (CATEGORY_COLORS[tool.category] ?? CATEGORY_COLORS.default) : getColor(tool.name);
            const paramKeys = Object.keys(tool.parameters ?? {});
            return (
              <Card key={tool.name} className="p-4">
                <div className="flex items-start gap-3">
                  <div
                    className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${color}22`, border: `1px solid ${color}44` }}
                  >
                    <Icon size={17} style={{ color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-bold text-white font-mono">{tool.name}</div>
                    <div className="text-xs text-slate-400 mt-0.5 leading-relaxed">{tool.description}</div>
                    {paramKeys.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {paramKeys.map((param) => (
                          <span
                            key={param}
                            className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                            style={{
                              background: "rgba(255,255,255,0.05)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              color: "#94a3b8",
                            }}
                          >
                            {param}: <span style={{ color }}>{String(tool.parameters[param])}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
