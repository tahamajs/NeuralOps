import React, { useState } from "react";
import { TerminalCard } from "@/components/ui/TerminalCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useSupportWebSocket } from "@/hooks/useSupportWebSocket";
import { Brain, Search, Code, Calculator, Loader2, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

export function SupportHub() {
  const [input, setInput] = useState("");
  const [problemTitle, setProblemTitle] = useState("");
  const [problemDesc, setProblemDesc] = useState("");
  const [problemCode, setProblemCode] = useState("");
  const [showProblemForm, setShowProblemForm] = useState(false);

  const {
    isConnected,
    response,
    isThinking,
    steps,
    sendMessage,
    submitProblem,
  } = useSupportWebSocket();

  const quickActions = [
    {
      icon: Search,
      label: "Web Search",
      description: "Search documentation and resources",
      action: () => setInput("Search web for: "),
    },
    {
      icon: Code,
      label: "Debug Code",
      description: "Analyze code for issues",
      action: () => setInput("Debug this code: "),
    },
    {
      icon: Calculator,
      label: "Math Help",
      description: "Solve mathematical problems",
      action: () => setInput("Solve this math problem: "),
    },
    {
      icon: Brain,
      label: "Concept Explain",
      description: "Explain ML concepts",
      action: () => setInput("Explain this concept: "),
    },
  ];

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput("");
  };

  const handleSubmitProblem = () => {
    submitProblem({
      title: problemTitle,
      description: problemDesc,
      code: problemCode,
      category: "other",
      severity: "medium",
    });
    setShowProblemForm(false);
    setProblemTitle("");
    setProblemDesc("");
    setProblemCode("");
  };

  return (
    <div className="flex h-full">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Quick Actions */}
        <div className="grid grid-cols-4 gap-3 p-4 border-b border-white/10">
          {quickActions.map((action, idx) => (
            <button
              key={idx}
              onClick={action.action}
              className="glass-terminal p-3 rounded-xl text-left hover:border-neon-mint transition-all group"
            >
              <action.icon
                size={20}
                className="text-neon-mint mb-2 group-hover:scale-110 transition-transform"
              />
              <div className="text-sm font-medium text-white">
                {action.label}
              </div>
              <div className="text-xs text-slate-500">{action.description}</div>
            </button>
          ))}
        </div>

        {/* Response Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {isThinking && (
            <div className="flex items-center gap-2 text-neon-mint">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">AI is analyzing your problem...</span>
            </div>
          )}

          {response && (
            <div className="glass-terminal p-4 rounded-xl">
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {response}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Intermediate steps (tool usage) */}
          {steps.length > 0 && (
            <div className="text-xs text-slate-500 space-y-1">
              {steps.map((step, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span>🔧</span>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-white/10 p-4">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask anything about your research problem..."
              className="flex-1"
              disabled={!isConnected}
            />
            <Button
              onClick={handleSend}
              disabled={!isConnected || !input.trim()}
            >
              <Send size={16} />
            </Button>
          </div>
        </div>
      </div>

      {/* Problem Submission Sidebar */}
      {showProblemForm && (
        <div className="w-96 glass-terminal border-l border-white/10 p-4 overflow-y-auto">
          <h3 className="text-lg font-semibold text-white mb-4">
            Submit a Problem
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Problem Title
              </label>
              <Input
                value={problemTitle}
                onChange={(e) => setProblemTitle(e.target.value)}
                placeholder="Brief problem summary"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Description
              </label>
              <textarea
                value={problemDesc}
                onChange={(e) => setProblemDesc(e.target.value)}
                className="w-full bg-black/30 border border-white/10 rounded p-2 text-white"
                rows={5}
                placeholder="Detailed description of the problem..."
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Code Snippet
              </label>
              <textarea
                value={problemCode}
                onChange={(e) => setProblemCode(e.target.value)}
                className="w-full bg-black/30 border border-white/10 rounded p-2 font-mono text-sm text-white"
                rows={6}
                placeholder="Paste your code here..."
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={handleSubmitProblem} className="flex-1">
                Submit Problem
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowProblemForm(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
