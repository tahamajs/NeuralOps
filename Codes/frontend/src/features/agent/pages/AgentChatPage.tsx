import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import {
  Bot,
  Search,
  BookOpen,
  FlaskConical,
  FileText,
  GitBranch,
  Wrench,
} from "lucide-react";
import { useAgentChat } from "../hooks/useAgentChat";
import SessionSidebar from "../components/SessionSidebar";
import ChatMessageBubble from "../components/ChatMessageBubble";
import ChatInput from "../components/ChatInput";
import PaperSearchModal from "../components/PaperSearchModal";
import CitationGraphModal from "../components/CitationGraphModal";
import AgentStatusBadge from "../components/AgentStatusBadge";

export default function AgentChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showSidebar, setShowSidebar] = useState(false);
  const [showPaperSearch, setShowPaperSearch] = useState(false);
  const [showCitationGraph, setShowCitationGraph] = useState(false);
  const [showArtifacts, setShowArtifacts] = useState(false);
  const [selectedPersona, setSelectedPersona] = useState("coscientist");
  const [selectedModel, setSelectedModel] = useState("claude-3-5-sonnet");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, loading, sending, wsStatus, sendMessage } =
    useAgentChat(sessionId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleNewChat = async () => {
    try {
      const res = await api.post("/agents/chats/", { title: "New Research Session" });
      navigate(`/agent/${res.data.id}`);
    } catch {
      navigate(`/agent/session-${Date.now()}`);
    }
  };

  const handleSelectSession = (id: string) => navigate(`/agent/${id}`);
  const handleDeleteSession = async (id: string) => {
    try {
      await api.delete(`/agents/chats/${id}/`);
      queryClient.invalidateQueries({ queryKey: ["agentSessions"] });
    } catch {}
    if (id === sessionId) navigate("/agent");
  };

  const handlePromptSend = (text: string) => {
    sendMessage(text, (newId) => {
      navigate(`/agent/${newId}`);
    });
  };

  const quickPrompts = [
    { title: "Dynamic Entropy SVD", desc: "Calculate singular entropy and layer rank allocation budget", prompt: "Compute dynamic entropy SVD rank allocation for BERT-base with 50% parameter reduction budget." },
    { title: "Griffin-SSD Kernel", desc: "Synthesize Triton fused kernel with in-SRAM RMSNorm", prompt: "Generate Triton fused low-rank attention kernel with in-SRAM RMSNorm and analytical backward pass." },
    { title: "LaTeX Monograph", desc: "Draft academic proof for Riemannian Stiefel QR retraction", prompt: "Write LaTeX proof and mathematical derivation for Riemannian Stiefel QR retraction orthogonality." },
    { title: "ArXiv Paper Search", desc: "Discover latest 2026 low-rank linear attention papers", prompt: "Search papers about linear attention, spectral LoRA, and state space models on ArXiv." },
  ];

  return (
    <div className="h-[calc(100vh-100px)] flex gap-4 max-w-7xl mx-auto p-4">
      {/* Mobile sidebar toggle */}
      <button
        className="lg:hidden fixed bottom-6 left-6 z-50 glass p-3 rounded-full shadow-2xl border border-neon-blue/30 text-neon-blue"
        onClick={() => setShowSidebar(!showSidebar)}
      >
        <Bot size={22} />
      </button>

      {/* Sessions Sidebar */}
      <div
        className={`${showSidebar ? "block" : "hidden"} lg:block fixed lg:relative z-40 h-full w-72 flex-shrink-0`}
      >
        <SessionSidebar
          activeId={sessionId}
          onSelect={handleSelectSession}
          onNew={handleNewChat}
          onDelete={handleDeleteSession}
        />
      </div>

      {/* Main Chat Workspace */}
      <div className="flex-1 flex flex-col min-w-0 glass rounded-2xl border border-white/10 overflow-hidden shadow-2xl backdrop-blur-xl">
        {/* Top Intelligence Control Bar */}
        <div className="bg-white/[0.03] border-b border-white/10 px-5 py-3 flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Bot size={20} className="text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold text-white tracking-wide">
                  Autonomous AI Co-Scientist
                </h1>
                <AgentStatusBadge status={wsStatus} />
              </div>
              <p className="text-[11px] text-slate-400">
                Self-directed hypothesis synthesis, kernel generation & experiment orchestration
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Persona Selector */}
            <select
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="bg-black/40 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500/50"
            >
              <option value="coscientist">🔬 AI Co-Scientist</option>
              <option value="kernel_architect">⚡ Triton Kernel Architect</option>
              <option value="manifold_math">📐 Manifold Mathematician</option>
              <option value="latex_editor">📄 LaTeX Editor</option>
            </select>

            {/* Model Selector */}
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-black/40 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500/50"
            >
              <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
              <option value="deepseek-v3">DeepSeek V3 (Reasoning)</option>
              <option value="gpt-4o">GPT-4o Omnimodal</option>
              <option value="triton-local">Local Triton SVD Engine</option>
            </select>

            <Button
              variant="ghost"
              size="sm"
              className="border border-white/10 text-xs"
              onClick={() => setShowPaperSearch(true)}
            >
              <Search size={14} className="mr-1 text-cyan-400" /> Search Papers
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="border border-white/10 text-xs"
              onClick={() => setShowCitationGraph(true)}
            >
              <GitBranch size={14} className="mr-1 text-purple-400" /> Citation Graph
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`border border-white/10 text-xs ${showArtifacts ? "bg-cyan-500/20 border-cyan-500/50" : ""}`}
              onClick={() => setShowArtifacts(!showArtifacts)}
            >
              <FileText size={14} className="mr-1 text-emerald-400" /> Artifacts
            </Button>
          </div>
        </div>

        {/* Message Viewport & Split Pane */}
        <div className="flex-1 flex min-h-0">
          {/* Messages Column */}
          <div className="flex-1 flex flex-col min-w-0 p-4 overflow-hidden">
            <div className="flex-1 overflow-y-auto scrollbar-thin pr-2 space-y-3">
              {(Array.isArray(messages) ? messages : []).length === 0 && !loading && (
                <div className="flex flex-col items-center justify-center h-full text-center px-4 max-w-2xl mx-auto">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center mb-4 text-cyan-400">
                    <FlaskConical size={32} />
                  </div>
                  <h2 className="text-lg font-bold text-white mb-2">
                    Research Assistant Workspace
                  </h2>
                  <p className="text-xs text-slate-400 mb-6 leading-relaxed">
                    Formulate novel attention architectures, compute singular value entropy, execute distributed GPU training, and generate publication-ready LaTeX documentation.
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full text-left">
                    {quickPrompts.map((item, idx) => (
                      <button
                        key={idx}
                        onClick={() => handlePromptSend(item.prompt)}
                        className="glass-hover p-3 rounded-xl border border-white/5 hover:border-cyan-500/30 text-left transition-all group"
                      >
                        <div className="text-xs font-semibold text-cyan-300 group-hover:text-cyan-200 flex items-center gap-1.5 mb-1">
                          <Wrench size={13} className="text-cyan-400" />
                          {item.title}
                        </div>
                        <div className="text-[11px] text-slate-400 group-hover:text-slate-300">
                          {item.desc}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {(Array.isArray(messages) ? messages : []).map((msg) => (
                <ChatMessageBubble key={msg.id} msg={msg} />
              ))}

              {(loading || sending) && (
                <div className="flex gap-3 items-start animate-fade-in">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center text-white flex-shrink-0 shadow-md">
                    <Bot size={16} />
                  </div>
                  <div className="glass rounded-2xl px-4 py-3 border border-cyan-500/20 max-w-lg">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-semibold text-cyan-400">Reasoning & Synthesizing</span>
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                    </div>
                    <div className="flex gap-1.5 py-1">
                      <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: "0s" }} />
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Bar */}
            <div className="pt-3">
              <ChatInput onSend={handlePromptSend} disabled={sending || loading} />
            </div>
          </div>

          {/* Right Artifact Preview Panel */}
          {showArtifacts && (
            <div className="w-96 border-l border-white/10 bg-black/40 flex flex-col p-4 overflow-y-auto scrollbar-thin">
              <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-4">
                <div className="flex items-center gap-2">
                  <FileText size={16} className="text-emerald-400" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">Research Artifacts</span>
                </div>
                <button
                  onClick={() => setShowArtifacts(false)}
                  className="text-xs text-slate-400 hover:text-white"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3">
                <div className="glass p-3 rounded-xl border border-emerald-500/20">
                  <div className="text-xs font-bold text-emerald-400 mb-1">Dynamic Entropy SVD Model</div>
                  <div className="text-[11px] text-slate-400 mb-2">Layer-adaptive singular rank allocation algorithm</div>
                  <pre className="text-[10px] bg-black/60 p-2 rounded text-slate-300 overflow-x-auto">
{`from framework.compression_engine.dynamic_rank import DynamicEntropySVDCompressor
compressor = DynamicEntropySVDCompressor(
    target_compression_ratio=0.5,
    min_rank=4,
    max_rank=32
)`}
                  </pre>
                </div>

                <div className="glass p-3 rounded-xl border border-cyan-500/20">
                  <div className="text-xs font-bold text-cyan-400 mb-1">Stiefel QR Retraction</div>
                  <div className="text-[11px] text-slate-400 mb-2">Preserves $\|W^\top W - I\|_F &lt; 10^{`{-12}`}$</div>
                  <div className="text-[11px] bg-cyan-950/30 p-2 rounded text-cyan-200 font-mono">
                    W_(t+1) = Q · diag(sgn(diag(R)))
                  </div>
                </div>

                <div className="glass p-3 rounded-xl border border-purple-500/20">
                  <div className="text-xs font-bold text-purple-400 mb-1">Master Monograph Reference</div>
                  <div className="text-[11px] text-slate-400">paper/main.pdf (319 pages compiled)</div>
                  <button
                    onClick={() => navigate("/overleaf")}
                    className="mt-2 text-xs bg-purple-600/30 hover:bg-purple-600/50 text-purple-200 px-3 py-1 rounded-lg border border-purple-500/30 transition-all w-full text-center"
                  >
                    Open LaTeX Editor
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modals */}
      {showPaperSearch && (
        <PaperSearchModal
          onClose={() => setShowPaperSearch(false)}
          onPaperSaved={() =>
            queryClient.invalidateQueries({ queryKey: ["savedPapers"] })
          }
        />
      )}
      {showCitationGraph && (
        <CitationGraphModal onClose={() => setShowCitationGraph(false)} />
      )}
    </div>
  );
}
