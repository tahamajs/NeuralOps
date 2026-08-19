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
  Sparkles,
  Zap,
  Cpu,
  RefreshCw,
  Copy,
  CheckCircle2,
  Sliders,
  Send,
  Plus,
  Trash2,
  Download,
  Terminal,
  Code,
  Layers,
  ArrowRight,
  HelpCircle,
  Database,
  ExternalLink,
} from "lucide-react";
import { useAgentChat } from "../hooks/useAgentChat";
import SessionSidebar from "../components/SessionSidebar";
import ChatMessageBubble from "../components/ChatMessageBubble";
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
  const [inputVal, setInputVal] = useState("");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [copiedFormula, setCopiedFormula] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, loading, sending, wsStatus, sendMessage } =
    useAgentChat(sessionId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, sending]);

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
    if (!text.trim()) return;
    sendMessage(text, (newId) => {
      navigate(`/agent/${newId}`);
    });
    setInputVal("");
  };

  const personas = [
    {
      id: "coscientist",
      name: "Autonomous Co-Scientist",
      icon: "🔬",
      badge: "Hypothesis Synthesis",
      desc: "Literature exploration, singular entropy rank allocation, and automated hypothesis tournament generation.",
    },
    {
      id: "kernel_architect",
      name: "Triton Kernel Architect",
      icon: "⚡",
      badge: "GPU Speedup",
      desc: "Synthesizes fused Triton GPU low-rank forward/backward kernels with in-SRAM RMSNorm.",
    },
    {
      id: "manifold_math",
      name: "Manifold Mathematician",
      icon: "📐",
      badge: "Stiefel Geometry",
      desc: "Riemannian Stiefel QR retractions, orthogonal tangent projections, and Petrov-Galerkin proofs.",
    },
    {
      id: "latex_editor",
      name: "LaTeX Monograph Publisher",
      icon: "📘",
      badge: "Paper Generator",
      desc: "Automated LaTeX derivation typesetting, ArXiv compilation, and BibTeX citation linking.",
    },
  ];

  const quickPromptCategories = [
    { id: "all", label: "All Topics" },
    { id: "svd", label: "SVD & Compression" },
    { id: "triton", label: "Triton Kernels" },
    { id: "math", label: "Riemannian Math" },
    { id: "paper", label: "Literature & Paper" },
  ];

  const quickPrompts = [
    {
      cat: "svd",
      title: "Dynamic Entropy SVD Budget",
      desc: "Calculate layer-wise singular value entropy H(Σ_l) and compute optimal rank budget for 50% parameter reduction.",
      prompt: "Compute dynamic entropy SVD rank allocation for BERT-base with 50% parameter reduction budget and min_rank=4, max_rank=32.",
      icon: <Sparkles size={16} className="text-cyan-400" />,
      tag: "Compression",
    },
    {
      cat: "triton",
      title: "Fused Triton Low-Rank Kernel",
      desc: "Generate PyTorch / Triton forward and analytical backward kernel for Q, K low-rank factorization with in-SRAM RMSNorm.",
      prompt: "Write a high-performance Triton fused low-rank attention kernel with in-SRAM RMSNorm and analytical autograd backward pass.",
      icon: <Zap size={16} className="text-yellow-400" />,
      tag: "GPU Kernel",
    },
    {
      cat: "math",
      title: "Riemannian Stiefel QR Proof",
      desc: "Derive mathematical proof for horizontal tangent projection grad L(W) = G - WGᵀW and QR retraction orthogonality.",
      prompt: "Provide exact mathematical formulation and proof for Riemannian gradient projection and QR retraction on Stiefel manifold St(d, r).",
      icon: <Code size={16} className="text-purple-400" />,
      tag: "Mathematics",
    },
    {
      cat: "paper",
      title: "ArXiv Literature Synthesis",
      desc: "Survey 2024–2026 low-rank attention methods: Griffin-SSD, Gated DeltaNet, MLA, and Dynamic Entropy SVD.",
      prompt: "Synthesize latest 2026 ArXiv papers comparing Linear Attention, Griffin-SSD, DeepSeek MLA, and Spectral SVD Compression.",
      icon: <BookOpen size={16} className="text-emerald-400" />,
      tag: "Literature",
    },
    {
      cat: "svd",
      title: "Spectral Residual LoRA (ResSVD)",
      desc: "Freeze dominant top-k singular vectors and fine-tune low-rank residual adapters on the spectral tail.",
      prompt: "Explain and generate code for Spectral Residual SVD (ResSVD) fine-tuning on GLUE benchmark tasks.",
      icon: <Layers size={16} className="text-blue-400" />,
      tag: "Fine-Tuning",
    },
    {
      cat: "paper",
      title: "GLUE Benchmark Training Config",
      desc: "Auto-generate multi-task training recipe for MRPC, SST-2, and MNLI with SVD linear attention.",
      prompt: "Generate an end-to-end experiment recipe and launch payload for fine-tuning SVD-Linear BERT on GLUE MRPC with lr=2e-5.",
      icon: <Database size={16} className="text-rose-400" />,
      tag: "Experiment",
    },
  ];

  const filteredPrompts =
    activeCategory === "all"
      ? quickPrompts
      : quickPrompts.filter((p) => p.cat === activeCategory);

  const activePersonaObj =
    personas.find((p) => p.id === selectedPersona) || personas[0];

  const handleCopyFormula = (formula: string) => {
    navigator.clipboard.writeText(formula);
    setCopiedFormula(true);
    setTimeout(() => setCopiedFormula(false), 2000);
  };

  return (
    <div className="h-[calc(100vh-100px)] flex gap-4 max-w-7xl mx-auto p-4 animate-fade-in select-text">
      {/* Mobile sidebar toggle */}
      <button
        className="lg:hidden fixed bottom-6 left-6 z-50 glass p-3 rounded-full shadow-2xl border border-cyan-500/30 text-cyan-400 cursor-pointer"
        onClick={() => setShowSidebar(!showSidebar)}
      >
        <Bot size={22} />
      </button>

      {/* Sessions Sidebar */}
      <div
        className={`${showSidebar ? "block" : "hidden"} lg:block fixed lg:relative z-40 h-full w-72 flex-shrink-0`}
      >
        <div className="h-full flex flex-col glass rounded-2xl border border-white/10 overflow-hidden shadow-2xl backdrop-blur-xl">
          <div className="p-3.5 border-b border-white/10 bg-white/[0.02] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center text-white shadow-md">
                <Bot size={15} />
              </div>
              <span className="text-xs font-bold text-white tracking-wide">
                Agent Sessions
              </span>
            </div>
            <button
              onClick={handleNewChat}
              className="px-2.5 py-1 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black text-[11px] font-bold flex items-center gap-1 shadow-md shadow-cyan-500/20 transition-all cursor-pointer"
            >
              <Plus size={13} /> New
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            <SessionSidebar
              activeId={sessionId}
              onSelect={handleSelectSession}
              onNew={handleNewChat}
              onDelete={handleDeleteSession}
            />
          </div>
        </div>
      </div>

      {/* Main Chat Workspace */}
      <div className="flex-1 flex flex-col min-w-0 glass rounded-2xl border border-white/10 overflow-hidden shadow-2xl backdrop-blur-xl">
        {/* Top Intelligence Control Bar */}
        <div className="bg-white/[0.03] border-b border-white/10 px-5 py-3 flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <span className="text-lg">{activePersonaObj.icon}</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold text-white tracking-wide flex items-center gap-1.5">
                  {activePersonaObj.name}
                  <span className="px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] border border-cyan-500/30 font-mono font-medium">
                    {activePersonaObj.badge}
                  </span>
                </h1>
                <AgentStatusBadge status={wsStatus} />
              </div>
              <p className="text-[11px] text-slate-400 line-clamp-1">
                {activePersonaObj.desc}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Persona Selector */}
            <select
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="bg-black/50 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500/50 cursor-pointer"
            >
              {personas.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.icon} {p.name}
                </option>
              ))}
            </select>

            {/* Model Selector */}
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-black/50 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500/50 cursor-pointer font-mono"
            >
              <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
              <option value="deepseek-v3">DeepSeek V3 (Reasoning)</option>
              <option value="gpt-4o">GPT-4o Omnimodal</option>
              <option value="gemini-2-flash">Gemini 2.0 Flash</option>
              <option value="triton-local">Local Triton SVD Engine</option>
            </select>

            <button
              onClick={() => setShowPaperSearch(true)}
              className="glass-hover border border-white/10 text-slate-200 px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <Search size={13} className="text-cyan-400" />
              <span className="hidden sm:inline">Search Papers</span>
            </button>

            <button
              onClick={() => setShowCitationGraph(true)}
              className="glass-hover border border-white/10 text-slate-200 px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <GitBranch size={13} className="text-purple-400" />
              <span className="hidden sm:inline">Citation Map</span>
            </button>

            <button
              onClick={() => setShowArtifacts(!showArtifacts)}
              className={`border border-white/10 px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                showArtifacts
                  ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-300 shadow-md shadow-cyan-500/10"
                  : "glass-hover text-slate-200"
              }`}
            >
              <FileText size={13} className="text-emerald-400" />
              <span>Artifacts</span>
            </button>
          </div>
        </div>

        {/* Message Viewport & Split Pane */}
        <div className="flex-1 flex min-h-0">
          {/* Messages Column */}
          <div className="flex-1 flex flex-col min-w-0 p-4 overflow-hidden">
            <div className="flex-1 overflow-y-auto scrollbar-thin pr-2 space-y-4">
              {(Array.isArray(messages) ? messages : []).length === 0 && !loading && (
                <div className="flex flex-col items-center justify-center h-full text-center px-4 max-w-4xl mx-auto py-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500/20 via-indigo-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center mb-4 text-cyan-400 shadow-2xl shadow-cyan-500/10">
                    <FlaskConical size={32} />
                  </div>
                  <h2 className="text-xl font-bold text-white mb-2 tracking-tight">
                    Autonomous AI Research Workspace
                  </h2>
                  <p className="text-xs text-slate-400 mb-5 leading-relaxed max-w-xl">
                    Formulate novel attention architectures, compute singular value entropy,
                    synthesize Triton GPU kernels, execute distributed training runs, and author
                    publication-grade LaTeX monographs.
                  </p>

                  {/* Category filter pills */}
                  <div className="flex items-center gap-1.5 mb-5 flex-wrap justify-center">
                    {quickPromptCategories.map((c) => (
                      <button
                        key={c.id}
                        onClick={() => setActiveCategory(c.id)}
                        className={`px-3 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                          activeCategory === c.id
                            ? "bg-cyan-500 text-black shadow-md shadow-cyan-500/20"
                            : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        {c.label}
                      </button>
                    ))}
                  </div>

                  {/* Prompt Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 w-full text-left">
                    {filteredPrompts.map((item, idx) => (
                      <button
                        key={idx}
                        onClick={() => handlePromptSend(item.prompt)}
                        className="glass-hover p-4 rounded-xl border border-white/5 hover:border-cyan-500/40 text-left transition-all group bg-white/[0.01] hover:bg-cyan-950/20 cursor-pointer shadow-lg"
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="text-xs font-bold text-cyan-300 group-hover:text-cyan-200 flex items-center gap-2">
                            {item.icon}
                            {item.title}
                          </div>
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-slate-400 border border-white/5 font-mono">
                            {item.tag}
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-400 group-hover:text-slate-300 leading-relaxed">
                          {item.desc}
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Mathematical Fast Inserter */}
                  <div className="mt-6 w-full glass p-3.5 rounded-xl border border-white/10 flex items-center justify-between gap-2 text-xs flex-wrap">
                    <span className="text-[11px] font-bold text-slate-400 flex items-center gap-1.5">
                      <Code size={13} className="text-purple-400" /> Math Snippets:
                    </span>
                    <div className="flex items-center gap-2 flex-wrap">
                      <button
                        onClick={() => handleCopyFormula("W^\\top W = I_r")}
                        className="bg-white/5 hover:bg-cyan-500/20 px-2 py-1 rounded text-cyan-300 font-mono text-[11px] transition-colors cursor-pointer"
                      >
                        {"$W^⊤W = I$"}
                      </button>
                      <button
                        onClick={() => handleCopyFormula("\\mathrm{grad} \\mathcal{L}(W) = G - WG^\\top W")}
                        className="bg-white/5 hover:bg-cyan-500/20 px-2 py-1 rounded text-cyan-300 font-mono text-[11px] transition-colors cursor-pointer"
                      >
                        {"grad L(W) = G - WGᵀW"}
                      </button>
                      <button
                        onClick={() => handleCopyFormula("r_l \\propto H(\\Sigma_l)")}
                        className="bg-white/5 hover:bg-cyan-500/20 px-2 py-1 rounded text-cyan-300 font-mono text-[11px] transition-colors cursor-pointer"
                      >
                        {"r_l ∝ H(Σ_l)"}
                      </button>
                      <button
                        onClick={() => handleCopyFormula("\\mathrm{Retr}_W(\\xi) = \\mathrm{qf}(W+\\xi)")}
                        className="bg-white/5 hover:bg-cyan-500/20 px-2 py-1 rounded text-cyan-300 font-mono text-[11px] transition-colors cursor-pointer"
                      >
                        {"Retr_W(ξ) = qf(W+ξ)"}
                      </button>
                    </div>
                    {copiedFormula && (
                      <span className="text-emerald-400 text-[11px] flex items-center gap-1">
                        <CheckCircle2 size={12} /> Copied!
                      </span>
                    )}
                  </div>
                </div>
              )}

              {(Array.isArray(messages) ? messages : []).map((msg) => (
                <ChatMessageBubble key={msg.id} msg={msg} />
              ))}

              {(loading || sending) && (
                <div className="flex gap-3 items-start animate-fade-in">
                  <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center text-white flex-shrink-0 shadow-md">
                    <Bot size={18} />
                  </div>
                  <div className="glass rounded-2xl px-4 py-3.5 border border-cyan-500/30 max-w-xl shadow-xl">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-bold text-cyan-400">
                        Synthesizing SVD Architecture & Mathematical Proof
                      </span>
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                    </div>
                    <div className="flex gap-1.5 py-1">
                      <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: "0s" }} />
                      <div className="w-2.5 h-2.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                      <div className="w-2.5 h-2.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }} />
                    </div>
                    <p className="text-[10px] text-slate-400 mt-2 font-mono">
                      Allocating GPU tensor blocks · Computing spectral entropy · Generating autograd kernels...
                    </p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Bottom Input Area */}
            <div className="pt-3">
              <div className="glass p-2.5 rounded-2xl border border-white/10 shadow-2xl flex items-center gap-2 bg-black/40">
                <div className="flex items-center gap-1.5 px-2">
                  <button
                    onClick={() => handlePromptSend("Execute dynamic entropy SVD rank allocation on BERT-base.")}
                    title="Quick SVD Command"
                    className="p-1.5 rounded-lg text-slate-400 hover:text-cyan-400 hover:bg-white/5 transition-all cursor-pointer"
                  >
                    <Sparkles size={16} />
                  </button>
                  <button
                    onClick={() => handlePromptSend("Generate high-performance Triton fused low-rank attention kernel.")}
                    title="Generate Triton Kernel"
                    className="p-1.5 rounded-lg text-slate-400 hover:text-yellow-400 hover:bg-white/5 transition-all cursor-pointer"
                  >
                    <Zap size={16} />
                  </button>
                </div>

                <input
                  type="text"
                  value={inputVal}
                  onChange={(e) => setInputVal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handlePromptSend(inputVal);
                    }
                  }}
                  disabled={sending || loading}
                  placeholder="Ask anything – formulate attention kernels, calculate singular entropy, dispatch GLUE runs..."
                  className="flex-1 bg-transparent text-xs text-white placeholder:text-slate-500 focus:outline-none px-2"
                />

                <button
                  onClick={() => handlePromptSend(inputVal)}
                  disabled={sending || loading || !inputVal.trim()}
                  className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold px-4 py-2 rounded-xl text-xs flex items-center gap-1.5 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Send size={14} />
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
            </div>
          </div>

          {/* Right Artifact Preview Panel */}
          {showArtifacts && (
            <div className="w-96 border-l border-white/10 bg-black/50 flex flex-col p-4 overflow-y-auto scrollbar-thin animate-fade-in">
              <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-4">
                <div className="flex items-center gap-2">
                  <FileText size={16} className="text-emerald-400" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Research Artifacts
                  </span>
                </div>
                <button
                  onClick={() => setShowArtifacts(false)}
                  className="text-xs text-slate-400 hover:text-white p-1 cursor-pointer"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3.5 text-xs">
                {/* Artifact 1 */}
                <div className="glass p-3.5 rounded-xl border border-emerald-500/20">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-emerald-400">
                      Dynamic Entropy SVD Model
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                      v2026.4
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-2">
                    Layer-adaptive singular rank allocation algorithm r_l ∝ H(Σ_l)
                  </p>
                  <pre className="text-[10px] bg-black/70 p-2.5 rounded-lg text-slate-300 font-mono overflow-x-auto border border-white/5">
{`from framework.compression_engine.dynamic_rank import DynamicEntropySVDCompressor
compressor = DynamicEntropySVDCompressor(
    target_compression_ratio=0.5,
    min_rank=4,
    max_rank=32
)`}
                  </pre>
                </div>

                {/* Artifact 2 */}
                <div className="glass p-3.5 rounded-xl border border-cyan-500/20">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-cyan-400">
                      Stiefel QR Retraction
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono">
                      Orthogonal
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-2">
                    Exact manifold projection guaranteeing ||WᵀW - I|| &lt; 1e-12
                  </p>
                  <div className="text-[11px] bg-cyan-950/30 p-2.5 rounded-lg text-cyan-200 font-mono border border-cyan-500/20">
                    W_(t+1) = Q · diag(sgn(diag(R)))
                  </div>
                </div>

                {/* Artifact 3 */}
                <div className="glass p-3.5 rounded-xl border border-purple-500/20">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-purple-400">
                      319-Page Academic Monograph
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 font-mono">
                      PDF Ready
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-3">
                    Includes full proofs, convergence lemmas, and GLUE benchmark results.
                  </p>
                  <button
                    onClick={() => navigate("/paper/editor")}
                    className="text-xs bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold px-3 py-1.5 rounded-lg shadow-md transition-all w-full text-center cursor-pointer flex items-center justify-center gap-1.5"
                  >
                    <ExternalLink size={13} /> Open LaTeX Editor & PDF
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
