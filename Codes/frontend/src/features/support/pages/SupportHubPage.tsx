// src/features/support/pages/SupportHubPage.tsx
import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  supportAPI,
  SupportTicket,
  Solution,
  TicketMessage,
} from "../services/supportApi";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { useToast } from "../../../components/ui/Toast";
import {
  LifeBuoy,
  PlusCircle,
  Search,
  BookOpen,
  ThumbsUp,
  MessageSquare,
  Sparkles,
  CheckCircle,
  AlertTriangle,
  Send,
  Terminal,
  Paperclip,
  Activity,
  Cpu,
  ArrowLeft,
  Star,
} from "lucide-react";

// Mock Fallback Data in case backend is empty
const MOCK_SOLUTIONS: Solution[] = [
  {
    id: "sol-1",
    ticket: "t-1",
    ticket_title: "How to configure custom Nova Attention?",
    content:
      "Nova Attention incorporates structural pruning (ARMOR and AST). To configure it, set `attn_type: 'nova'` in your run configuration and specify the `rank` parameter (e.g. `rank: 64`). Make sure your dataset tokenizer is initialized with padding='max_length' to prevent CUDA compilation bounds errors.",
    upvotes: 42,
    created_at: new Date().toISOString(),
  },
  {
    id: "sol-2",
    ticket: "t-2",
    ticket_title: "GPU listing shows offline - how to fix?",
    content:
      "If your GPU marketplace listing shows offline, verify that the daemon agent is running on your compute node. Run `python -m gpu_marketplace.agent` and inspect `~/.neuralops/agent.log`. Ensure ports 8000 (API) and 50051 (gRPC) are open and accessible.",
    upvotes: 18,
    created_at: new Date().toISOString(),
  },
  {
    id: "sol-3",
    ticket: "t-3",
    ticket_title: "Optuna sweep trials failing with CUDA Out of Memory (OOM)",
    content:
      "Optuna sweeps launch parallel trials that can saturate VRAM. To resolve this, use `batch_runner.py` with the `--parallel 2` flag instead of 4, or implement HybridKV caching to prune key-value size dynamically. You can also specify `--max_samples 1000` to limit sequence size.",
    upvotes: 29,
    created_at: new Date().toISOString(),
  },
];

export default function SupportHubPage() {
  const [activeTab, setActiveTab] = useState<"tickets" | "faq" | "diagnostics">(
    "tickets",
  );
  const [selectedTicketId, setSelectedTicketId] = useState<string | null>(null);

  // Form states for new ticket
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [codeSnippet, setCodeSnippet] = useState("");
  const [logAttachment, setLogAttachment] = useState("");
  const [tagsInput, setTagsInput] = useState("");

  // Message input state
  const [newMessage, setNewMessage] = useState("");

  // Feedback state
  const [feedbackRating, setFeedbackRating] = useState(5);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // Search state for FAQ
  const [faqSearch, setFaqSearch] = useState("");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  // Queries
  const {
    data: rawTicketsData,
    isLoading: isLoadingTickets,
    refetch: refetchTickets,
  } = useQuery({
    queryKey: ["support-tickets"],
    queryFn: supportAPI.getTickets,
    retry: false,
  });

  const { data: rawSolutionsData, isLoading: isLoadingSolutions } = useQuery({
    queryKey: ["support-solutions"],
    queryFn: supportAPI.getSolutions,
    retry: false,
  });

  const { data: rawSelectedTicket, isLoading: isLoadingDetail } = useQuery({
    queryKey: ["support-ticket-detail", selectedTicketId],
    queryFn: () => supportAPI.getTicketDetail(selectedTicketId!),
    enabled: !!selectedTicketId,
  });

  // Mutations
  const createTicketMutation = useMutation({
    mutationFn: supportAPI.createTicket,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["support-tickets"] });
      success("Support ticket submitted successfully");
      setShowCreateForm(false);
      setTitle("");
      setDescription("");
      setCodeSnippet("");
      setLogAttachment("");
      setTagsInput("");
    },
    onError: (err: any) => {
      error(err.userMessage || "Failed to submit ticket");
    },
  });

  const sendMessageMutation = useMutation({
    mutationFn: ({
      ticketId,
      content,
    }: {
      ticketId: string;
      content: string;
    }) => supportAPI.sendTicketMessage(ticketId, content),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["support-ticket-detail", selectedTicketId],
      });
      setNewMessage("");
      success("Message sent");
    },
  });

  const submitFeedbackMutation = useMutation({
    mutationFn: ({
      ticketId,
      rating,
      comment,
    }: {
      ticketId: string;
      rating: number;
      comment?: string;
    }) => supportAPI.submitFeedback(ticketId, rating, comment),
    onSuccess: () => {
      setFeedbackSubmitted(true);
      success("Feedback submitted, thank you!");
    },
  });

  const upvoteSolutionMutation = useMutation({
    mutationFn: supportAPI.upvoteSolution,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["support-solutions"] });
      success("Solution upvoted");
    },
  });

  const ticketsData =
    rawTicketsData && "data" in rawTicketsData
      ? (rawTicketsData as any).data
      : rawTicketsData;
  const solutionsData =
    rawSolutionsData && "data" in rawSolutionsData
      ? (rawSolutionsData as any).data
      : rawSolutionsData;
  const selectedTicket =
    rawSelectedTicket && "data" in rawSelectedTicket
      ? (rawSelectedTicket as any).data
      : rawSelectedTicket;

  // Safe data retrieval (use mock data if backend empty/fails)
  const tickets: SupportTicket[] = Array.isArray(ticketsData)
    ? ticketsData
    : [];
  const solutions: Solution[] =
    Array.isArray(solutionsData) && solutionsData.length > 0
      ? solutionsData
      : MOCK_SOLUTIONS;

  const handleCreateTicketSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !description.trim()) {
      error("Title and Description are required");
      return;
    }
    const tags = tagsInput
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    createTicketMutation.mutate({
      title,
      description,
      code_snippet: codeSnippet || undefined,
      log_attachment: logAttachment || undefined,
      tags,
    });
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || !selectedTicketId) return;
    sendMessageMutation.mutate({
      ticketId: selectedTicketId,
      content: newMessage,
    });
  };

  const handleFeedbackSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedTicketId) return;
    submitFeedbackMutation.mutate({
      ticketId: selectedTicketId,
      rating: feedbackRating,
      comment: feedbackComment || undefined,
    });
  };

  const filteredSolutions = solutions.filter(
    (s) =>
      s.ticket_title.toLowerCase().includes(faqSearch.toLowerCase()) ||
      s.content.toLowerCase().includes(faqSearch.toLowerCase()),
  );

  return (
    <div className="min-h-[85vh] p-6 font-sans text-slate-200 bg-slate-950/20">
      {/* Header banner */}
      <div className="relative overflow-hidden mb-6 p-6 border bg-slate-900/40 backdrop-blur-xl border-slate-800/60 rounded-3xl shadow-2xl">
        <div className="absolute -top-24 -right-24 w-72 h-72 rounded-full blur-3xl bg-neon-blue/10 pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-72 h-72 rounded-full blur-3xl bg-purple-500/5 pointer-events-none" />
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-2">
              <LifeBuoy
                className="text-neon-blue animate-spin-slow"
                size={28}
              />{" "}
              SUPPORT <span className="text-neon-blue font-light">HUB</span>
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-xl">
              Create tickets, view AI troubleshooting runs, search verified
              platform answers, or check system clusters.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                setActiveTab("tickets");
                setShowCreateForm(true);
                setSelectedTicketId(null);
              }}
              className="px-4 py-2 text-xs font-black uppercase tracking-wider rounded-xl bg-neon-blue hover:bg-neon-blue/80 text-white flex items-center gap-1.5 transition-all shadow-glow-sm"
            >
              <PlusCircle size={14} /> Open Ticket
            </button>
          </div>
        </div>

        {/* Tab Controls */}
        <div className="flex gap-2 mt-6 border-t border-white/5 pt-4">
          {(["tickets", "faq", "diagnostics"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => {
                setActiveTab(tab);
                setShowCreateForm(false);
                setSelectedTicketId(null);
              }}
              className={`px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider transition-all border ${
                activeTab === tab
                  ? "bg-neon-blue/10 border-neon-blue/30 text-neon-blue"
                  : "bg-slate-900/20 border-transparent text-slate-400 hover:text-white"
              }`}
            >
              {tab === "tickets" && "Support Tickets"}
              {tab === "faq" && "Knowledge Base"}
              {tab === "diagnostics" && "System Diagnostics"}
            </button>
          ))}
        </div>
      </div>

      {/* Main workspace */}
      <div className="grid grid-cols-1 gap-6">
        {/* ============================================================== */}
        {/* TABS 1: SUPPORT TICKETS */}
        {/* ============================================================== */}
        {activeTab === "tickets" && (
          <div className="w-full">
            {showCreateForm ? (
              <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl max-w-3xl mx-auto">
                <div className="flex items-center gap-2 border-b border-white/5 pb-4 mb-4">
                  <PlusCircle className="text-neon-blue" size={20} />
                  <h2 className="text-lg font-bold text-white">
                    Create Support Ticket
                  </h2>
                </div>
                <form onSubmit={handleCreateTicketSubmit} className="space-y-4">
                  <Input
                    label="Subject Title"
                    placeholder="e.g. CUDA device assertion fail in Nova attention layer"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                  />
                  <div className="flex flex-col">
                    <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1">
                      Problem Description
                    </label>
                    <textarea
                      rows={4}
                      className="bg-white/5 border border-white/10 rounded-xl p-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
                      placeholder="Describe what occurred, your environment settings, etc."
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      required
                    />
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="flex flex-col">
                      <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1 flex items-center gap-1.5">
                        <Terminal size={12} className="text-neon-mint" /> Code
                        Snippet (Optional)
                      </label>
                      <textarea
                        rows={6}
                        className="bg-slate-950/80 border border-white/10 rounded-xl p-3 text-xs text-slate-300 font-mono focus:outline-none focus:border-neon-blue/50 transition-colors"
                        placeholder="Paste code or configs..."
                        value={codeSnippet}
                        onChange={(e) => setCodeSnippet(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col">
                      <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1 flex items-center gap-1.5">
                        <Paperclip size={12} className="text-neon-blue" /> Log
                        Output (Optional)
                      </label>
                      <textarea
                        rows={6}
                        className="bg-slate-950/80 border border-white/10 rounded-xl p-3 text-xs text-slate-300 font-mono focus:outline-none focus:border-neon-blue/50 transition-colors"
                        placeholder="Paste terminal log stack traces..."
                        value={logAttachment}
                        onChange={(e) => setLogAttachment(e.target.value)}
                      />
                    </div>
                  </div>
                  <Input
                    label="Tags (comma-separated)"
                    placeholder="cuda, nova-attention, optuna"
                    value={tagsInput}
                    onChange={(e) => setTagsInput(e.target.value)}
                  />
                  <div className="flex justify-end gap-2 pt-2">
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => setShowCreateForm(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      className="bg-neon-blue text-white"
                      disabled={createTicketMutation.isPending}
                    >
                      {createTicketMutation.isPending ? (
                        <Spinner size="sm" />
                      ) : (
                        "Submit Ticket"
                      )}
                    </Button>
                  </div>
                </form>
              </Card>
            ) : selectedTicketId ? (
              /* Ticket Detail / Discussion view */
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                  {/* Back button */}
                  <button
                    onClick={() => {
                      setSelectedTicketId(null);
                      setFeedbackSubmitted(false);
                      setFeedbackComment("");
                    }}
                    className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-white transition-colors"
                  >
                    <ArrowLeft size={14} /> Back to Ticket Log
                  </button>

                  {isLoadingDetail ? (
                    <div className="flex justify-center py-20">
                      <Spinner />
                    </div>
                  ) : selectedTicket ? (
                    <div className="space-y-6">
                      {/* Ticket header info */}
                      <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl relative overflow-hidden">
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <Badge variant={selectedTicket.status as any}>
                            {selectedTicket.status.replace("_", " ")}
                          </Badge>
                        </div>
                        <h2 className="text-xl font-black text-white pr-20">
                          {selectedTicket.title}
                        </h2>
                        <div className="flex items-center gap-2 text-xs text-slate-500 mt-2">
                          <span>User ID: {selectedTicket.user}</span>
                          <span>•</span>
                          <span>
                            Opened:{" "}
                            {new Date(
                              selectedTicket.created_at,
                            ).toLocaleString()}
                          </span>
                        </div>
                        <div className="mt-4 text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                          {selectedTicket.description}
                        </div>

                        {/* Tags */}
                        {selectedTicket.tags &&
                          selectedTicket.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mt-4">
                              {selectedTicket.tags.map((t) => (
                                <Badge key={t} variant="neutral">
                                  #{t}
                                </Badge>
                              ))}
                            </div>
                          )}

                        {/* Attachments */}
                        {(selectedTicket.code_snippet ||
                          selectedTicket.log_attachment) && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 border-t border-white/5 pt-4">
                            {selectedTicket.code_snippet && (
                              <div>
                                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1.5 flex items-center gap-1">
                                  <Terminal
                                    size={10}
                                    className="text-neon-mint"
                                  />{" "}
                                  Attached Code Snippet
                                </h4>
                                <pre className="p-3 bg-slate-950/80 border border-white/5 rounded-xl text-[10px] font-mono text-slate-300 overflow-x-auto max-h-48 scrollbar-thin">
                                  {selectedTicket.code_snippet}
                                </pre>
                              </div>
                            )}
                            {selectedTicket.log_attachment && (
                              <div>
                                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1.5 flex items-center gap-1">
                                  <Paperclip
                                    size={10}
                                    className="text-neon-blue"
                                  />{" "}
                                  Attached Log Output
                                </h4>
                                <pre className="p-3 bg-slate-950/80 border border-white/5 rounded-xl text-[10px] font-mono text-slate-300 overflow-x-auto max-h-48 scrollbar-thin">
                                  {selectedTicket.log_attachment}
                                </pre>
                              </div>
                            )}
                          </div>
                        )}
                      </Card>

                      {/* Chat / Messages Panel */}
                      <Card className="p-4 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl flex flex-col min-h-[400px]">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-white/5 pb-2">
                          Ticket Messages
                        </h3>
                        <div className="flex-1 overflow-y-auto space-y-3 pr-2 max-h-[350px] scrollbar-thin">
                          {selectedTicket.messages?.length === 0 ? (
                            <div className="text-center py-12 text-slate-600 italic text-xs">
                              No discussion yet. Support or AI agents will
                              respond shortly.
                            </div>
                          ) : (
                            selectedTicket.messages?.map(
                              (msg: TicketMessage) => {
                                const isUser = msg.role === "user";
                                const isAgent = msg.role === "agent";
                                return (
                                  <div
                                    key={msg.id}
                                    className={`flex flex-col max-w-[85%] p-3 rounded-2xl border ${
                                      isUser
                                        ? "bg-neon-blue/10 border-neon-blue/20 self-end text-right"
                                        : isAgent
                                          ? "bg-purple-500/10 border-purple-500/20 self-start text-left"
                                          : "bg-white/5 border-white/10 self-start text-left"
                                    }`}
                                  >
                                    <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">
                                      {msg.role === "user"
                                        ? "You"
                                        : msg.role === "agent"
                                          ? "Support Tech"
                                          : "AI Copilot"}
                                    </span>
                                    <p className="text-xs text-slate-200 leading-relaxed break-words whitespace-pre-wrap">
                                      {msg.content}
                                    </p>
                                    <span className="text-[8px] text-slate-600 mt-1 block">
                                      {new Date(
                                        msg.created_at,
                                      ).toLocaleTimeString()}
                                    </span>
                                  </div>
                                );
                              },
                            )
                          )}
                        </div>

                        {selectedTicket.status !== "closed" && (
                          <form
                            onSubmit={handleSendMessage}
                            className="mt-4 flex gap-2 border-t border-white/5 pt-3"
                          >
                            <input
                              type="text"
                              value={newMessage}
                              onChange={(e) => setNewMessage(e.target.value)}
                              placeholder="Write a message response..."
                              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
                            />
                            <Button
                              type="submit"
                              size="sm"
                              className="bg-neon-blue text-white"
                              disabled={sendMessageMutation.isPending}
                            >
                              <Send size={12} />
                            </Button>
                          </form>
                        )}
                      </Card>

                      {/* Feedback rating widget (shown when resolved or closed) */}
                      {(selectedTicket.status === "resolved" ||
                        selectedTicket.status === "closed") &&
                        !feedbackSubmitted && (
                          <Card className="p-6 bg-slate-900/40 backdrop-blur-md border border-emerald-500/20 rounded-3xl">
                            <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-1.5">
                              <CheckCircle
                                className="text-emerald-400"
                                size={16}
                              />{" "}
                              Rate Ticket Resolution
                            </h3>
                            <p className="text-xs text-slate-400 mb-4">
                              Your ticket was marked resolved. Please rate the
                              service quality below.
                            </p>
                            <form
                              onSubmit={handleFeedbackSubmit}
                              className="space-y-4"
                            >
                              <div className="flex items-center gap-3">
                                <span className="text-xs text-slate-400 uppercase font-semibold">
                                  Rating:
                                </span>
                                <div className="flex gap-1.5">
                                  {[1, 2, 3, 4, 5].map((star) => (
                                    <button
                                      key={star}
                                      type="button"
                                      onClick={() => setFeedbackRating(star)}
                                      className="transition-transform hover:scale-110"
                                    >
                                      <Star
                                        size={20}
                                        className={
                                          star <= feedbackRating
                                            ? "fill-yellow-400 text-yellow-400"
                                            : "text-slate-600"
                                        }
                                      />
                                    </button>
                                  ))}
                                </div>
                                <span className="text-sm font-bold text-yellow-400 font-mono">
                                  {feedbackRating}/5
                                </span>
                              </div>
                              <textarea
                                rows={2}
                                placeholder="Comment details (optional)"
                                value={feedbackComment}
                                onChange={(e) =>
                                  setFeedbackComment(e.target.value)
                                }
                                className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-xs text-white placeholder-slate-500 focus:outline-none"
                              />
                              <Button
                                type="submit"
                                size="sm"
                                className="bg-emerald-500 text-white"
                              >
                                Submit Rating
                              </Button>
                            </form>
                          </Card>
                        )}
                    </div>
                  ) : (
                    <div className="text-center py-20 text-slate-500">
                      Ticket not found
                    </div>
                  )}
                </div>

                {/* Right Column: AI Auto Troubleshooting steps */}
                <div className="space-y-6">
                  {selectedTicket && (
                    <Card className="p-6 bg-gradient-to-b from-purple-950/20 to-slate-900/30 backdrop-blur-md border border-purple-500/20 rounded-3xl relative overflow-hidden shadow-[0_0_30px_rgba(168,85,247,0.05)]">
                      <div className="absolute top-0 right-0 w-32 h-32 rounded-full blur-2xl bg-purple-500/5 pointer-events-none" />
                      <div className="flex items-center gap-2 mb-3 border-b border-purple-500/10 pb-2">
                        <Sparkles
                          size={16}
                          className="text-purple-400 animate-pulse"
                        />
                        <h3 className="text-xs uppercase font-extrabold tracking-widest text-purple-400">
                          AI Co-Scientist Agent
                        </h3>
                      </div>

                      {selectedTicket.ai_summary ? (
                        <div className="space-y-4">
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">
                              Automated Diagnostic
                            </span>
                            <p className="text-xs text-slate-300 leading-relaxed">
                              {selectedTicket.ai_summary}
                            </p>
                          </div>

                          {selectedTicket.ai_steps &&
                            selectedTicket.ai_steps.length > 0 && (
                              <div>
                                <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-2">
                                  Recommended Actions
                                </span>
                                <ul className="space-y-2 text-xs">
                                  {selectedTicket.ai_steps.map((step, idx) => (
                                    <li
                                      key={idx}
                                      className="flex gap-2 items-start bg-purple-500/5 border border-purple-500/10 rounded-xl p-2.5"
                                    >
                                      <span className="w-4 h-4 bg-purple-500/20 text-[10px] font-bold text-purple-400 rounded-full flex items-center justify-center shrink-0 mt-0.5">
                                        {idx + 1}
                                      </span>
                                      <span className="text-slate-300 leading-tight">
                                        {step}
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                        </div>
                      ) : (
                        <div className="space-y-4 py-4 text-center">
                          <Spinner
                            size="sm"
                            className="mx-auto text-purple-400"
                          />
                          <p className="text-xs text-slate-400 italic">
                            Analyzing log trace metrics & compiling
                            recommendations...
                          </p>
                        </div>
                      )}
                    </Card>
                  )}

                  <Card className="p-6 bg-slate-900/30 border-slate-800/50 rounded-3xl">
                    <h3 className="text-xs uppercase font-bold text-slate-400 tracking-wider mb-3">
                      Ticket Parameters
                    </h3>
                    <div className="space-y-3.5 text-xs text-slate-400">
                      <div className="flex justify-between border-b border-white/5 pb-2">
                        <span>Ticket ID</span>
                        <span className="font-mono text-[10px] text-white">
                          {selectedTicketId?.slice(0, 12)}
                        </span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-2">
                        <span>Server Cluster</span>
                        <span className="text-white">US-East-GPU-Grid</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-2">
                        <span>Assigned Agent</span>
                        <span className="text-neon-blue flex items-center gap-1">
                          <Sparkles size={12} /> AutoPilot-CoSci
                        </span>
                      </div>
                    </div>
                  </Card>
                </div>
              </div>
            ) : (
              /* Normal Ticket listing table */
              <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl">
                <div className="flex justify-between items-center mb-4 border-b border-white/5 pb-3">
                  <h3 className="font-bold text-white text-sm">
                    Active Support Tickets
                  </h3>
                  <button
                    onClick={() => setShowCreateForm(true)}
                    className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-semibold text-white border border-white/10 transition-colors"
                  >
                    Create Ticket
                  </button>
                </div>

                {isLoadingTickets ? (
                  <div className="flex justify-center py-10">
                    <Spinner />
                  </div>
                ) : tickets.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 italic text-xs">
                    You have no active support tickets. Click 'Create Ticket' to
                    open one.
                  </div>
                ) : (
                  <div className="overflow-x-auto border border-white/5 rounded-2xl">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400 font-bold uppercase tracking-wider">
                          <th className="p-3">Ticket ID</th>
                          <th className="p-3">Subject</th>
                          <th className="p-3">Status</th>
                          <th className="p-3">Tags</th>
                          <th className="p-3">Opened</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5 bg-slate-950/20 text-slate-300">
                        {tickets.map((t) => (
                          <tr
                            key={t.id}
                            onClick={() => setSelectedTicketId(t.id)}
                            className="hover:bg-white/5 cursor-pointer transition-colors"
                          >
                            <td className="p-3 font-mono text-[10px] text-neon-blue">
                              {t.id.slice(0, 8)}
                            </td>
                            <td className="p-3 font-semibold text-white max-w-xs truncate">
                              {t.title}
                            </td>
                            <td className="p-3">
                              <Badge variant={t.status as any}>
                                {t.status.replace("_", " ")}
                              </Badge>
                            </td>
                            <td className="p-3">
                              <div className="flex gap-1">
                                {t.tags.slice(0, 2).map((tag) => (
                                  <span
                                    key={tag}
                                    className="px-1.5 py-0.5 rounded bg-white/5 text-[9px] text-slate-400"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="p-3 text-[10px] text-slate-500 font-mono">
                              {new Date(t.created_at).toLocaleDateString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* TABS 2: KNOWLEDGE BASE / AI SOLUTIONS */}
        {/* ============================================================== */}
        {activeTab === "faq" && (
          <div className="space-y-6">
            <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
                <div>
                  <h3 className="font-bold text-white text-base">
                    Platform Solutions Database
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Search verified solutions, configuration guidelines, and
                    cluster answers.
                  </p>
                </div>
                <div className="relative w-full md:w-80">
                  <Search
                    size={14}
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                  />
                  <input
                    type="text"
                    value={faqSearch}
                    onChange={(e) => setFaqSearch(e.target.value)}
                    placeholder="Search errors, configurations..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
                  />
                </div>
              </div>

              {filteredSolutions.length === 0 ? (
                <div className="text-center py-12 text-slate-500 italic text-xs">
                  No matching answers found for "{faqSearch}". Try searching for
                  'nova', 'OOM', or 'offline'.
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-4">
                  {filteredSolutions.map((sol) => (
                    <Card
                      key={sol.id}
                      className="p-5 bg-slate-950/20 hover:bg-slate-900/20 border border-white/5 hover:border-white/10 transition-all rounded-2xl relative overflow-hidden"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-bold text-white text-sm flex items-center gap-2">
                          <BookOpen
                            size={15}
                            className="text-neon-blue shrink-0"
                          />
                          {sol.ticket_title}
                        </h4>
                        <button
                          onClick={() => upvoteSolutionMutation.mutate(sol.id)}
                          className="px-2.5 py-1 rounded-lg bg-neon-mint/10 hover:bg-neon-mint/20 border border-neon-mint/20 text-neon-mint font-semibold text-[10px] transition-all flex items-center gap-1.5 shadow-inner shrink-0"
                        >
                          <ThumbsUp size={11} /> {sol.upvotes} Upvotes
                        </button>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed font-sans mt-2 whitespace-pre-wrap">
                        {sol.content}
                      </p>
                      <div className="text-[10px] text-slate-500 font-mono mt-3">
                        Published:{" "}
                        {new Date(sol.created_at).toLocaleDateString()}
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </Card>
          </div>
        )}

        {/* ============================================================== */}
        {/* TABS 3: SYSTEM DIAGNOSTICS */}
        {/* ============================================================== */}
        {activeTab === "diagnostics" && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2 space-y-6">
              <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-slate-800/50 rounded-3xl">
                <h3 className="font-bold text-white text-sm mb-4 border-b border-white/5 pb-2 flex items-center gap-2">
                  <Activity
                    size={16}
                    className="text-neon-mint animate-pulse"
                  />{" "}
                  Live Status Health-Check
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center bg-white/5 p-3 rounded-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                      <div>
                        <h4 className="text-xs font-bold text-white">
                          HTTP Core REST API
                        </h4>
                        <p className="text-[10px] text-slate-500">
                          Normal operations on port 8000
                        </p>
                      </div>
                    </div>
                    <span className="font-mono text-xs text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full">
                      12ms Latency
                    </span>
                  </div>

                  <div className="flex justify-between items-center bg-white/5 p-3 rounded-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                      <div>
                        <h4 className="text-xs font-bold text-white">
                          Real-Time WebSockets
                        </h4>
                        <p className="text-[10px] text-slate-500">
                          Live logs streaming & notifications
                        </p>
                      </div>
                    </div>
                    <span className="font-mono text-xs text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full">
                      Connected
                    </span>
                  </div>

                  <div className="flex justify-between items-center bg-white/5 p-3 rounded-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                      <div>
                        <h4 className="text-xs font-bold text-white">
                          Preemptible Job Scheduler
                        </h4>
                        <p className="text-[10px] text-slate-500">
                          Priority training job dispatcher active
                        </p>
                      </div>
                    </div>
                    <span className="font-mono text-xs text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full">
                      0 Jobs Waiting
                    </span>
                  </div>

                  <div className="flex justify-between items-center bg-white/5 p-3 rounded-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
                      <div>
                        <h4 className="text-xs font-bold text-white">
                          GPU Compute Clusters
                        </h4>
                        <p className="text-[10px] text-slate-500">
                          Marketplace listings provisioning load
                        </p>
                      </div>
                    </div>
                    <span className="font-mono text-xs text-amber-400 font-bold bg-amber-500/10 border border-amber-500/20 px-2.5 py-0.5 rounded-full">
                      87% Saturation
                    </span>
                  </div>
                </div>
              </Card>
            </div>

            <div className="space-y-6">
              <Card className="p-6 bg-slate-900/30 border-slate-800/50 rounded-3xl">
                <h3 className="text-xs uppercase font-extrabold tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                  <Cpu size={14} className="text-neon-blue" /> Cluster Host
                  Specifications
                </h3>
                <div className="space-y-4 text-xs text-slate-400">
                  <div className="flex justify-between">
                    <span>Host Node OS</span>
                    <span className="text-white">Ubuntu 22.04.4 LTS</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CUDA Runtime</span>
                    <span className="text-white">v12.2</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Driver Version</span>
                    <span className="text-white">535.161.07</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total GPU Threads</span>
                    <span className="text-white">4x NVIDIA H100 PCIe</span>
                  </div>
                </div>
              </Card>

              <Card className="p-6 bg-gradient-to-r from-neon-blue/5 to-purple-500/5 border border-neon-blue/10 rounded-3xl">
                <h3 className="text-xs uppercase font-extrabold tracking-widest text-neon-blue mb-2">
                  Proactive Alert System
                </h3>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  In case of critical host failures or outages, the NeuralOps
                  controller automatically migrates active docker sweep
                  containers to healthy backup nodes.
                </p>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
