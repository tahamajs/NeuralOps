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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, loading, sending, wsStatus, sendMessage } = useAgentChat(sessionId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleNewChat = async () => {
    const res = await api.post("/agents/chats/", { title: "" });
    navigate(`/agent/${res.data.id}`);
  };

  const handleSelectSession = (id: string) => navigate(`/agent/${id}`);
  const handleDeleteSession = async (id: string) => {
    await api.delete(`/agents/chats/${id}/`);
    queryClient.invalidateQueries({ queryKey: ["agentSessions"] });
    if (id === sessionId) navigate("/agent");
  };

  const quickPrompts = [
    "Search papers about attention compression",
    "Configure BERT on MRPC with LoRA",
    "Analyze my latest training run",
    "Show citation graph for paper...",
    "Suggest next experiment",
    "Edit my LaTeX paper",
  ];

  return (
    <div className="h-[calc(100vh-120px)] flex gap-0 max-w-6xl mx-auto">
      {/* Mobile sidebar toggle */}
      <button
        className="lg:hidden fixed bottom-4 left-4 z-50 glass p-3 rounded-full shadow-lg"
        onClick={() => setShowSidebar(!showSidebar)}
      >
        <Bot size={20} />
      </button>

      {/* Sidebar */}
      <div
        className={`${showSidebar ? "block" : "hidden"} lg:block fixed lg:relative z-40 h-full`}
      >
        <SessionSidebar
          activeId={sessionId}
          onSelect={handleSelectSession}
          onNew={handleNewChat}
          onDelete={handleDeleteSession}
        />
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="glass px-4 py-2 mb-4 flex items-center gap-2 flex-wrap">
          <Bot size={18} className="text-neon-blue" />
          <h1 className="text-sm font-bold text-white mr-2">
            ML Research Assistant
          </h1>
          <AgentStatusBadge status={wsStatus} />
          <div className="flex-1" />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowPaperSearch(true)}
          >
            <Search size={14} className="mr-1" /> Search Papers
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/agent/papers")}
          >
            <BookOpen size={14} className="mr-1" /> Library
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowCitationGraph(true)}
          >
            <GitBranch size={14} className="mr-1" /> Citation Graph
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/agent/tools")}
          >
            <Wrench size={14} className="mr-1" /> Tools
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/agent/prompts")}
          >
            <FlaskConical size={14} className="mr-1" /> Prompts
          </Button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 space-y-1">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-slate-500">
              <Bot size={48} className="mb-4 text-slate-600" />
              <p className="text-sm">
                How can I help with your research today?
              </p>
              <div className="grid grid-cols-2 gap-2 mt-6 max-w-lg">
                {quickPrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(prompt)}
                    className="text-left text-xs glass-hover px-3 py-2 rounded-xl border border-white/5 text-slate-400 hover:text-white hover:border-white/10 transition-all"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <ChatMessageBubble key={msg.id} msg={msg} />
          ))}
          {(loading || sending) && (
            <div className="flex gap-2 mb-3">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center">
                <Bot size={14} className="text-white" />
              </div>
              <div className="glass rounded-2xl px-4 py-2.5">
                <div className="flex gap-1">
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0s" }}
                  />
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  />
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <ChatInput onSend={sendMessage} disabled={sending || loading} />
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
