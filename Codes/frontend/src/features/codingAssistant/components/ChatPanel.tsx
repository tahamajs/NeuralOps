import React, { useState, useRef, useEffect } from "react";
import { Message, CodeSuggestion } from "../types";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Send, Sparkles, AlertCircle, Terminal } from "lucide-react";

interface ChatPanelProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  isStreaming: boolean;
  streamingContent?: string;
  suggestions?: CodeSuggestion[];
  onApplySuggestion?: (suggestion: CodeSuggestion) => void;
  isConnected: boolean;
}

export function ChatPanel({
  messages = [],
  onSendMessage,
  isStreaming,
  streamingContent = "",
  suggestions = [],
  onApplySuggestion,
  isConnected,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/20 border border-white/5 rounded-2xl overflow-hidden backdrop-blur-md text-slate-200">
      {/* Header status */}
      <div className="flex items-center justify-between p-3.5 border-b border-white/10 bg-slate-950/20 text-xs">
        <span className="font-bold flex items-center gap-1.5 text-slate-300">
          <Terminal size={12} className="text-neon-mint" />
          Copilot Console
        </span>
        <span className="flex items-center gap-1">
          <span
            className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-400" : "bg-rose-400 animate-pulse"}`}
          />
          <span className="text-[10px] text-slate-500 font-mono">
            {isConnected ? "ONLINE" : "OFFLINE"}
          </span>
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 font-sans text-xs">
        {messages.length === 0 && !streamingContent && (
          <div className="text-center py-12 text-slate-500 italic max-w-xs mx-auto">
            Ask Copilot to write model configurations, custom attention layers,
            or run sweeps.
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col max-w-[85%] rounded-2xl p-3 ${
              msg.role === "user"
                ? "bg-neon-blue/10 border border-neon-blue/20 self-end text-right"
                : "bg-white/5 border border-white/10 self-start text-left"
            }`}
          >
            <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1 block">
              {msg.role === "user" ? "You" : "Copilot"}
            </span>
            <p className="text-slate-200 leading-relaxed break-words whitespace-pre-wrap">
              {msg.content}
            </p>
          </div>
        ))}

        {streamingContent && (
          <div className="bg-white/5 border border-white/10 rounded-2xl p-3 self-start max-w-[85%] text-left">
            <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1 block">
              Copilot (writing...)
            </span>
            <p className="text-slate-200 leading-relaxed break-words whitespace-pre-wrap font-mono">
              {streamingContent}
              <span className="inline-block w-1.5 h-3 bg-neon-blue ml-0.5 animate-pulse" />
            </p>
          </div>
        )}

        <div ref={scrollRef} />
      </div>

      {/* Suggestion Chips */}
      {suggestions.length > 0 && (
        <div className="px-4 py-2 border-t border-white/5 bg-slate-900/10 flex flex-wrap gap-2 shrink-0 max-h-24 overflow-y-auto">
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => onApplySuggestion?.(s)}
              className="px-2.5 py-1 rounded-lg bg-neon-blue/10 hover:bg-neon-blue/20 border border-neon-blue/20 text-neon-blue font-semibold text-[10px] transition-all flex items-center gap-1.5 shadow-inner"
            >
              <Sparkles size={10} /> Apply Diff: {s.title}
            </button>
          ))}
        </div>
      )}

      {/* Form Input */}
      <form
        onSubmit={handleSend}
        className="p-3 border-t border-white/10 bg-slate-950/40 shrink-0 flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or request a change..."
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
        />
        <Button
          type="submit"
          size="sm"
          className="bg-neon-blue hover:bg-neon-blue/80 text-white shrink-0"
        >
          <Send size={12} />
        </Button>
      </form>
    </div>
  );
}
