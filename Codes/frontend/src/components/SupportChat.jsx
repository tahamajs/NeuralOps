// components/SupportChat.jsx
import { useEffect, useState, useRef } from "react";

export default function SupportChat({ ticketId, token }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(
      `ws://localhost:8000/ws/support/${ticketId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "history") {
        setMessages(data.messages.map((m, i) => ({ id: i, ...m })));
      } else if (data.type === "message") {
        setMessages((prev) => [
          ...prev,
          { id: Date.now(), role: data.role, content: data.content },
        ]);
      } else if (data.type === "stream") {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            last.content += data.content;
            return [...prev];
          } else {
            return [
              ...prev,
              {
                id: Date.now(),
                role: "assistant",
                content: data.content,
                streaming: true,
              },
            ];
          }
        });
      }
    };
    return () => ws.close();
  }, [ticketId, token]);

  const send = () => {
    if (!input.trim()) return;
    wsRef.current?.send(
      JSON.stringify({ type: "user_message", content: input }),
    );
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: "user", content: input },
    ]);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full bg-surface">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] px-4 py-2 rounded-2xl ${
                msg.role === "user"
                  ? "bg-neon-blue text-white"
                  : "bg-white/10 text-slate-200"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 border-t border-white/10 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Describe your research problem..."
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-white"
        />
        <button
          onClick={send}
          className="bg-neon-blue px-4 py-2 rounded-xl font-semibold"
        >
          Send
        </button>
      </div>
    </div>
  );
}
