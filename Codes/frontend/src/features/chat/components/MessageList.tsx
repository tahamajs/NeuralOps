import { useEffect, useRef } from "react";
import { FixedSizeList as List } from "react-window";
type Message = any;
import { Avatar } from "../../../components/ui/Avatar";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const listRef = useRef<List>(null);

  useEffect(() => {
    if (listRef.current && messages.length) {
      listRef.current.scrollToItem(messages.length - 1, "end");
    }
  }, [messages]);

  const Row = ({
    index,
    style,
  }: {
    index: number;
    style: React.CSSProperties;
  }) => {
    const msg = messages[index];
    return (
      <div
        style={style}
        className={`flex gap-3 p-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
      >
        {msg.role === "assistant" && (
          <div className="w-10 h-10 rounded-full flex items-center justify-center bg-neon-blue/20">
            <Bot size={20} />
          </div>
        )}
        <div
          className={`max-w-[80%] rounded-xl p-3 ${msg.role === "user" ? "bg-neon-blue/20 text-white" : "bg-white/5 text-slate-300"}`}
        >
          {msg.isLoading ? (
            <div className="flex gap-1">
              <span className="animate-pulse">●</span>
              <span className="animate-pulse delay-100">●</span>
              <span className="animate-pulse delay-200">●</span>
            </div>
          ) : (
            <ReactMarkdown className="prose prose-invert prose-sm max-w-none">
              {msg.content}
            </ReactMarkdown>
          )}
          <div className="text-[10px] text-slate-500 mt-1">
            {new Date(msg.createdAt).toLocaleTimeString()}
          </div>
        </div>
        {msg.role === "user" && (
          <div className="w-10 h-10 rounded-full flex items-center justify-center bg-slate-700">
            <User size={20} />
          </div>
        )}
      </div>
    );
  };

  return (
    <List
      ref={listRef}
      height={500}
      itemCount={messages.length}
      itemSize={120}
      width="100%"
    >
      {Row}
    </List>
  );
}
