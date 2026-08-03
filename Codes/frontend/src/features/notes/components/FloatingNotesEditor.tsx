import { useState } from "react";
import { Button } from "../../../components/ui/Button";
import {
  Save,
  Paperclip,
  CheckSquare,
  Trash,
  Search,
  Bold,
  Italic,
  List,
} from "lucide-react";

export default function FloatingNotesEditor() {
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("Untitled Note");

  return (
    <div className="flex flex-col h-full bg-surface-elevated rounded-b-xl overflow-hidden animate-fade-in text-slate-200">
      {/* Editor Toolbar */}
      <div className="flex items-center justify-between p-2 border-b border-white/5 bg-black/20">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <Bold size={14} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <Italic size={14} />
          </Button>
          <div className="w-px h-4 bg-white/10 mx-1"></div>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <List size={14} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <CheckSquare size={14} />
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="h-8 text-xs flex gap-1">
            <Paperclip size={14} /> Attach
          </Button>
          <Button
            size="sm"
            className="h-8 text-xs bg-neon-blue/20 text-neon-blue hover:bg-neon-blue/30 border border-neon-blue/20"
          >
            <Save size={14} className="mr-1" /> Save
          </Button>
        </div>
      </div>

      {/* Editor Content area */}
      <div className="flex-1 flex flex-col p-4">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Note Title..."
          className="bg-transparent border-none outline-none text-xl font-bold text-white mb-4 placeholder-slate-600 focus:ring-0"
        />
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Start typing your research notes, findings, or code snippets here..."
          className="flex-1 bg-transparent border-none outline-none resize-none text-sm leading-relaxed text-slate-300 placeholder-slate-600 font-mono-custom custom-scrollbar focus:ring-0"
        />
      </div>

      {/* Status bar */}
      <div className="p-2 bg-black/40 border-t border-white/5 flex items-center justify-between text-xs text-slate-500">
        <span>{content.length} characters</span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>{" "}
          Auto-saving enabled
        </span>
      </div>
    </div>
  );
}
