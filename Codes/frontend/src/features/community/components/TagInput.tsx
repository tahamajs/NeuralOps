import { useState } from "react";
import { X } from "lucide-react";
import { Input } from "../../../components/ui/Input";

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  maxTags?: number;
  placeholder?: string;
}

export function TagInput({
  tags,
  onChange,
  maxTags = 5,
  placeholder = "Add a tag...",
}: TagInputProps) {
  const [input, setInput] = useState("");

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && input.trim() && tags.length < maxTags) {
      e.preventDefault();
      if (!tags.includes(input.trim().toLowerCase())) {
        onChange([...tags, input.trim().toLowerCase()]);
      }
      setInput("");
    } else if (e.key === "Backspace" && !input && tags.length) {
      onChange(tags.slice(0, -1));
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(tags.filter((tag) => tag !== tagToRemove));
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-neon-blue/10 text-neon-blue text-xs"
          >
            {tag}
            <button
              type="button"
              onClick={() => removeTag(tag)}
              className="hover:text-white"
            >
              <X size={12} />
            </button>
          </span>
        ))}
      </div>
      <Input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={
          tags.length >= maxTags
            ? `Maximum ${maxTags} tags reached`
            : placeholder
        }
        disabled={tags.length >= maxTags}
      />
    </div>
  );
}
