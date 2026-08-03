import React from "react";
import { Button } from "../../../components/ui/Button";

interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (content: string) => void;
}

export function SuggestionChips({
  suggestions,
  onSelect,
}: SuggestionChipsProps) {
  if (!suggestions.length) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((suggestion, i) => (
        <Button
          key={i}
          variant="secondary"
          size="sm"
          onClick={() => onSelect(suggestion)}
          className="text-xs py-1 h-auto"
        >
          {suggestion}
        </Button>
      ))}
    </div>
  );
}
