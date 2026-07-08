import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Users, Lightbulb } from "lucide-react";

export function EmptyState({ type }: { type: "suggestions" | "history" }) {
  return (
    <TerminalCard className="p-12 text-center">
      <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
        {type === "suggestions" ? (
          <Users size={32} className="text-slate-500" />
        ) : (
          <Lightbulb size={32} className="text-slate-500" />
        )}
      </div>
      <h3 className="text-lg font-medium text-white mb-1">
        {type === "suggestions"
          ? "No suggestions yet"
          : "No collaboration history"}
      </h3>
      <p className="text-slate-400 text-sm">
        {type === "suggestions"
          ? "Complete your research profile to get personalized recommendations."
          : "Start collaborating by accepting a suggestion or inviting a colleague."}
      </p>
    </TerminalCard>
  );
}
