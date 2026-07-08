import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { cn } from "../../../lib/utils";

export const AttentionPicker = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) => {
  const { data } = useQuery({
    queryKey: ["config/attentions"],
    queryFn: async () => (await api.get(ENDPOINTS.CONFIG.ATTENTIONS)).data,
  });
  const attentions: string[] = data?.attentions || [];

  return (
    <div className="space-y-2">
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">
        Attention Mechanism
      </label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-40 overflow-y-auto scrollbar-thin">
        {attentions.map((a) => (
          <button
            key={a}
            onClick={() => onChange(a)}
            className={cn(
              "px-3 py-2 rounded-xl text-xs font-bold uppercase tracking-wide border transition-all",
              value === a
                ? "bg-neon-blue/20 border-neon-blue/50 text-white shadow-neon-blue/10"
                : "bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-white",
            )}
          >
            {a.replace(/_/g, " ")}
          </button>
        ))}
      </div>
    </div>
  );
};
