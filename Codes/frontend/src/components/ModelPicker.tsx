import { useQuery } from "@tanstack/react-query";
import api from "../api/client";

export default function ModelPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const { data } = useQuery({
    queryKey: ["config/models"],
    queryFn: async () => (await api.get("/config/models/")).data,
  });

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-black text-slate-500 uppercase tracking-widest leading-none">
        Foundation Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm font-bold text-white focus:ring-2 focus:ring-blue-500/50 outline-none transition-all appearance-none cursor-pointer"
      >
        <option value="" className="bg-[#0a0c10]">
          Select Model...
        </option>
        {(data?.models || []).map((m: string) => (
          <option key={m} value={m} className="bg-[#0a0c10]">
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
