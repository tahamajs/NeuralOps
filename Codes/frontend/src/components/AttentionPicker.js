import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
export default function AttentionPicker({ value, onChange }) {
    const { data } = useQuery({ queryKey: ['config/attentions'], queryFn: async () => (await api.get('/config/attentions/')).data });
    return (_jsxs("div", { className: "flex flex-col gap-3", children: [_jsx("label", { className: "text-xs font-black text-slate-500 uppercase tracking-[0.2em]", children: "Attention Mechanism" }), _jsx("div", { className: "grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 max-h-[250px] overflow-y-auto scroll-thin pr-2", children: (data?.attentions || []).map((a) => (_jsx("button", { type: "button", onClick: () => onChange(a), className: `px-3 py-2 rounded-xl text-[10px] font-black uppercase tracking-tighter border transition-all ${value === a
                        ? 'bg-blue-600 border-blue-400 text-white shadow-lg shadow-blue-500/20 scale-[1.02]'
                        : 'bg-white/5 border-white/10 text-slate-400 hover:border-white/20 hover:bg-white/[0.08]'}`, children: a.replace(/_/g, ' ') }, a))) })] }));
}
