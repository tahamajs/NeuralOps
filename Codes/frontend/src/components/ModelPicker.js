import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
export default function ModelPicker({ value, onChange }) {
    const { data } = useQuery({ queryKey: ['config/models'], queryFn: async () => (await api.get('/config/models/')).data });
    return (_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("label", { className: "text-xs font-black text-slate-500 uppercase tracking-widest leading-none", children: "Foundation Model" }), _jsxs("select", { value: value, onChange: (e) => onChange(e.target.value), className: "bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm font-bold text-white focus:ring-2 focus:ring-blue-500/50 outline-none transition-all appearance-none cursor-pointer", children: [_jsx("option", { value: "", className: "bg-[#0a0c10]", children: "Select Model..." }), (data?.models || []).map((m) => (_jsx("option", { value: m, className: "bg-[#0a0c10]", children: m }, m)))] })] }));
}
