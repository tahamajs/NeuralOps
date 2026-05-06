import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
export default function MethodPicker({ value, onChange }) {
    const { data } = useQuery({ queryKey: ['config/methods'], queryFn: async () => (await api.get('/config/methods/')).data });
    return (_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium", children: "Method" }), _jsx("select", { value: value, onChange: (e) => onChange(e.target.value), className: "w-full border rounded px-3 py-2", children: (data?.methods || []).map((m) => (_jsx("option", { value: m, children: m }, m))) })] }));
}
