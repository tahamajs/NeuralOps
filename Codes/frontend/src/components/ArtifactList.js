import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
export default function ArtifactList({ experimentId }) {
    const { data } = useQuery({
        queryKey: ['artifacts', experimentId],
        queryFn: async () => (await api.get(`/experiments/${experimentId}/artifacts/`)).data,
    });
    if (!data?.artifacts?.length)
        return null;
    return (_jsxs("div", { className: "bg-white rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-lg font-semibold mb-2", children: "Artifacts" }), _jsx("ul", { className: "text-sm space-y-1", children: data.artifacts.map((a) => (_jsx("li", { children: _jsx("a", { href: `/api/experiments/${experimentId}/artifacts/${a}`, download: true, className: "text-blue-600 hover:underline", children: a }) }, a))) })] }));
}
