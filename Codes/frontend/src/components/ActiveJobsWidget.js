import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useNavigate } from 'react-router-dom';
export default function ActiveJobsWidget({ experiments }) {
    const nav = useNavigate();
    const active = experiments.filter((e) => e.status === 'running' || e.status === 'queued');
    return (_jsxs("div", { className: "bg-white rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-lg font-semibold mb-2", children: "Active Jobs" }), active.length === 0 && _jsx("div", { className: "text-sm text-gray-500", children: "No active jobs" }), active.map((job) => (_jsxs("div", { className: "text-sm py-1 cursor-pointer hover:bg-gray-50", onClick: () => nav(`/experiments/${job.id}`), children: [_jsx("span", { className: "font-medium", children: job.name || job.id.slice(0, 8) }), _jsxs("span", { className: "text-gray-500 ml-2 capitalize", children: ["(", job.status, ")"] })] }, job.id)))] }));
}
