import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function MetricSummaryCards({ experiments }) {
    const counts = {
        queued: experiments.filter((e) => e.status === 'queued').length,
        running: experiments.filter((e) => e.status === 'running').length,
        completed: experiments.filter((e) => e.status === 'completed').length,
        failed: experiments.filter((e) => e.status === 'failed').length,
    };
    return (_jsx("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: Object.entries(counts).map(([key, val]) => (_jsxs("div", { className: "bg-white rounded-lg shadow p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold", children: val }), _jsx("div", { className: "text-sm text-gray-500 capitalize", children: key })] }, key))) }));
}
