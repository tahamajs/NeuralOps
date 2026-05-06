import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
export default function ProgressBar({ progress }) {
    const pct = progress.total > 0 ? Math.min(100, (progress.current / progress.total) * 100) : 0;
    return (_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-sm text-gray-600 mb-1", children: [_jsxs("span", { children: [progress.current, " / ", progress.total, " steps"] }), progress.eta > 0 && _jsxs("span", { children: ["ETA: ", Math.round(progress.eta), "s"] })] }), _jsx("div", { className: "w-full bg-gray-200 rounded-full h-3", children: _jsx("div", { className: "bg-blue-600 h-3 rounded-full", style: { width: `${pct}%` } }) })] }));
}
