import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ConfigSummary({ config }) {
    return (_jsxs("div", { className: "bg-white rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-lg font-semibold mb-2", children: "Configuration" }), _jsx("pre", { className: "text-xs text-gray-700 overflow-auto max-h-64", children: JSON.stringify(config, null, 2) })] }));
}
