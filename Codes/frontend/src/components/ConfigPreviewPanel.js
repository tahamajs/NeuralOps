import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ConfigPreviewPanel({ config }) {
    return (_jsxs("div", { className: "bg-gray-50 rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-sm font-medium mb-1", children: "Config Preview" }), _jsx("pre", { className: "text-xs text-gray-600 overflow-auto max-h-32", children: JSON.stringify(config, null, 2) })] }));
}
