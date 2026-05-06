import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function PaginationControls({ page, total, onChange, }) {
    if (total <= 1)
        return null;
    return (_jsxs("div", { className: "flex justify-center gap-2", children: [_jsx("button", { disabled: page <= 1, onClick: () => onChange(page - 1), className: "px-3 py-1 border rounded disabled:opacity-50", children: "Prev" }), _jsxs("span", { className: "py-1 text-sm", children: [page, " / ", total] }), _jsx("button", { disabled: page >= total, onClick: () => onChange(page + 1), className: "px-3 py-1 border rounded disabled:opacity-50", children: "Next" })] }));
}
