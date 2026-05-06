import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function DownloadModelButton({ experimentId }) {
    return (_jsxs("div", { className: "bg-white rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-lg font-semibold mb-2", children: "Model" }), _jsx("a", { href: `/api/experiments/${experimentId}/model/`, download: true, className: "inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700", children: "Download Model (ZIP)" })] }));
}
