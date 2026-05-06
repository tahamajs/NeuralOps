import { jsx as _jsx } from "react/jsx-runtime";
const colors = {
    queued: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    stopped: 'bg-gray-100 text-gray-800',
};
export default function StatusBadge({ status }) {
    return (_jsx("span", { className: `px-2 py-1 rounded-full text-xs font-medium ${colors[status] || ''}`, children: status }));
}
