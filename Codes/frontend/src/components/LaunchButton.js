import { jsx as _jsx } from "react/jsx-runtime";
export default function LaunchButton({ loading, onClick }) {
    return (_jsx("button", { disabled: loading, onClick: onClick, className: "bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 text-lg font-semibold w-full", children: loading ? 'Launching...' : '🚀 Launch Experiment' }));
}
