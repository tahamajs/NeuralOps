import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useRef } from 'react';
export default function LiveLogViewer({ logs }) {
    const ref = useRef(null);
    useEffect(() => {
        if (ref.current)
            ref.current.scrollTop = ref.current.scrollHeight;
    }, [logs]);
    return (_jsx("div", { className: "bg-gray-900 rounded-lg shadow p-4 text-green-400 font-mono text-sm h-64 overflow-y-auto", ref: ref, children: logs.map((line, i) => (_jsx("div", { children: line }, i))) }));
}
