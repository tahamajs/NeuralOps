import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import api from '../api/client';
export default function SystemResourcesWidget() {
    const [sys, setSys] = useState(null);
    const [gpu, setGpu] = useState(null);
    useEffect(() => {
        const fetch = async () => {
            try {
                const [sysRes, gpuRes] = await Promise.all([
                    api.get('/system/status/'),
                    api.get('/system/gpu/')
                ]);
                setSys(sysRes.data);
                setGpu(gpuRes.data);
            }
            catch { }
        };
        fetch();
        const interval = setInterval(fetch, 3000);
        return () => clearInterval(interval);
    }, []);
    if (!sys)
        return null;
    return (_jsxs("div", { className: "glass-card", children: [_jsxs("div", { className: "flex items-center justify-between mb-6", children: [_jsxs("h2", { className: "text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] flex items-center gap-2", children: [_jsx("div", { className: "w-1 h-1 rounded-full bg-blue-500" }), "Hardware Telemetry"] }), _jsx("span", { className: "text-[10px] font-mono text-blue-400", children: "REFRESH_3000MS" })] }), _jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-bold", children: [_jsx("span", { className: "text-slate-400", children: "vCPU LOAD" }), _jsxs("span", { className: "text-white", children: [sys.cpu?.percent, "%"] })] }), _jsx("div", { className: "w-full h-1.5 bg-white/5 rounded-full overflow-hidden", children: _jsx("div", { className: "h-full bg-blue-500 transition-all duration-500", style: { width: `${sys.cpu?.percent}%` } }) })] }), _jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-bold", children: [_jsx("span", { className: "text-slate-400", children: "MEMORY_ALLOC" }), _jsxs("span", { className: "text-white", children: [sys.memory?.used_gb, " / ", sys.memory?.total_gb, " GB"] })] }), _jsx("div", { className: "w-full h-1.5 bg-white/5 rounded-full overflow-hidden", children: _jsx("div", { className: "h-full bg-indigo-500 transition-all duration-500", style: { width: `${sys.memory?.percent}%` } }) })] }), gpu && gpu.type !== 'cpu' && (_jsxs("div", { className: "pt-4 border-t border-white/5 space-y-4", children: [gpu.devices?.map((dev, i) => (_jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-bold", children: [_jsxs("span", { className: "text-slate-500", children: ["GPU_", i, " (", dev.name.split(' ')[0], ")"] }), _jsxs("span", { className: "text-blue-400", children: [dev.utilization_percent, "%"] })] }), _jsxs("div", { className: "flex justify-between text-[10px] font-mono text-slate-600", children: [_jsx("span", { children: "VRAM" }), _jsxs("span", { children: [dev.memory_allocated_mb, " / ", dev.memory_total_mb, " MB"] })] })] }, i))), gpu.type === 'mps' && (_jsxs("div", { className: "text-xs font-bold flex justify-between", children: [_jsx("span", { className: "text-slate-500", children: "METAL CORE (MPS)" }), _jsx("span", { className: "text-green-400", children: "ACTIVE" })] }))] }))] })] }));
}
