import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function PresetSelector({ onApply }) {
    const presets = {
        fast: { epochs: 1, batch_size: 32, learning_rate: 5e-5, compression_rank: 32 },
        accurate: { epochs: 5, batch_size: 16, learning_rate: 2e-5, compression_rank: 64 },
        extreme_compress: { epochs: 3, batch_size: 8, learning_rate: 1e-5, compression_rank: 16 },
    };
    return (_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium", children: "Preset" }), _jsxs("select", { className: "w-full border rounded px-3 py-2", onChange: (e) => onApply(presets[e.target.value] || {}), children: [_jsx("option", { value: "", children: "None" }), _jsx("option", { value: "fast", children: "Fast" }), _jsx("option", { value: "accurate", children: "Accurate" }), _jsx("option", { value: "extreme_compress", children: "Extreme Compress" })] })] }));
}
