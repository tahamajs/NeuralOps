import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Plot from 'react-plotly.js';
export default function LiveLossPlot({ metrics }) {
    return (_jsxs("div", { className: "bg-white rounded-lg shadow p-4", children: [_jsx("h2", { className: "text-lg font-semibold mb-2", children: "Training Loss" }), _jsx(Plot, { data: [
                    {
                        x: metrics.loss.map((_, i) => i + 1),
                        y: metrics.loss,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Loss',
                    },
                ], layout: {
                    autosize: true,
                    margin: { t: 10, r: 10, b: 30, l: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    height: 300,
                }, useResizeHandler: true, style: { width: '100%' } })] }));
}
