// NeuralOps Observable Plot Interactive Visualizations
import * as Plot from "/static/plot.min.js";

async function renderRooflineChart() {
  const container = document.getElementById("roofline-chart");
  if (!container) return;

  try {
    const res = await fetch("/api/chart/roofline");
    const data = await res.json();
    if (!data || data.length === 0) {
      container.innerHTML = '<p style="color: var(--text-muted); padding: 1rem;">No roofline data available.</p>';
      return;
    }

    // Theoretical roofline reference line points (e.g. Memory Bandwidth bound to Compute bound transition)
    const lineData = [
      { intensity: 1, gflops: 300 },
      { intensity: 16, gflops: 4800 },
      { intensity: 64, gflops: 19200 },
      { intensity: 256, gflops: 19200 },
    ];

    const plot = Plot.plot({
      title: "Kernel Operational Intensity vs. Speedup (Roofline Proxy)",
      subtitle: "Arithmetic Volume Proxy (B*S*D / 1024) vs Measured Acceleration",
      style: {
        background: "var(--bg-secondary)",
        color: "var(--text-primary)",
        fontSize: "12px",
      },
      grid: true,
      x: {
        type: "log",
        label: "Operational Intensity Proxy (FLOP / Byte proxy) →",
      },
      y: {
        type: "log",
        label: "Observed Speedup (x) →",
      },
      color: {
        legend: true,
      },
      marks: [
        Plot.ruleY([1], { stroke: "var(--border-color)", strokeDasharray: "4,4" }),
        Plot.dot(data, {
          x: "intensity",
          y: "speedup",
          fill: "backend",
          r: 5,
          title: (d) => `${d.module} (${d.backend})\nIntensity: ${d.intensity}\nSpeedup: ${d.speedup}x\nLatency: ${d.latency_ms}ms`,
        }),
      ],
      width: container.clientWidth || 600,
      height: 380,
    });

    container.innerHTML = "";
    container.appendChild(plot);
  } catch (err) {
    console.error("Failed to render Roofline chart:", err);
    container.innerHTML = `<p style="color: var(--accent-red);">Error loading roofline chart: ${err.message}</p>`;
  }
}

async function renderParetoChart() {
  const container = document.getElementById("pareto-chart");
  if (!container) return;

  try {
    const res = await fetch("/api/chart/pareto");
    const data = await res.json();
    if (!data || data.length === 0) {
      container.innerHTML = '<p style="color: var(--text-muted); padding: 1rem;">No Pareto data available.</p>';
      return;
    }

    const plot = Plot.plot({
      title: "Compression Ratio vs. Accuracy Pareto Frontier",
      subtitle: "Empirical downstream task retention across compression recipes",
      style: {
        background: "var(--bg-secondary)",
        color: "var(--text-primary)",
        fontSize: "12px",
      },
      grid: true,
      x: {
        label: "Compression Ratio (x) →",
      },
      y: {
        label: "Retained Accuracy (%) →",
      },
      marks: [
        Plot.dot(data, {
          x: "compression_ratio",
          y: "accuracy",
          fill: "var(--accent-blue)",
          r: 5,
          title: (d) => `${d.module}\nCompression Ratio: ${d.compression_ratio}x\nAccuracy: ${d.accuracy}%\nPPL: ${d.ppl}`,
        }),
      ],
      width: container.clientWidth || 600,
      height: 380,
    });

    container.innerHTML = "";
    container.appendChild(plot);
  } catch (err) {
    console.error("Failed to render Pareto chart:", err);
    container.innerHTML = `<p style="color: var(--accent-red);">Error loading Pareto chart: ${err.message}</p>`;
  }
}

// Initial render
window.addEventListener("DOMContentLoaded", () => {
  renderRooflineChart();
  renderParetoChart();
});

// Re-render on window resize
window.addEventListener("resize", () => {
  renderRooflineChart();
  renderParetoChart();
});
