import { useEffect, useRef } from 'react';
import { createChart, IChartApi, CandlestickData } from 'lightweight-charts';

interface CandlestickChartProps {
  data: { time: number; open: number; high: number; low: number; close: number }[];
  height?: number;
}

export function CandlestickChart({ data, height = 400 }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: { background: { color: '#0a0f1a' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
    });
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
      wickUpColor: '#10b981', wickDownColor: '#ef4444',
    });
    candlestickSeries.setData(data as CandlestickData[]);
    chart.timeScale().fitContent();
    chartRef.current = chart;
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth || 0 });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, height]);

  return <div ref={chartContainerRef} style={{ width: '100%', height }} />;
}