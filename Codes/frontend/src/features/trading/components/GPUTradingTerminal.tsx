import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '../../../api/client';
import CandlestickChart from './CandlestickChart';
import OrderBook from './OrderBook';
import PlaceOrderForm from './PlaceOrderForm';
import PortfolioTracker from './PortfolioTracker';
import { Trade } from '../types';

const GPU_MODELS = ['H100', 'A100', 'RTX 4090', 'RTX 3090'];

export default function GPUTradingTerminal() {
  const [selectedGPU, setSelectedGPU] = useState('H100');

  return (
    <div className="p-4 space-y-6">
      <div className="flex gap-2">
        {GPU_MODELS.map(model => (
          <button
            key={model}
            onClick={() => setSelectedGPU(model)}
            className={`px-4 py-2 rounded-lg font-semibold ${selectedGPU === model ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'}`}
          >
            {model}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <CandlestickChart gpuModel={selectedGPU} />
        </div>
        <div>
          <OrderBook gpuModel={selectedGPU} />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PlaceOrderForm gpuModel={selectedGPU} />
        <PortfolioTracker />
      </div>
      <div>
        <RecentTrades gpuModel={selectedGPU} />
      </div>
    </div>
  );
}

function RecentTrades({ gpuModel }: { gpuModel: string }) {
  const { data: trades } = useQuery({
    queryKey: ['trades', gpuModel],
    queryFn: () => api.get(`/trading/orders/trades/?gpu_model=${gpuModel}`).then(r => r.data),
    refetchInterval: 5000,
  });

  return (
    <div className="glass p-4">
      <h2 className="text-lg font-bold mb-2">Recent Trades</h2>
      <div className="overflow-auto max-h-64">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400 border-b border-slate-700">
              <th>Price</th><th>Qty</th><th>Time</th>
            </tr>
          </thead>
          <tbody>
            {trades?.map((trade: any, i: number) => (
              <tr key={i} className="border-b border-slate-800">
                <td className={trade.side === 'buy' ? 'text-green-400' : 'text-red-400'}>${trade.price}</td>
                <td>{trade.quantity}</td>
                <td className="text-slate-500">{new Date(trade.created_at).toLocaleTimeString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}