import React, { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, Wallet, Download } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';

export default function Earnings() {
  const [balance, setBalance] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [monthlyData, setMonthlyData] = useState([]);

  useEffect(() => {
    api.getBalance().then(setBalance);
    api.getTransactions({ tx_type: 'release' }).then(d => setTransactions(d.results || []));
    api.getInvoices().then(inv => {
      // Build monthly data
      const byMonth = {};
      inv.forEach(i => {
        const m = i.period_start?.slice(0,7);
        if (m) byMonth[m] = (byMonth[m]||0) + parseFloat(i.total_usd||0);
      });
      setMonthlyData(Object.entries(byMonth).map(([month, earnings]) => ({ month, earnings })));
    });
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Earnings</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { icon: Wallet, label: 'Available Balance', value: `$${balance?.balance_usd || '0.00'}` },
          { icon: TrendingUp, label: 'Locked (in Escrow)', value: `$${balance?.locked_balance_usd || '0.00'}` },
          { icon: DollarSign, label: 'Lifetime Earnings', value: `$${balance?.total_earned || '0.00'}` },
        ].map(({ icon: Icon, label, value }) => (
          <div key={label} className="bg-dark-card border border-dark-border rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3"><Icon size={18} className="text-primary-400" /><span className="text-gray-400 text-sm">{label}</span></div>
            <p className="text-3xl font-bold">{value}</p>
          </div>
        ))}
      </div>

      <div className="bg-dark-card border border-dark-border rounded-xl p-5">
        <h3 className="text-lg font-semibold mb-4">Monthly Earnings</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={monthlyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="month" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
            <Bar dataKey="earnings" fill="#5c7cfa" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-dark-card border border-dark-border rounded-xl p-5">
        <h3 className="text-lg font-semibold mb-4">Recent Transactions</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {transactions.length === 0 && <p className="text-gray-500">No transactions yet</p>}
          {transactions.map(tx => (
            <div key={tx.id} className="flex justify-between items-center py-2 border-b border-dark-border">
              <div>
                <p className="text-sm">{tx.description || tx.tx_type}</p>
                <p className="text-xs text-gray-500">{new Date(tx.created_at).toLocaleDateString()}</p>
              </div>
              <span className={`font-semibold ${tx.amount_usd > 0 ? 'text-green-400' : 'text-red-400'}`}>
                {tx.amount_usd > 0 ? '+' : ''}{tx.amount_usd}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}