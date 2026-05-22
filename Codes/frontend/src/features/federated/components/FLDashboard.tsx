// src/features/federated/components/FLDashboard.tsx
import { useQuery } from '@tanstack/react-query';
import { TerminalCard } from '../../../components/ui/TerminalCard';
import { Progress } from '../../../components/ui/Progress';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

export function FLDashboard({ runId }: { runId: string }) {
  const { data: globalMetrics } = useQuery({
    queryKey: ['fl-global', runId],
    queryFn: () => fetch(`/api/federated/${runId}/global/`).then(res => res.json()),
    refetchInterval: 5000,
  });
  const { data: clients } = useQuery({
    queryKey: ['fl-clients', runId],
    queryFn: () => fetch(`/api/federated/${runId}/clients/`).then(res => res.json()),
    refetchInterval: 10000,
  });
  const globalHistory = globalMetrics?.history || [];
  const clientList = clients?.clients || [];
  const totalClients = clientList.length;
  const activeClients = clientList.filter((c: any) => c.status === 'active').length;
  const avgClientAccuracy = clientList.reduce((sum: number, c: any) => sum + c.accuracy, 0) / totalClients || 0;
  const finalAccuracy = globalMetrics?.final_accuracy || 0;

  const clientData = clientList.map((c: any) => ({ name: c.id.slice(0, 8), accuracy: c.accuracy, data: c.data_samples }));

  return (
    <div className="flex flex-col space-y-4 p-4 h-full overflow-y-auto">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <TerminalCard className="p-4 text-center"><div className="text-2xl font-bold text-neon-mint">{totalClients}</div><div className="text-xs text-slate-500">TOTAL CLIENTS</div></TerminalCard>
        <TerminalCard className="p-4 text-center"><div className="text-2xl font-bold text-green-400">{activeClients}</div><div className="text-xs text-slate-500">ACTIVE CLIENTS</div></TerminalCard>
        <TerminalCard className="p-4 text-center"><div className="text-2xl font-bold text-blue-400">{finalAccuracy.toFixed(2)}%</div><div className="text-xs text-slate-500">GLOBAL ACCURACY</div></TerminalCard>
        <TerminalCard className="p-4 text-center"><div className="text-2xl font-bold text-yellow-400">{avgClientAccuracy.toFixed(2)}%</div><div className="text-xs text-slate-500">AVG CLIENT ACCURACY</div></TerminalCard>
      </div>
      <TerminalCard className="p-4"><div className="text-sm font-bold text-white mb-2">Global Model Accuracy</div><ResponsiveContainer width="100%" height={200}><AreaChart data={globalHistory}><XAxis dataKey="round" stroke="#475569" /><YAxis stroke="#475569" /><Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} /><Area type="monotone" dataKey="accuracy" stroke="#00ff9d" fill="#00ff9d20" /></AreaChart></ResponsiveContainer></TerminalCard>
      {clientData.length > 0 && (<TerminalCard className="p-4"><div className="text-sm font-bold text-white mb-2">Client Performance</div><ResponsiveContainer width="100%" height={300}><BarChart data={clientData}><XAxis dataKey="name" stroke="#475569" /><YAxis stroke="#475569" /><Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} /><Bar dataKey="accuracy" fill="#3b82f6"><Cell fill="#00ff9d" /></Bar></BarChart></ResponsiveContainer></TerminalCard>)}
      <TerminalCard className="p-4"><div className="flex justify-between items-center"><div><div className="text-sm font-bold text-white">Current Round</div><div className="text-2xl font-mono text-neon-mint">{globalMetrics?.current_round || 0}</div></div><div><div className="text-sm font-bold text-white">Next Aggregation</div><Progress value={globalMetrics?.next_aggregation_progress || 0} className="w-32 h-1" /></div></div></TerminalCard>
    </div>
  );
}