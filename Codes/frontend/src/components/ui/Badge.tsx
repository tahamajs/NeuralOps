const colors: Record<string, string> = {
  queued: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  running: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
  completed: 'bg-green-500/10 text-green-400 border-green-500/30',
  failed: 'bg-red-500/10 text-red-400 border-red-500/30',
  stopped: 'bg-gray-500/10 text-gray-400 border-gray-500/30',
};

export const Badge = ({ status }: { status: string }) => (
  <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border ${colors[status] || ''}`}>
    {status}
  </span>
);