export function PortfolioTracker() {
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: () => api.get('/marketplace/portfolio/').then(res => res.data),
    refetchInterval: 10000,
  });

  if (!portfolio) return <div className="glass-terminal p-4">Loading portfolio...</div>;

  return (
    <div className="glass-terminal p-4 rounded-xl">
      <h3 className="text-sm font-mono text-neon-mint mb-2">My Portfolio</h3>
      <div className="grid grid-cols-2 gap-4 text-center">
        <div>
          <div className="text-2xl font-bold text-white">${portfolio.total_value.toFixed(2)}</div>
          <div className="text-xs text-slate-400">Total Value</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-green-400">+{portfolio.total_gain.toFixed(2)}%</div>
          <div className="text-xs text-slate-400">Total Return</div>
        </div>
      </div>
      <div className="mt-3 space-y-2">
        {portfolio.holdings.map((holding: any) => (
          <div key={holding.symbol} className="flex justify-between text-sm border-t border-white/10 pt-2">
            <span>{holding.symbol}</span>
            <span>{holding.quantity} hrs</span>
            <span className="text-neon-mint">${holding.current_value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}