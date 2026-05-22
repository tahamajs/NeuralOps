export function GPUTradingTerminal() {
  const { listingId } = useParams();
  const [selectedView, setSelectedView] = useState<'trading' | 'portfolio'>('trading');

  return (
    <div className="flex h-full">
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="mb-4 flex gap-2">
          <button onClick={() => setSelectedView('trading')} className={`px-4 py-2 rounded ${selectedView === 'trading' ? 'bg-neon-mint/20 text-white' : 'bg-white/5'}`}>Trading View</button>
          <button onClick={() => setSelectedView('portfolio')} className={`px-4 py-2 rounded ${selectedView === 'portfolio' ? 'bg-neon-mint/20 text-white' : 'bg-white/5'}`}>Portfolio</button>
        </div>
        {selectedView === 'trading' ? (
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2 space-y-4">
              <GPUChart listingId={listingId} />
              <OrderBook listingId={listingId} />
            </div>
            <div><PlaceOrderForm listingId={listingId} /></div>
          </div>
        ) : (
          <PortfolioTracker />
        )}
      </div>
    </div>
  );
}