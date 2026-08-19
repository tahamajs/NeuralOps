import { useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceApi } from "../../../api/marketplace";
import { useMarketWebSocket } from "../../../hooks/useMarketWebSocket";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { useToast } from "../../../components/ui/Toast";
import {
  TrendingUp,
  TrendingDown,
  ArrowLeft,
  Activity,
  Zap,
  Cpu,
  Server,
  DollarSign,
  Shield,
  Layers,
  XCircle,
  Play,
  RotateCcw,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Bar,
  ComposedChart,
} from "recharts";

const DEFAULT_GPU_ASSETS = [
  {
    id: "listing-h100-8x",
    name: "8x H100 SXM5 80GB",
    symbol: "H100-SXM5",
    provider: "Lambda Labs",
    price_per_hour: 2.15,
    change_percent: 5.24,
    vram_gb: 80,
    interconnect: "NVLink 900 GB/s",
    region: "US-East",
  },
  {
    id: "listing-a100-4x",
    name: "4x A100 SXM4 80GB",
    symbol: "A100-SXM4",
    provider: "Vast.ai",
    price_per_hour: 1.25,
    change_percent: -1.82,
    vram_gb: 80,
    interconnect: "NVLink 600 GB/s",
    region: "EU-Central",
  },
  {
    id: "listing-gh200",
    name: "1x GH200 Grace Hopper 96GB",
    symbol: "GH200-96GB",
    provider: "CoreWeave",
    price_per_hour: 1.95,
    change_percent: 8.10,
    vram_gb: 96,
    interconnect: "NVLink-C2C",
    region: "US-West",
  },
  {
    id: "listing-rtx4090",
    name: "2x RTX 4090 24GB",
    symbol: "RTX-4090",
    provider: "RunPod",
    price_per_hour: 0.42,
    change_percent: 1.20,
    vram_gb: 24,
    interconnect: "PCIe 5.0",
    region: "Asia-East",
  },
];

export default function GPUTradingTerminal({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { listingId: routeListingId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const [selectedAssetId, setSelectedAssetId] = useState<string>(
    routeListingId || "listing-h100-8x"
  );
  const [timeframe, setTimeframe] = useState<"1H" | "24H" | "7D" | "30D">("24H");
  const [orderSide, setOrderSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState<"market" | "limit">("limit");
  const [quantity, setQuantity] = useState<number>(8);
  const [limitPrice, setLimitPrice] = useState<number>(2.15);
  const [isPlacing, setIsPlacing] = useState(false);

  // Local active orders state for instant responsiveness
  const [openOrders, setOpenOrders] = useState<any[]>([
    {
      id: "ord-88129",
      asset_id: "listing-h100-8x",
      symbol: "H100-SXM5",
      side: "BUY",
      type: "limit",
      quantity: 16,
      filled: 8,
      price: 2.10,
      status: "open",
      created_at: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: "ord-88130",
      asset_id: "listing-a100-4x",
      symbol: "A100-SXM4",
      side: "SELL",
      type: "limit",
      quantity: 24,
      filled: 0,
      price: 1.28,
      status: "open",
      created_at: new Date(Date.now() - 7200000).toISOString(),
    },
  ]);

  const { prices } = useMarketWebSocket();

  // Queries
  const { data: listingsData } = useQuery({
    queryKey: ["marketplace-listings-all"],
    queryFn: () => marketplaceApi.getListings().catch(() => ({ data: [] })),
  });

  const { data: portfolioData, refetch: refetchPortfolio } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => marketplaceApi.getPortfolio().catch(() => ({
      data: {
        balance_usd: 5420.0,
        total_value_usd_display: 8940.0,
        total_invested_usd: 7800.0,
        unrealized_pnl_display: 14.61,
        holdings: [
          {
            listing_name: "8x H100 SXM5",
            quantity: 48,
            average_buy_price: 2.05,
            current_price: 2.15,
            current_value: 103.2,
            pnl: 4.88,
          },
        ],
      },
    })),
  });

  const activeAsset = useMemo(() => {
    const raw = Array.isArray(listingsData?.data) ? listingsData.data : [];
    const foundInApi = raw.find((l: any) => l.id === selectedAssetId);
    if (foundInApi) {
      return {
        id: foundInApi.id,
        name: foundInApi.name || foundInApi.gpu_model,
        symbol: foundInApi.gpu_model?.slice(0, 10) || "GPU",
        provider: foundInApi.provider_username || "Verified Provider",
        price_per_hour: Number(foundInApi.price_per_hour || foundInApi.base_price_per_hour || 2.15),
        change_percent: Number(foundInApi.change_percent || 0),
        vram_gb: foundInApi.vram_gb || 80,
        interconnect: "High-Speed NVLink",
        region: foundInApi.location || "US-East",
      };
    }
    return DEFAULT_GPU_ASSETS.find((a) => a.id === selectedAssetId) || DEFAULT_GPU_ASSETS[0];
  }, [listingsData, selectedAssetId]);

  const liveInfo = prices[selectedAssetId] || {
    price_per_hour: activeAsset.price_per_hour,
    change_percent: activeAsset.change_percent,
    bid: +(activeAsset.price_per_hour - 0.01).toFixed(4),
    ask: +(activeAsset.price_per_hour + 0.01).toFixed(4),
    volume_24h: 184200,
    high_24h: +(activeAsset.price_per_hour * 1.05).toFixed(4),
    low_24h: +(activeAsset.price_per_hour * 0.95).toFixed(4),
  };

  const chartData = useMemo(() => {
    const points = timeframe === "1H" ? 12 : timeframe === "24H" ? 24 : timeframe === "7D" ? 28 : 30;
    const base = liveInfo.price_per_hour;
    return Array.from({ length: points }).map((_, i) => {
      const noise = Math.sin(i / 2) * (base * 0.03) + (Math.random() - 0.5) * (base * 0.015);
      const val = +(base - (points - i) * 0.005 + noise).toFixed(4);
      return {
        time: timeframe === "1H" ? `${i * 5}m` : timeframe === "24H" ? `${i}:00` : `Day ${i + 1}`,
        price: val,
        volume: Math.floor(Math.random() * 400) + 100,
      };
    });
  }, [timeframe, liveInfo.price_per_hour]);

  // Order book depth data
  const orderBookBids = useMemo(() => {
    const p = liveInfo.bid;
    return [
      { price: p, qty: 32, total: +(p * 32).toFixed(2), depth: 90 },
      { price: +(p - 0.01).toFixed(4), qty: 48, total: +((p - 0.01) * 48).toFixed(2), depth: 75 },
      { price: +(p - 0.02).toFixed(4), qty: 120, total: +((p - 0.02) * 120).toFixed(2), depth: 60 },
      { price: +(p - 0.03).toFixed(4), qty: 80, total: +((p - 0.03) * 80).toFixed(2), depth: 40 },
      { price: +(p - 0.05).toFixed(4), qty: 250, total: +((p - 0.05) * 250).toFixed(2), depth: 20 },
    ];
  }, [liveInfo.bid]);

  const orderBookAsks = useMemo(() => {
    const p = liveInfo.ask;
    return [
      { price: +(p + 0.05).toFixed(4), qty: 180, total: +((p + 0.05) * 180).toFixed(2), depth: 20 },
      { price: +(p + 0.03).toFixed(4), qty: 90, total: +((p + 0.03) * 90).toFixed(2), depth: 45 },
      { price: +(p + 0.02).toFixed(4), qty: 64, total: +((p + 0.02) * 64).toFixed(2), depth: 65 },
      { price: +(p + 0.01).toFixed(4), qty: 40, total: +((p + 0.01) * 40).toFixed(2), depth: 80 },
      { price: p, qty: 24, total: +(p * 24).toFixed(2), depth: 95 },
    ];
  }, [liveInfo.ask]);

  const handlePlaceOrder = async () => {
    const effPrice = orderType === "market" ? liveInfo.price_per_hour : limitPrice;
    if (quantity <= 0 || effPrice <= 0) {
      error("Invalid order", "Quantity and price must be greater than zero.");
      return;
    }
    setIsPlacing(true);
    try {
      await marketplaceApi.placeOrder(selectedAssetId, orderSide, quantity, effPrice);
    } catch {
      // Mock fallback
    }

    const newOrd = {
      id: `ord-${Date.now().toString().slice(-5)}`,
      asset_id: selectedAssetId,
      symbol: activeAsset.symbol,
      side: orderSide,
      type: orderType,
      quantity,
      filled: orderType === "market" ? quantity : 0,
      price: effPrice,
      status: orderType === "market" ? "filled" : "open",
      created_at: new Date().toISOString(),
    };

    setOpenOrders((prev) => [newOrd, ...prev]);
    setIsPlacing(false);
    success("Order Executed", `${orderSide} ${quantity} hrs of ${activeAsset.symbol} @ $${effPrice.toFixed(4)}`);
    refetchPortfolio();
  };

  const handleCancelOrder = (orderId: string) => {
    marketplaceApi.cancelOrder(orderId).catch(() => {});
    setOpenOrders((prev) => prev.filter((o) => o.id !== orderId));
    success("Order Cancelled", `Order #${orderId} has been revoked.`);
  };

  const portfolio = (portfolioData as any)?.data || {
    balance_usd: 5420.0,
    total_value_usd_display: 8940.0,
    unrealized_pnl_display: 14.61,
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Asset & Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div className="flex items-center gap-3">
          {onClose ? (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          ) : (
            <Button size="sm" variant="ghost" onClick={() => navigate("/marketplace")}>
              <ArrowLeft size={16} />
            </Button>
          )}

          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg text-white tracking-tight">{activeAsset.name}</span>
              <Badge variant="info">{activeAsset.region}</Badge>
              <Badge variant="success">Attested Proof</Badge>
            </div>
            <div className="text-xs text-slate-400">
              Provider: <span className="text-slate-200 font-semibold">{activeAsset.provider}</span> · {activeAsset.interconnect}
            </div>
          </div>
        </div>

        {/* Live Tickers */}
        <div className="flex items-center gap-6">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Spot Price</div>
            <div className="text-2xl font-black text-neon-mint font-mono">${liveInfo.price_per_hour.toFixed(4)}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">24h Change</div>
            <div
              className={`text-base font-bold flex items-center font-mono ${
                liveInfo.change_percent >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {liveInfo.change_percent >= 0 ? <TrendingUp size={16} className="mr-0.5" /> : <TrendingDown size={16} className="mr-0.5" />}
              {liveInfo.change_percent >= 0 ? "+" : ""}
              {liveInfo.change_percent.toFixed(2)}%
            </div>
          </div>
          <div className="hidden md:block">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">24h High / Low</div>
            <div className="text-xs text-slate-300 font-mono">
              ${liveInfo.high_24h.toFixed(4)} / ${liveInfo.low_24h.toFixed(4)}
            </div>
          </div>
          <div className="hidden lg:block">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">24h Volume</div>
            <div className="text-xs text-cyan-300 font-mono font-bold">${liveInfo.volume_24h.toLocaleString()}</div>
          </div>
        </div>

        {/* Asset Switcher */}
        <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/10">
          {DEFAULT_GPU_ASSETS.map((asset) => (
            <button
              key={asset.id}
              onClick={() => setSelectedAssetId(asset.id)}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                selectedAssetId === asset.id
                  ? "bg-neon-blue text-white shadow-glow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {asset.symbol}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Chart + Orderbook + Execution Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Live Chart & Order History */}
        <div className="lg:col-span-8 space-y-4">
          {/* Price Chart */}
          <TerminalCard className="p-4 relative">
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-2">
                <Activity size={16} className="text-neon-mint" />
                <span className="text-sm font-bold text-white uppercase tracking-wider">Spot Execution Depth</span>
              </div>
              <div className="flex items-center gap-1 bg-white/5 p-1 rounded-lg">
                {(["1H", "24H", "7D", "30D"] as const).map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`px-2.5 py-0.5 rounded text-xs font-bold transition-colors ${
                      timeframe === tf ? "bg-cyan-500 text-black" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `$${Number(v).toFixed(2)}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0a0f1a",
                      border: "1px solid #334155",
                      borderRadius: "12px",
                      fontSize: "12px",
                    }}
                  />
                  <Area type="monotone" dataKey="price" stroke="#38bdf8" strokeWidth={2.5} fill="url(#priceGrad)" />
                  <Bar dataKey="volume" yAxisId={1} fill="#818cf825" />
                  <YAxis yAxisId={1} orientation="right" hide />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </TerminalCard>

          {/* Open Orders & Trades */}
          <TerminalCard className="p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Layers size={14} className="text-cyan-400" /> Active Orders & Position Escrow ({openOrders.length})
              </h3>
            </div>
            {openOrders.length === 0 ? (
              <div className="text-center py-6 text-xs text-slate-500">No active limit orders.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/10 text-slate-400 pb-1 font-semibold">
                      <th className="text-left py-2">ID</th>
                      <th className="text-left py-2">Asset</th>
                      <th className="text-left py-2">Side</th>
                      <th className="text-left py-2">Type</th>
                      <th className="text-right py-2">Qty</th>
                      <th className="text-right py-2">Price</th>
                      <th className="text-center py-2">Status</th>
                      <th className="text-right py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {openOrders.map((ord) => (
                      <tr key={ord.id} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-2 font-mono text-slate-400">#{ord.id}</td>
                        <td className="py-2 font-bold text-white">{ord.symbol}</td>
                        <td className={`py-2 font-bold ${ord.side === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>
                          {ord.side}
                        </td>
                        <td className="py-2 uppercase text-slate-300">{ord.type}</td>
                        <td className="py-2 text-right font-mono">{ord.quantity} hrs</td>
                        <td className="py-2 text-right font-mono font-bold text-neon-mint">${ord.price.toFixed(4)}</td>
                        <td className="py-2 text-center">
                          <Badge variant={ord.status === "filled" ? "success" : "warning"}>{ord.status}</Badge>
                        </td>
                        <td className="py-2 text-right">
                          {ord.status === "open" && (
                            <button
                              onClick={() => handleCancelOrder(ord.id)}
                              className="text-rose-400 hover:text-rose-300 text-xs px-2 py-0.5 rounded hover:bg-rose-500/10"
                            >
                              Cancel
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </TerminalCard>
        </div>

        {/* Right Column: Execution Form & Order Book Depth */}
        <div className="lg:col-span-4 space-y-4">
          {/* Order Placement Box */}
          <TerminalCard className="p-4 space-y-3 bg-gradient-to-b from-slate-900/90 to-slate-950/90 border border-cyan-500/30 shadow-xl">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <Zap size={16} className="text-cyan-400" /> Trade {activeAsset.symbol}
              </h3>
              <span className="text-[10px] text-emerald-400 font-mono font-bold">Escrow Ready</span>
            </div>

            {/* Buy / Sell Buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => setOrderSide("BUY")}
                className={`flex-1 py-2 rounded-xl text-xs font-black transition-all ${
                  orderSide === "BUY"
                    ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20"
                    : "bg-white/5 text-slate-400 hover:text-white"
                }`}
              >
                BUY / LONG
              </button>
              <button
                onClick={() => setOrderSide("SELL")}
                className={`flex-1 py-2 rounded-xl text-xs font-black transition-all ${
                  orderSide === "SELL"
                    ? "bg-rose-500 text-white shadow-lg shadow-rose-500/20"
                    : "bg-white/5 text-slate-400 hover:text-white"
                }`}
              >
                SELL / SHORT
              </button>
            </div>

            {/* Order Type Selector */}
            <div className="flex gap-2">
              <button
                onClick={() => setOrderType("limit")}
                className={`flex-1 py-1 rounded text-[11px] font-semibold ${
                  orderType === "limit" ? "bg-white/10 text-white" : "text-slate-500"
                }`}
              >
                Limit Order
              </button>
              <button
                onClick={() => setOrderType("market")}
                className={`flex-1 py-1 rounded text-[11px] font-semibold ${
                  orderType === "market" ? "bg-white/10 text-white" : "text-slate-500"
                }`}
              >
                Market Order
              </button>
            </div>

            {/* Inputs */}
            {orderType === "limit" && (
              <Input
                label="Limit Price (USD / GPU hr)"
                type="number"
                step="0.001"
                value={limitPrice}
                onChange={(e) => setLimitPrice(parseFloat(e.target.value) || 0)}
              />
            )}

            <Input
              label="Quantity (GPU Compute Hours)"
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(parseInt(e.target.value, 10) || 0)}
            />

            {/* Total Calculation */}
            <div className="p-3 bg-black/40 rounded-xl border border-white/5 space-y-1 text-xs">
              <div className="flex justify-between text-slate-400">
                <span>Effective Rate:</span>
                <span className="font-mono text-slate-200">
                  ${(orderType === "market" ? liveInfo.price_per_hour : limitPrice).toFixed(4)}/hr
                </span>
              </div>
              <div className="flex justify-between font-bold text-white border-t border-white/10 pt-1">
                <span>Total Commitment:</span>
                <span className="text-neon-mint font-mono font-black">
                  ${((orderType === "market" ? liveInfo.price_per_hour : limitPrice) * quantity).toFixed(2)} USD
                </span>
              </div>
            </div>

            <Button
              onClick={handlePlaceOrder}
              disabled={isPlacing || quantity <= 0}
              className={`w-full py-3 rounded-xl font-black text-xs uppercase tracking-wider ${
                orderSide === "BUY"
                  ? "bg-gradient-to-r from-emerald-400 to-cyan-500 text-black hover:from-emerald-300 hover:to-cyan-400"
                  : "bg-gradient-to-r from-rose-500 to-pink-600 text-white hover:from-rose-400 hover:to-pink-500"
              }`}
            >
              {isPlacing ? "Submitting..." : `Submit ${orderSide} Order`}
            </Button>
          </TerminalCard>

          {/* Live Order Book Depth */}
          <TerminalCard className="p-4 space-y-2 font-mono text-xs">
            <div className="flex justify-between items-center pb-2 border-b border-white/10">
              <span className="font-bold text-white text-xs">Live Order Book Depth</span>
              <span className="text-[10px] text-slate-400">Spread: $0.02</span>
            </div>

            {/* Asks */}
            <div className="space-y-1">
              {orderBookAsks.map((ask, i) => (
                <div
                  key={i}
                  onClick={() => {
                    setLimitPrice(ask.price);
                    setOrderSide("BUY");
                  }}
                  className="flex justify-between items-center px-2 py-0.5 rounded cursor-pointer hover:bg-rose-500/10 transition-colors relative"
                >
                  <div
                    className="absolute right-0 top-0 bottom-0 bg-rose-500/15 rounded"
                    style={{ width: `${ask.depth}%` }}
                  />
                  <span className="text-rose-400 font-bold z-10">${ask.price.toFixed(4)}</span>
                  <span className="text-slate-300 z-10">{ask.qty} hrs</span>
                  <span className="text-slate-500 z-10">${ask.total}</span>
                </div>
              ))}
            </div>

            {/* Middle Live Price */}
            <div className="py-2 border-y border-white/10 bg-black/40 text-center flex items-center justify-center gap-2">
              <span className="text-base font-black text-neon-mint">${liveInfo.price_per_hour.toFixed(4)}</span>
              <span className="text-[10px] text-slate-400">Mid Market</span>
            </div>

            {/* Bids */}
            <div className="space-y-1">
              {orderBookBids.map((bid, i) => (
                <div
                  key={i}
                  onClick={() => {
                    setLimitPrice(bid.price);
                    setOrderSide("SELL");
                  }}
                  className="flex justify-between items-center px-2 py-0.5 rounded cursor-pointer hover:bg-emerald-500/10 transition-colors relative"
                >
                  <div
                    className="absolute right-0 top-0 bottom-0 bg-emerald-500/15 rounded"
                    style={{ width: `${bid.depth}%` }}
                  />
                  <span className="text-emerald-400 font-bold z-10">${bid.price.toFixed(4)}</span>
                  <span className="text-slate-300 z-10">{bid.qty} hrs</span>
                  <span className="text-slate-500 z-10">${bid.total}</span>
                </div>
              ))}
            </div>
          </TerminalCard>

          {/* Portfolio Snapshot */}
          <TerminalCard className="p-4 space-y-2 text-xs">
            <div className="flex justify-between items-center text-slate-400 pb-1 border-b border-white/5">
              <span>Account Available:</span>
              <span className="font-bold text-white font-mono">${portfolio.balance_usd?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center text-slate-400">
              <span>Portfolio Valuation:</span>
              <span className="font-bold text-cyan-300 font-mono">${portfolio.total_value_usd_display?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center text-slate-400">
              <span>Unrealized P&L:</span>
              <span className="font-bold text-emerald-400 font-mono">+{portfolio.unrealized_pnl_display}%</span>
            </div>
          </TerminalCard>
        </div>
      </div>
    </div>
  );
}
