import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { marketplaceApi } from "../../../api/marketplace";
import { useMarketWebSocket } from "../../../hooks/useMarketWebSocket";
import { CandlestickChart } from "../../../components/CandlestickChart";
import { OrderBook } from "../../../components/OrderBook";
import { PlaceOrderForm } from "../../../components/PlaceOrderForm";
import { PortfolioTracker } from "../../../components/PortfolioTracker";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { ArrowLeft, TrendingUp, TrendingDown } from "lucide-react";

export default function GPUTradingTerminal() {
  const { listingId } = useParams();
  const navigate = useNavigate();
  const { data: listing, isLoading: listingLoading } = useQuery({
    queryKey: ["listing", listingId],
    queryFn: () => marketplaceApi.getListingDetail(listingId!),
    enabled: !!listingId,
  });
  const { data: orderBook, refetch: refetchOrderBook } = useQuery({
    queryKey: ["orderBook", listingId],
    queryFn: () => marketplaceApi.getOrderBook(listingId!),
    enabled: !!listingId,
    refetchInterval: 5000,
  });
  const { data: priceHistory } = useQuery({
    queryKey: ["priceHistory", listingId],
    queryFn: () => marketplaceApi.getPriceHistory(listingId!),
    enabled: !!listingId,
  });
  const { prices } = useMarketWebSocket();
  const livePrice = prices[listingId || ""]?.price_per_hour;
  const change =
    prices[listingId || ""]?.change_percent || listing?.change_percent || 0;

  if (listingLoading)
    return <div className="p-4">Loading trading terminal...</div>;
  if (!listing)
    return <div className="p-4 text-red-400">Listing not found</div>;

  // Build candlestick data from price history (simplified: group by minute)
  const candleData =
    priceHistory?.map((p) => ({
      time: p.time,
      open: p.price,
      high: p.price,
      low: p.price,
      close: p.price,
    })) || [];

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate("/marketplace")}
        >
          <ArrowLeft size={16} />
        </Button>
        <div>
          <h1 className="text-xl font-bold text-white">
            {listing.gpu_model_detail.name}
          </h1>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-mono text-neon-mint">
              ${(livePrice ?? listing.price_per_hour).toFixed(4)}
            </span>
            <span
              className={`text-sm flex items-center ${change >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {change >= 0 ? (
                <TrendingUp size={14} className="mr-0.5" />
              ) : (
                <TrendingDown size={14} className="mr-0.5" />
              )}
              {change >= 0 ? "+" : ""}
              {change.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 space-y-4">
          <TerminalCard className="p-2">
            <CandlestickChart data={candleData} height={300} />
          </TerminalCard>
          {orderBook && (
            <OrderBook bids={orderBook.bids} asks={orderBook.asks} />
          )}
        </div>
        <div className="space-y-4">
          <PlaceOrderForm
            listingId={listingId!}
            onOrderPlaced={refetchOrderBook}
          />
          <PortfolioTracker />
        </div>
      </div>
    </div>
  );
}
