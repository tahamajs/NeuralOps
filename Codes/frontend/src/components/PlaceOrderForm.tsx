import { useState } from "react";
import { Button } from "./ui/Button";
import { Input } from "./ui/Input";
import { useToast } from "./ui/Toast";
import { marketplaceApi } from "../api/marketplace";

interface PlaceOrderFormProps {
  listingId: string;
  onOrderPlaced: () => void;
}

export function PlaceOrderForm({
  listingId,
  onOrderPlaced,
}: PlaceOrderFormProps) {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [quantity, setQuantity] = useState<number>(1);
  const [price, setPrice] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const { success, error } = useToast();

  const handleSubmit = async () => {
    if (quantity <= 0 || price <= 0) {
      error("Invalid input", "Quantity and price must be positive.");
      return;
    }
    setLoading(true);
    try {
      await marketplaceApi.placeOrder(listingId, side, quantity, price);
      success(
        "Order placed",
        "Your order has been submitted to the order book.",
      );
      onOrderPlaced();
    } catch (err) {
      error(
        "Failed to place order",
        "Insufficient balance or invalid parameters.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-terminal p-4 rounded-xl">
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setSide("BUY")}
          className={`flex-1 py-2 rounded ${side === "BUY" ? "bg-green-500/30 text-green-400" : "bg-white/5"}`}
        >
          Buy
        </button>
        <button
          onClick={() => setSide("SELL")}
          className={`flex-1 py-2 rounded ${side === "SELL" ? "bg-red-500/30 text-red-400" : "bg-white/5"}`}
        >
          Sell
        </button>
      </div>
      <Input
        label="Quantity (GPU hours)"
        type="number"
        value={quantity}
        onChange={(e) => setQuantity(parseInt(e.target.value))}
      />
      <Input
        label="Price (USD / hour)"
        type="number"
        step="0.0001"
        value={price}
        onChange={(e) => setPrice(parseFloat(e.target.value))}
      />
      <div className="mt-3 text-xs text-slate-500">
        Total: {(quantity * price).toFixed(4)} USD
      </div>
      <Button onClick={handleSubmit} className="w-full mt-4" disabled={loading}>
        {loading ? "Placing..." : `Place ${side} Order`}
      </Button>
    </div>
  );
}
