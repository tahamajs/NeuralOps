import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";

export default function PlaceOrderForm({ gpuModel }: { gpuModel: string }) {
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [orderType, setOrderType] = useState<"limit" | "market">("limit");
  const [price, setPrice] = useState("");
  const [quantity, setQuantity] = useState("");

  const placeOrder = useMutation({
    mutationFn: (data: any) => api.post("/trading/orders/", data),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    placeOrder.mutate({
      gpu_model: gpuModel,
      side,
      order_type: orderType,
      price: orderType === "limit" ? price : null,
      quantity: parseInt(quantity),
    });
  };

  return (
    <div className="glass p-4">
      <h2 className="text-lg font-bold mb-2">Place Order</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setSide("buy")}
            className={`flex-1 py-2 rounded font-bold ${side === "buy" ? "bg-green-600 text-white" : "bg-slate-700 text-slate-300"}`}
          >
            Buy
          </button>
          <button
            type="button"
            onClick={() => setSide("sell")}
            className={`flex-1 py-2 rounded font-bold ${side === "sell" ? "bg-red-600 text-white" : "bg-slate-700 text-slate-300"}`}
          >
            Sell
          </button>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setOrderType("limit")}
            className={`flex-1 py-1 rounded text-sm ${orderType === "limit" ? "bg-blue-600 text-white" : "bg-slate-700"}`}
          >
            Limit
          </button>
          <button
            type="button"
            onClick={() => setOrderType("market")}
            className={`flex-1 py-1 rounded text-sm ${orderType === "market" ? "bg-blue-600 text-white" : "bg-slate-700"}`}
          >
            Market
          </button>
        </div>
        {orderType === "limit" && (
          <input
            type="number"
            step="0.01"
            placeholder="Price (USD/hr)"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-white"
            required
          />
        )}
        <input
          type="number"
          placeholder="Quantity (hours)"
          value={quantity}
          onChange={(e) => setQuantity(e.target.value)}
          className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-white"
          required
        />
        <button
          type="submit"
          className="w-full bg-blue-600 py-2 rounded font-bold"
        >
          Place Order
        </button>
      </form>
    </div>
  );
}
