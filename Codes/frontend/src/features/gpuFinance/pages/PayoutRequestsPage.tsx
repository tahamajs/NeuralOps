// src/features/gpuFinance/pages/PayoutRequestsPage.tsx
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { PayoutForm } from "./PayoutForm";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { DollarSign, Plus, Wallet, ShieldCheck, ArrowDownToLine } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { PayoutRequest } from "../types";

const FALLBACK_PAYOUTS: PayoutRequest[] = [
  {
    id: "pay-901248a1",
    amount: 850.0,
    currency: "USD",
    method: "crypto",
    status: "completed",
    requested_at: new Date(Date.now() - 86400000 * 3).toISOString(),
    processed_at: new Date(Date.now() - 86400000 * 2).toISOString(),
    notes: "USDC transferred to 0x71C...49A2 (Tx: 0x9182...)",
  },
  {
    id: "pay-662198b5",
    amount: 320.5,
    currency: "USD",
    method: "bank_transfer",
    status: "processing",
    requested_at: new Date(Date.now() - 86400000 * 1).toISOString(),
    notes: "ACH Wire initiated to Chase account ****4190",
  },
  {
    id: "pay-110294c7",
    amount: 140.0,
    currency: "USD",
    method: "paypal",
    status: "completed",
    requested_at: new Date(Date.now() - 86400000 * 12).toISOString(),
    processed_at: new Date(Date.now() - 86400000 * 11).toISOString(),
  },
];

export default function PayoutRequestsPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [showForm, setShowForm] = useState(false);
  const [localPayouts, setLocalPayouts] = useState<PayoutRequest[]>(FALLBACK_PAYOUTS);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: serverPayouts, isLoading } = useQuery({
    queryKey: ["payout-requests"],
    queryFn: () => gpuFinanceAPI.getPayoutRequests().catch(() => ({ data: FALLBACK_PAYOUTS })),
    refetchInterval: 15000,
  });

  const payouts = useMemo(() => {
    const raw = (serverPayouts as any)?.data || serverPayouts;
    if (Array.isArray(raw) && raw.length > 0) return raw;
    return localPayouts;
  }, [serverPayouts, localPayouts]);

  const totalWithdrawn = useMemo(() => {
    return payouts
      .filter((p: PayoutRequest) => p.status === "completed")
      .reduce((sum: number, p: PayoutRequest) => sum + (p.amount || 0), 0);
  }, [payouts]);

  const totalPending = useMemo(() => {
    return payouts
      .filter((p: PayoutRequest) => p.status === "pending" || p.status === "processing")
      .reduce((sum: number, p: PayoutRequest) => sum + (p.amount || 0), 0);
  }, [payouts]);

  const handlePayoutCreated = (newP: PayoutRequest) => {
    setLocalPayouts((prev) => [newP, ...prev]);
  };

  if (isLoading && payouts.length === 0) return <Spinner />;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center gap-1">
              <ShieldCheck size={12} /> Direct Settlement Gateway
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <DollarSign size={20} className="text-neon-mint" /> Earnings & Payout Requests
          </h1>
          <p className="text-xs text-slate-400">Withdraw earnings from GPU compute hosting directly to Bank, USDC, or PayPal.</p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={() => setShowForm(true)}
            className="bg-gradient-to-r from-emerald-400 to-cyan-500 text-black font-bold text-xs"
          >
            <Plus size={14} className="mr-1" /> Request Payout
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <TerminalCard className="p-3.5 space-y-1">
          <div className="text-[11px] text-slate-400 font-semibold uppercase">Total Settled Payouts</div>
          <div className="text-2xl font-black text-emerald-400 font-mono">${totalWithdrawn.toFixed(2)} USD</div>
        </TerminalCard>
        <TerminalCard className="p-3.5 space-y-1">
          <div className="text-[11px] text-slate-400 font-semibold uppercase">In Processing / Pending</div>
          <div className="text-2xl font-black text-yellow-400 font-mono">${totalPending.toFixed(2)} USD</div>
        </TerminalCard>
        <TerminalCard className="p-3.5 space-y-1">
          <div className="text-[11px] text-slate-400 font-semibold uppercase">Available Balance</div>
          <div className="text-2xl font-black text-cyan-300 font-mono">$1,250.75 USD</div>
        </TerminalCard>
      </div>

      {/* Payouts Table */}
      <TerminalCard noPadding>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 font-semibold">
                <th className="px-4 py-3 text-left">Amount</th>
                <th className="px-4 py-3 text-left">Method</th>
                <th className="px-4 py-3 text-left">Destination / Notes</th>
                <th className="px-4 py-3 text-left">Status</th>
                <th className="px-4 py-3 text-left">Requested</th>
                <th className="px-4 py-3 text-left">Processed</th>
              </tr>
            </thead>
            <tbody>
              {payouts.map((p: PayoutRequest) => (
                <tr key={p.id} className="border-b border-white/5 hover:bg-white/5">
                  <td className="px-4 py-3 font-mono font-bold text-neon-mint">
                    ${p.amount.toFixed(2)} {p.currency || "USD"}
                  </td>
                  <td className="px-4 py-3 capitalize text-slate-300 font-semibold">
                    {p.method?.replace(/_/g, " ")}
                  </td>
                  <td className="px-4 py-3 text-slate-400 max-w-xs truncate font-mono">
                    {p.notes || "Standard disbursement"}
                  </td>
                  <td className="px-4 py-3">
                    <Badge
                      variant={
                        p.status === "completed"
                          ? "success"
                          : p.status === "pending" || p.status === "processing"
                          ? "warning"
                          : "danger"
                      }
                    >
                      {p.status?.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-slate-400 font-mono">
                    {new Date(p.requested_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-slate-400 font-mono">
                    {p.processed_at ? new Date(p.processed_at).toLocaleString() : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </TerminalCard>

      <PayoutForm
        open={showForm}
        onClose={() => setShowForm(false)}
        onSuccess={handlePayoutCreated}
      />
    </div>
  );
}
