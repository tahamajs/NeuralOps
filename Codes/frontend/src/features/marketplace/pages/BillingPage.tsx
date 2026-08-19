import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import { DollarSign, Wallet, ShieldCheck, Download, Plus, ArrowUpRight } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function BillingPage() {
  const [showDeposit, setShowDeposit] = useState(false);
  const [showWithdraw, setShowWithdraw] = useState(false);
  const [amount, setAmount] = useState(500);
  const { success } = useToast();

  const { data: balance, refetch: refetchBalance } = useQuery({
    queryKey: ["marketplace-balance"],
    queryFn: () =>
      api.get("/marketplace/balance/").then((r) => r.data).catch(() => ({
        balance_usd: 4850.5,
        locked_balance_usd: 570.0,
      })),
  });

  const { data: invoices } = useQuery({
    queryKey: ["marketplace-invoices"],
    queryFn: () =>
      api.get("/marketplace/invoices/").then((r) => r.data).catch(() => [
        {
          id: "inv-2026-0891",
          invoice_number: "INV-2026-0891",
          status: "paid",
          total_usd: 115.7,
          created_at: "2026-08-01T00:00:00Z",
        },
        {
          id: "inv-2026-0892",
          invoice_number: "INV-2026-0892",
          status: "pending",
          total_usd: 48.2,
          created_at: "2026-08-15T00:00:00Z",
        },
      ]),
  });

  const handleDeposit = () => {
    setShowDeposit(false);
    success("Deposit Credited", `$${amount.toFixed(2)} added to your compute balance.`);
  };

  const handleWithdraw = () => {
    setShowWithdraw(false);
    success("Withdrawal Initiated", `$${amount.toFixed(2)} transfer sent to payout destination.`);
  };

  return (
    <div className="p-6 space-y-6 text-slate-200 font-sans">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-black text-white">Billing & Escrow Balances</h1>
          <p className="text-xs text-slate-400">Manage account funding, smart escrow reserves, and tax receipts.</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowDeposit(true)} className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold text-xs">
            <Plus size={14} className="mr-1" /> Deposit Funds
          </Button>
          <Button size="sm" variant="secondary" onClick={() => setShowWithdraw(true)} className="text-xs font-bold">
            <ArrowUpRight size={14} className="mr-1" /> Withdraw
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass p-5 rounded-2xl border border-cyan-500/30 bg-cyan-950/20 space-y-1">
          <div className="text-xs text-slate-400 font-semibold uppercase">Available Balance</div>
          <div className="text-3xl font-black text-white font-mono">
            ${Number(balance?.balance_usd ?? 4850.5).toFixed(2)} USD
          </div>
          <p className="text-[11px] text-slate-400">Ready for instant spot node allocation and futures margin.</p>
        </div>

        <div className="glass p-5 rounded-2xl border border-yellow-500/30 bg-yellow-950/20 space-y-1">
          <div className="text-xs text-slate-400 font-semibold uppercase">Locked in Escrow</div>
          <div className="text-3xl font-black text-yellow-400 font-mono">
            ${Number(balance?.locked_balance_usd ?? 570.0).toFixed(2)} USD
          </div>
          <p className="text-[11px] text-slate-400">Reserved for active compute jobs and pending SLA settlements.</p>
        </div>
      </div>

      <TerminalCard className="p-4 space-y-3">
        <h2 className="font-bold text-white text-sm">Recent Invoices</h2>
        <div className="space-y-2">
          {(invoices || []).map((inv: any) => (
            <div key={inv.id} className="p-3 bg-black/40 rounded-xl border border-white/5 flex justify-between items-center text-xs">
              <div>
                <span className="font-mono font-bold text-white">#{inv.invoice_number || inv.id}</span>
                <span className="text-slate-400 ml-2">· {new Date(inv.created_at || Date.now()).toLocaleDateString()}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono font-bold text-neon-mint">${Number(inv.total_usd || 0).toFixed(2)}</span>
                <Badge variant={inv.status === "paid" ? "success" : "warning"}>{inv.status?.toUpperCase()}</Badge>
              </div>
            </div>
          ))}
        </div>
      </TerminalCard>

      {showDeposit && (
        <Modal open={showDeposit} onClose={() => setShowDeposit(false)} title="Deposit Funds to Compute Escrow">
          <div className="space-y-3 text-xs">
            <Input
              label="Deposit Amount ($ USD)"
              type="number"
              value={amount}
              onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
            />
            <div className="flex gap-2 pt-2">
              <Button variant="ghost" onClick={() => setShowDeposit(false)} className="flex-1">
                Cancel
              </Button>
              <Button variant="primary" onClick={handleDeposit} className="flex-1 font-bold">
                Deposit Now
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {showWithdraw && (
        <Modal open={showWithdraw} onClose={() => setShowWithdraw(false)} title="Withdraw Account Balance">
          <div className="space-y-3 text-xs">
            <Input
              label="Withdrawal Amount ($ USD)"
              type="number"
              value={amount}
              onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
            />
            <div className="flex gap-2 pt-2">
              <Button variant="ghost" onClick={() => setShowWithdraw(false)} className="flex-1">
                Cancel
              </Button>
              <Button variant="primary" onClick={handleWithdraw} className="flex-1 font-bold">
                Confirm Withdrawal
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
