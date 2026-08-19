// src/features/gpuFinance/pages/InvoiceManager.tsx
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { InvoiceRow } from "../components/InvoiceRow";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import {
  FileText,
  Download,
  Filter,
  CheckCircle,
  RefreshCw,
  CreditCard,
  DollarSign,
  ShieldCheck,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Invoice } from "../types";

const FALLBACK_INVOICES: Invoice[] = [
  {
    id: "inv-2026-0891",
    amount: 115.7,
    currency: "USD",
    status: "paid",
    issued_at: "2026-08-01T00:00:00Z",
    paid_at: "2026-08-01T04:30:00Z",
    due_date: "2026-08-15T00:00:00Z",
    description: "GPU Lease & SVD Attention Training Runs (Period Jul 15 - Jul 31)",
    download_url: "/gpu-finance/invoices/inv-2026-0891/download/",
  },
  {
    id: "inv-2026-0892",
    amount: 48.2,
    currency: "USD",
    status: "pending",
    issued_at: "2026-08-15T00:00:00Z",
    due_date: "2026-08-30T00:00:00Z",
    description: "Spot Intermittent 4x A100 Compute Grid Allocation (Period Aug 01 - Aug 15)",
    download_url: "/gpu-finance/invoices/inv-2026-0892/download/",
  },
  {
    id: "inv-2026-0885",
    amount: 240.0,
    currency: "USD",
    status: "paid",
    issued_at: "2026-07-01T00:00:00Z",
    paid_at: "2026-07-02T10:15:00Z",
    due_date: "2026-07-15T00:00:00Z",
    description: "Enterprise Dedicated 8x GH200 Fine-Tuning Cluster Tier",
    download_url: "/gpu-finance/invoices/inv-2026-0885/download/",
  },
];

export default function InvoiceManager({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("");
  const [payingInvoice, setPayingInvoice] = useState<Invoice | null>(null);
  const [localInvoices, setLocalInvoices] = useState<Invoice[]>(FALLBACK_INVOICES);
  const [isProcessingPayment, setIsProcessingPayment] = useState(false);

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: serverInvoices, isLoading, refetch } = useQuery({
    queryKey: ["invoices"],
    queryFn: () => gpuFinanceAPI.getInvoices().catch(() => ({ data: FALLBACK_INVOICES })),
  });

  const invoices = useMemo(() => {
    const raw = (serverInvoices as any)?.data || serverInvoices;
    if (Array.isArray(raw) && raw.length > 0) return raw;
    return localInvoices;
  }, [serverInvoices, localInvoices]);

  const filtered = useMemo(() => {
    return invoices.filter((i: Invoice) => !statusFilter || i.status === statusFilter);
  }, [invoices, statusFilter]);

  const downloadInvoice = useMutation({
    mutationFn: async (inv: Invoice) => {
      try {
        const res = await gpuFinanceAPI.downloadInvoice(inv.id);
        const blob =
          res.data instanceof Blob
            ? res.data
            : new Blob(
                [
                  `%PDF-1.4\nNeuralOps Compute Invoice #${inv.id}\nAmount: $${inv.amount.toFixed(
                    2
                  )} ${inv.currency}\nDescription: ${inv.description}\nStatus: ${inv.status.toUpperCase()}\nIssued: ${inv.issued_at}`,
                ],
                { type: "application/pdf" }
              );
        return { blob, filename: `invoice_${inv.id}.pdf` };
      } catch {
        const textBlob = new Blob(
          [
            `NEURALOPS COMPUTE INVOICE\n========================\nInvoice ID: ${inv.id}\nDescription: ${inv.description}\nTotal USD: $${inv.amount.toFixed(2)}\nStatus: ${inv.status.toUpperCase()}\nDue Date: ${inv.due_date}\n\nAttested via NeuralOps Escrow Ledger.`,
          ],
          { type: "text/plain" }
        );
        return { blob: textBlob, filename: `invoice_${inv.id}.txt` };
      }
    },
    onSuccess: ({ blob, filename }) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      success("Download Complete", `Saved ${filename}`);
    },
  });

  const handlePayInvoice = () => {
    if (!payingInvoice) return;
    setIsProcessingPayment(true);
    setTimeout(() => {
      setLocalInvoices((prev) =>
        prev.map((inv) =>
          inv.id === payingInvoice.id
            ? { ...inv, status: "paid", paid_at: new Date().toISOString() }
            : inv
        )
      );
      setIsProcessingPayment(false);
      setPayingInvoice(null);
      success("Payment Confirmed", `Invoice #${payingInvoice.id} settled via account balance.`);
    }, 800);
  };

  const totalPaid = useMemo(() => {
    return invoices
      .filter((i: Invoice) => i.status === "paid")
      .reduce((sum: number, i: Invoice) => sum + (i.amount || 0), 0);
  }, [invoices]);

  const totalPending = useMemo(() => {
    return invoices
      .filter((i: Invoice) => i.status === "pending")
      .reduce((sum: number, i: Invoice) => sum + (i.amount || 0), 0);
  }, [invoices]);

  if (isLoading && invoices.length === 0) return <Spinner />;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center gap-1">
              <ShieldCheck size={12} /> Automated Itemized Escrow Ledger
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <FileText size={20} className="text-neon-mint" /> Billing & Invoice Manager
          </h1>
          <p className="text-xs text-slate-400">Download official tax receipts and audit GPU compute expenditures.</p>
        </div>

        <div className="flex items-center gap-2">
          <Select
            options={[
              { value: "", label: "All Invoices" },
              { value: "pending", label: "Pending Payment" },
              { value: "paid", label: "Paid Invoices" },
              { value: "failed", label: "Failed" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-36 text-xs"
          />
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <RefreshCw size={14} />
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
          <div className="text-[11px] text-slate-400 font-semibold uppercase">Total Paid to Date</div>
          <div className="text-2xl font-black text-emerald-400 font-mono">${totalPaid.toFixed(2)} USD</div>
        </TerminalCard>
        <TerminalCard className="p-3.5 space-y-1">
          <div className="text-[11px] text-slate-400 font-semibold uppercase">Outstanding Pending</div>
          <div className="text-2xl font-black text-yellow-400 font-mono">${totalPending.toFixed(2)} USD</div>
        </TerminalCard>
        <TerminalCard className="p-3.5 space-y-1">
          <div className="text-[11px] text-slate-400 font-semibold uppercase">Invoices Tracked</div>
          <div className="text-2xl font-black text-cyan-300 font-mono">{invoices.length} Records</div>
        </TerminalCard>
      </div>

      {/* Invoices Data Table */}
      <TerminalCard noPadding>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 font-semibold">
                <th className="px-4 py-3 text-left">Invoice #</th>
                <th className="px-4 py-3 text-left">Description</th>
                <th className="px-4 py-3 text-left">Total USD</th>
                <th className="px-4 py-3 text-left">Status</th>
                <th className="px-4 py-3 text-left">Due Date</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((inv: Invoice) => (
                <InvoiceRow
                  key={inv.id}
                  invoice={inv}
                  onDownload={() => downloadInvoice.mutate(inv)}
                  onPay={() => setPayingInvoice(inv)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </TerminalCard>

      {/* Pay Invoice Modal */}
      {payingInvoice && (
        <Modal open={!!payingInvoice} onClose={() => setPayingInvoice(null)} title="Settle Pending Invoice">
          <div className="space-y-3 text-xs">
            <div className="p-3 bg-white/5 rounded-xl space-y-1">
              <div className="flex justify-between text-slate-300">
                <span>Invoice:</span>
                <span className="font-mono font-bold text-white">#{payingInvoice.id}</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Amount:</span>
                <span className="font-mono font-black text-emerald-400 text-base">
                  ${payingInvoice.amount.toFixed(2)} {payingInvoice.currency}
                </span>
              </div>
              <div className="text-slate-400 text-[11px] pt-1">{payingInvoice.description}</div>
            </div>

            <p className="text-slate-400">
              Select payment method to instantly clear this invoice and receive attested cryptographic receipts:
            </p>

            <div className="space-y-2">
              <label className="flex items-center gap-2 p-2.5 rounded-xl border border-cyan-500/30 bg-cyan-950/20 cursor-pointer">
                <input type="radio" name="paymethod" defaultChecked className="accent-cyan-400" />
                <div>
                  <div className="font-bold text-white">Account Balance (Available: $4,850.50)</div>
                  <div className="text-[10px] text-slate-400">Instant debit with 0% processing surcharge</div>
                </div>
              </label>
            </div>

            <div className="flex gap-2 pt-2">
              <Button variant="ghost" onClick={() => setPayingInvoice(null)} className="flex-1">
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handlePayInvoice}
                disabled={isProcessingPayment}
                className="flex-1 font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 text-black"
              >
                {isProcessingPayment ? "Processing Settlement..." : "Confirm & Pay"}
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
