// src/features/gpuFinance/pages/InvoiceManager.tsx
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { InvoiceRow } from "../components/InvoiceRow";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import {
  FileText,
  Download,
  ExternalLink,
  Filter,
  CheckCircle,
  RefreshCw,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

const InvoiceCard = ({ invoice, onDownload }: any) => {
  return <div />;
};

export default function InvoiceManager({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("");
  const {
    data: invoices,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["invoices"],
    queryFn: () => gpuFinanceAPI.getInvoices(),
  });

  const downloadInvoice = useMutation({
    mutationFn: (id: string) => gpuFinanceAPI.downloadInvoice(id),
    onSuccess: (res) => {
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `invoice_${Date.now()}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    },
  });

  const filtered =
    invoices?.filter((i) => !statusFilter || i.status === statusFilter) || [];

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FileText size={20} className="text-neon-mint" /> Invoices
          </h1>
          <p className="text-xs text-slate-400">
            Billing history and payment tracking
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={[
              { value: "", label: "All" },
              { value: "pending", label: "Pending" },
              { value: "paid", label: "Paid" },
              { value: "failed", label: "Failed" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-32"
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
      <div className="flex-1 overflow-auto p-4">
        <TerminalCard noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-slate-400">
                  <th className="px-4 py-3 text-left">Invoice #</th>
                  <th className="px-4 py-3 text-left">Description</th>
                  <th className="px-4 py-3 text-left">Amount</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Due Date</th>
                  <th className="px-4 py-3 text-left"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((inv) => (
                  <InvoiceRow
                    key={inv.id}
                    invoice={inv}
                    onDownload={() => downloadInvoice.mutate(inv.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </TerminalCard>
      </div>
    </div>
  );
}
