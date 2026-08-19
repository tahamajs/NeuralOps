import React from "react";
import { Download, CreditCard, CheckCircle2 } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";

interface InvoiceRowProps {
  invoice: any;
  onDownload: () => void;
  onPay?: () => void;
}

export function InvoiceRow({ invoice, onDownload, onPay }: InvoiceRowProps) {
  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case "paid":
        return "success";
      case "draft":
        return "secondary";
      case "open":
      case "pending":
        return "warning";
      case "overdue":
      case "failed":
        return "danger";
      default:
        return "info";
    }
  };

  const invoiceNumber = invoice.invoice_number || invoice.id?.slice(0, 8) || "INV-1001";
  const amount = parseFloat(invoice.total_usd ?? invoice.amount ?? 0).toFixed(2);
  const isPending = invoice.status?.toLowerCase() === "pending";

  return (
    <tr className="border-b border-white/5 hover:bg-white/5 transition-colors font-sans text-xs">
      <td className="px-4 py-3 font-mono font-extrabold text-white">
        #{invoiceNumber}
      </td>
      <td className="px-4 py-3 text-slate-300 max-w-xs truncate">
        {invoice.description || `GPU Compute Billing Allocation`}
      </td>
      <td className="px-4 py-3 font-mono font-bold text-neon-mint">
        ${amount} {invoice.currency || "USD"}
      </td>
      <td className="px-4 py-3">
        <Badge variant={getStatusColor(invoice.status)}>{invoice.status?.toUpperCase()}</Badge>
      </td>
      <td className="px-4 py-3 text-slate-400 font-mono">
        {invoice.due_date ? new Date(invoice.due_date).toLocaleDateString() : "N/A"}
      </td>
      <td className="px-4 py-3 text-right">
        <div className="flex items-center justify-end gap-2">
          {isPending && onPay && (
            <Button
              size="sm"
              onClick={onPay}
              className="text-[11px] font-bold py-1 px-2.5 bg-gradient-to-r from-emerald-400 to-cyan-500 text-black h-7"
            >
              <CreditCard size={12} className="mr-1" /> Pay Now
            </Button>
          )}
          <button
            onClick={onDownload}
            className="p-1.5 hover:bg-white/10 rounded-lg text-neon-blue hover:text-white transition-colors"
            title="Download PDF Invoice"
          >
            <Download size={14} />
          </button>
        </div>
      </td>
    </tr>
  );
}
export default InvoiceRow;
