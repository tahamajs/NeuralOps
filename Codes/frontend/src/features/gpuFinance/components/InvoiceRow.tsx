import React from "react";
import { Download } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";

interface InvoiceRowProps {
  invoice: any;
  onDownload: () => void;
}

export function InvoiceRow({ invoice, onDownload }: InvoiceRowProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "paid":
        return "success";
      case "draft":
        return "secondary";
      case "open":
        return "warning";
      case "overdue":
        return "failed";
      default:
        return "info";
    }
  };

  return (
    <tr className="border-b border-white/5 hover:bg-white/5 transition-colors font-sans text-xs">
      <td className="px-4 py-3 font-mono font-extrabold text-white">
        #{invoice.invoice_number}
      </td>
      <td className="px-4 py-3 text-slate-300">
        {invoice.description ||
          `GPU Lease Billing for Period ${invoice.period_start || ""} - ${invoice.period_end || ""}`}
      </td>
      <td className="px-4 py-3 font-mono font-bold text-neon-mint">
        ${parseFloat(invoice.total_usd || 0).toFixed(2)}
      </td>
      <td className="px-4 py-3">
        <Badge variant={getStatusColor(invoice.status)}>{invoice.status}</Badge>
      </td>
      <td className="px-4 py-3 text-slate-500 font-mono">
        {invoice.due_date
          ? new Date(invoice.due_date).toLocaleDateString()
          : "N/A"}
      </td>
      <td className="px-4 py-3 text-right">
        <button
          onClick={onDownload}
          className="p-1 hover:bg-white/5 rounded text-neon-blue hover:text-white transition-colors"
          title="Download PDF"
        >
          <Download size={14} />
        </button>
      </td>
    </tr>
  );
}
export default InvoiceRow;
