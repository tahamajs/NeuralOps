import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Download } from "lucide-react";

export function InvoiceRow({ invoice, onDownload }: any) {
  const statusConfig = {
    pending: "warning",
    paid: "success",
    failed: "danger",
  };
  return (
    <tr className="border-b border-white/5 hover:bg-white/5">
      <td className="px-4 py-3 font-mono">{invoice.id.slice(0, 8)}</td>
      <td className="px-4 py-3">{invoice.description}</td>
      <td className="px-4 py-3">
        ${invoice.amount.toFixed(2)} {invoice.currency}
      </td>
      <td className="px-4 py-3">
        <Badge variant={statusConfig[invoice.status]}>{invoice.status}</Badge>
      </td>
      <td className="px-4 py-3">
        {new Date(invoice.due_date).toLocaleDateString()}
      </td>
      <td className="px-4 py-3">
        <Button size="sm" variant="ghost" onClick={onDownload}>
          <Download size={14} />
        </Button>
      </td>
    </tr>
  );
}
