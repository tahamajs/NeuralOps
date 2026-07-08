import { DataGrid } from "../../../components/ui/DataGrid";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Edit, Trash2, Zap } from "lucide-react";
import { Webhook } from "../types";

export function WebhookTable({ webhooks, onEdit, onDelete, onTest }: any) {
  const columns = [
    { accessorKey: "name", header: "Name" },
    {
      accessorKey: "url",
      header: "URL",
      cell: ({ row }: any) => (
        <span className="font-mono text-sm">{row.original.url}</span>
      ),
    },
    {
      accessorKey: "events",
      header: "Events",
      cell: ({ row }: any) =>
        row.original.events
          .slice(0, 2)
          .map((e: string) => <Badge key={e}>{e}</Badge>),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }: any) => (
        <Badge
          variant={row.original.status === "active" ? "success" : "failed"}
        >
          {row.original.status}
        </Badge>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }: any) => (
        <div className="flex gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onTest(row.original.id)}
          >
            <Zap size={12} />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onEdit(row.original)}
          >
            <Edit size={12} />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onDelete(row.original.id)}
          >
            <Trash2 size={12} className="text-red-400" />
          </Button>
        </div>
      ),
    },
  ];
  return <DataGrid data={webhooks} columns={columns} />;
}
