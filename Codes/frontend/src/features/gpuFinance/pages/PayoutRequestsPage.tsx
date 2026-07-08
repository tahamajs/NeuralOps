// src/features/gpuFinance/pages/PayoutRequestsPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { PayoutForm } from "./PayoutForm";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { DollarSign, Plus } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function PayoutRequestsPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [showForm, setShowForm] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: payouts, isLoading } = useQuery({
    queryKey: ["payout-requests"],
    queryFn: () => gpuFinanceAPI.getPayoutRequests(),
    refetchInterval: 10000,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <DollarSign size={20} className="text-neon-mint" /> Payout Requests
          </h1>
          <p className="text-xs text-slate-400">
            Withdraw earnings from GPU marketplace
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowForm(true)}>
            <Plus size={14} className="mr-1" /> Request Payout
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
                  <th className="px-4 py-3">Amount</th>
                  <th className="px-4 py-3">Method</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Requested</th>
                  <th className="px-4 py-3">Processed</th>
                </tr>
              </thead>
              <tbody>
                {payouts?.map((p) => (
                  <tr key={p.id} className="border-b border-white/5">
                    <td className="px-4 py-3">
                      ${p.amount.toFixed(2)} {p.currency}
                    </td>
                    <td className="px-4 py-3 capitalize">{p.method}</td>
                    <td className="px-4 py-3">
                      <Badge
                        variant={
                          p.status === "completed"
                            ? "success"
                            : p.status === "pending"
                              ? "warning"
                              : "failed"
                        }
                      >
                        {p.status}
                      </Badge>
                    </td>
                    <td className="px-4 py-3">
                      {new Date(p.requested_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">
                      {p.processed_at
                        ? new Date(p.processed_at).toLocaleString()
                        : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </TerminalCard>
      </div>
      {(PayoutForm as any)({
        open: showForm,
        onClose: () => setShowForm(false),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ["payout-requests"] })
      })}
    </div>
  );
}
