import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card, Button, Input, Spinner, Table } from "../../../components/ui";
import { useState } from "react";

export default function EscrowBalancePage() {
  const queryClient = useQueryClient();
  const [amount, setAmount] = useState("");
  const [withdrawAmount, setWithdrawAmount] = useState("");
  const { data: balance, isLoading } = useQuery({
    queryKey: ["escrow-balance"],
    queryFn: async () => (await api.get("/balance/")).data,
  });
  const { data: transactions } = useQuery({
    queryKey: ["escrow-transactions"],
    queryFn: async () => (await api.get("/transactions/")).data,
  });

  const deposit = useMutation({
    mutationFn: () => api.post("/deposit/", { amount: parseFloat(amount) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["escrow-balance"] });
      setAmount("");
    },
  });
  const withdraw = useMutation({
    mutationFn: () =>
      api.post("/withdraw/", { amount: parseFloat(withdrawAmount) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["escrow-balance"] });
      setWithdrawAmount("");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold text-white">Escrow Wallet</h1>

      <Card className="text-center">
        <div className="text-3xl font-bold text-neon-blue">
          {balance?.balance || 0} tokens
        </div>
        <div className="text-xs text-slate-400">Available balance</div>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <Input
              type="number"
              placeholder="Amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
            <Button className="mt-2" onClick={() => deposit.mutate()}>
              Deposit
            </Button>
          </div>
          <div>
            <Input
              type="number"
              placeholder="Amount"
              value={withdrawAmount}
              onChange={(e) => setWithdrawAmount(e.target.value)}
            />
            <Button
              className="mt-2"
              variant="secondary"
              onClick={() => withdraw.mutate()}
            >
              Withdraw
            </Button>
          </div>
        </div>
      </Card>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">
          Transaction History
        </h2>
        <Table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Amount</th>
              <th>Status</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {transactions?.map((tx: any) => (
              <tr key={tx.id}>
                <td>{tx.type}</td>
                <td
                  className={tx.amount > 0 ? "text-green-400" : "text-red-400"}
                >
                  {tx.amount} tokens
                </td>
                <td>
                  <Badge variant={tx.status}>{tx.status}</Badge>
                </td>
                <td>{new Date(tx.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  );
}
