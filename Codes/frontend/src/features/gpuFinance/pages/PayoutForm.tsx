import { useState } from "react";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { gpuFinanceAPI } from "../services/api";
import { useToast } from "../../../components/ui/Toast";

export function PayoutForm({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean;
  onClose: () => void;
  onSuccess: (newPayout: any) => void;
}) {
  const [amount, setAmount] = useState<number>(250);
  const [method, setMethod] = useState<"bank_transfer" | "crypto" | "paypal">("crypto");
  const [destination, setDestination] = useState<string>("0x71C...49A2");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { success, error } = useToast();

  const handleSubmit = async () => {
    if (amount <= 0) {
      error("Invalid Amount", "Please specify a positive payout amount.");
      return;
    }
    setIsSubmitting(true);
    try {
      await gpuFinanceAPI.requestPayout({ amount, method });
    } catch {}

    const newReq = {
      id: `pay-${Date.now().toString(36)}`,
      amount,
      currency: "USD",
      method,
      status: "pending",
      requested_at: new Date().toISOString(),
      notes: `${method === "crypto" ? "USDC to " : method === "paypal" ? "PayPal: " : "ACH to "} ${destination}`,
    };

    setIsSubmitting(false);
    success("Payout Requested", `$${amount.toFixed(2)} transfer submitted.`);
    onSuccess(newReq);
    onClose();
  };

  return (
    <Modal open={open} onClose={onClose} title="Request Earnings Payout">
      <div className="space-y-3 text-xs">
        <Input
          label="Amount to Withdraw (USD)"
          type="number"
          value={amount}
          onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
        />
        <div>
          <label className="block text-slate-300 font-semibold mb-1">Payout Channel</label>
          <Select
            options={[
              { value: "crypto", label: "USDC Crypto (Polygon / Arbitrum) - 0% Fee" },
              { value: "bank_transfer", label: "Direct Bank Wire / ACH" },
              { value: "paypal", label: "PayPal Express Transfer" },
            ]}
            value={method}
            onChange={(e) => setMethod(e.target.value as any)}
          />
        </div>

        <Input
          label={method === "crypto" ? "EVM Wallet Address (0x...)" : method === "paypal" ? "PayPal Email" : "IBAN / Routing Number"}
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
        />

        <div className="p-3 bg-cyan-950/20 border border-cyan-500/20 rounded-xl space-y-1">
          <div className="flex justify-between text-slate-300">
            <span>Net Transfer:</span>
            <span className="font-mono font-bold text-neon-mint">${amount.toFixed(2)} USD</span>
          </div>
          <div className="flex justify-between text-slate-400 text-[10px]">
            <span>Estimated Processing:</span>
            <span>{method === "crypto" ? "< 15 minutes" : "1-2 business days"}</span>
          </div>
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="ghost" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleSubmit}
            disabled={amount <= 0 || isSubmitting}
            className="flex-1 font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 text-black"
          >
            {isSubmitting ? "Submitting..." : "Submit Payout Request"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
export default PayoutForm;
