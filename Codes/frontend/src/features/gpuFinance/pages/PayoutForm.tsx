import { useState } from "react";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { gpuFinanceAPI } from "../services/api";
import { useToast } from "../../../components/ui/Toast";

export function PayoutForm({ open, onClose, onSuccess }: any) {
  const [amount, setAmount] = useState(0);
  const [method, setMethod] = useState("bank_transfer");
  const { success, error } = useToast();

  const handleSubmit = async () => {
    try {
      await gpuFinanceAPI.requestPayout({ amount, method });
      success("Payout request submitted");
      onSuccess();
      onClose();
    } catch {
      error("Request failed");
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Request Payout">
      <div className="space-y-4">
        <Input
          label="Amount (USD)"
          type="number"
          value={amount}
          onChange={(e) => setAmount(parseFloat(e.target.value))}
        />
        <Select
          label="Payout Method"
          options={[
            { value: "bank_transfer", label: "Bank Transfer" },
            { value: "crypto", label: "Cryptocurrency" },
            { value: "paypal", label: "PayPal" },
          ]}
          value={method}
          onChange={(e) => setMethod(e.target.value)}
        />
        <Button onClick={handleSubmit} disabled={amount <= 0}>
          Submit Request
        </Button>
      </div>
    </Modal>
  );
}
