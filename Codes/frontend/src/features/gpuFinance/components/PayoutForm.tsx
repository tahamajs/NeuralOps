import React, { useState } from "react";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Wallet, DollarSign, ArrowRight, ShieldCheck } from "lucide-react";

export function PayoutForm() {
  const [amount, setAmount] = useState("");
  const [address, setAddress] = useState("");
  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState(false);

  const availableBalance = 1250.45;

  const handleWithdraw = (e: React.FormEvent) => {
    e.preventDefault();
    if (!amount || !address) return;

    setProcessing(true);
    setTimeout(() => {
      setProcessing(false);
      setSuccess(true);
      setAmount("");
      setAddress("");
      setTimeout(() => setSuccess(false), 3000);
    }, 1500);
  };

  return (
    <Card className="glass border-white/5 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-neon-blue/10 flex items-center justify-center">
          <Wallet className="text-neon-blue w-5 h-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">
            Withdraw Earnings
          </h3>
          <p className="text-sm text-slate-400">
            Transfer your GPU hosting earnings to a crypto wallet
          </p>
        </div>
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-4 mb-6 flex justify-between items-center">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">
            Available Balance
          </p>
          <p className="text-2xl font-display font-bold text-white flex items-center">
            <DollarSign size={20} className="text-neon-blue mr-1" />
            {availableBalance.toFixed(2)}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="text-xs"
          onClick={() => setAmount(availableBalance.toString())}
        >
          Max
        </Button>
      </div>

      <form onSubmit={handleWithdraw} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1.5">
            Amount to Withdraw
          </label>
          <div className="relative">
            <DollarSign
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
              size={16}
            />
            <Input
              type="number"
              step="0.01"
              max={availableBalance}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="pl-9"
              placeholder="0.00"
              required
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1.5">
            Destination Wallet (USDC/ERC20)
          </label>
          <Input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="0x..."
            required
            pattern="^0x[a-fA-F0-9]{40}$"
          />
        </div>

        <div className="pt-2">
          <Button
            type="submit"
            className="w-full bg-gradient-to-r from-neon-blue to-purple-600 border-0 flex items-center justify-center gap-2"
            disabled={
              processing ||
              !amount ||
              !address ||
              parseFloat(amount) > availableBalance
            }
          >
            {processing ? (
              "Processing..."
            ) : success ? (
              <>
                <ShieldCheck size={18} /> Withdrawal Initiated
              </>
            ) : (
              <>
                Withdraw Funds <ArrowRight size={18} />
              </>
            )}
          </Button>
        </div>

        <p className="text-xs text-slate-500 text-center mt-4">
          Withdrawals are processed on the Ethereum network. Standard network
          fees apply.
        </p>
      </form>
    </Card>
  );
}
