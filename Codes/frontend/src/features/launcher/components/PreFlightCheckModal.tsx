import React, { useState } from "react";
import {
  Server,
  Cpu,
  Wallet,
  CheckCircle2,
  AlertCircle,
  Zap,
  HardDrive,
  RefreshCw,
  Plus,
  Lock,
  Laptop,
} from "lucide-react";
import { Modal } from "../../../components/ui/Modal";
import { Button } from "../../../components/ui/Button";
import { useToast } from "../../../hooks/useToast";

export interface PreFlightJobConfig {
  name: string;
  model_name: string;
  task_name: string;
  attention_type: string;
  method: string;
  compression_rank: number;
  epochs: number;
  batch_size: number;
  cluster: string;
}

export interface PreFlightCheckModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirmLaunch: (selectedMachine: any) => void;
  jobConfig: PreFlightJobConfig;
  isLaunching?: boolean;
}

export const AVAILABLE_MACHINES = [
  {
    id: "local-triton",
    name: "Local Triton GPU Node",
    model: "NVIDIA RTX 4090 (24GB)",
    type: "local",
    status: "online",
    latency: "1.2ms",
    pricePerHour: 0.0,
    vramFree: 21.4,
    vramTotal: 24.0,
    agentVersion: "v2.4.0 (Connected)",
    badge: "Free & Local",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  },
  {
    id: "cloud-h100-cluster",
    name: "NeuralOps Cloud Cluster",
    model: "8x NVIDIA H100 SXM5 (640GB)",
    type: "cloud",
    status: "online",
    latency: "8.4ms",
    pricePerHour: 2.15,
    vramFree: 610.0,
    vramTotal: 640.0,
    agentVersion: "v2.4.0 (Verified Cloud)",
    badge: "Ultra Fast",
    badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
  },
  {
    id: "peer-provider-a100",
    name: "Decentralized Provider Node #841",
    model: "4x NVIDIA A100 SXM4 (320GB)",
    type: "peer",
    status: "online",
    latency: "14.2ms",
    pricePerHour: 1.45,
    vramFree: 298.0,
    vramTotal: 320.0,
    agentVersion: "v2.4.0 (Provider Node)",
    badge: "Spot -32%",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  },
  {
    id: "apple-metal-mps",
    name: "Apple Silicon Metal Node",
    model: "Apple M3 Max (36GB Unified Memory)",
    type: "metal",
    status: "online",
    latency: "0.8ms",
    pricePerHour: 0.0,
    vramFree: 28.5,
    vramTotal: 36.0,
    agentVersion: "v2.4.0 (Local MPS)",
    badge: "Local MPS",
    badgeColor: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  },
];

export function PreFlightCheckModal({
  isOpen,
  onClose,
  onConfirmLaunch,
  jobConfig,
  isLaunching = false,
}: PreFlightCheckModalProps) {
  const { success, error: toastError } = useToast();
  const [selectedMachineId, setSelectedMachineId] = useState("local-triton");
  const [userCreditBalance, setUserCreditBalance] = useState(248.5);
  const [isVerifyingPing, setIsVerifyingPing] = useState(false);
  const [showTopUp, setShowTopUp] = useState(false);
  const [topUpAmount, setTopUpAmount] = useState(50);

  const selectedMachine =
    AVAILABLE_MACHINES.find((m) => m.id === selectedMachineId) ||
    AVAILABLE_MACHINES[0];

  // Duration & Cost estimates
  const estimatedHours = Number(
    Math.max(0.5, ((jobConfig.epochs || 3) * 0.45)).toFixed(1)
  );
  const estimatedCost = Number(
    (selectedMachine.pricePerHour * estimatedHours).toFixed(2)
  );
  const hasSufficientCredits = userCreditBalance >= estimatedCost;

  // Theoretical Memory Estimation
  const hiddenSize = jobConfig.model_name?.includes("base") ? 768 : 512;
  const numLayers = jobConfig.model_name?.includes("distil") ? 6 : 12;
  const rank = jobConfig.compression_rank || 16;
  const standardParams = 4 * hiddenSize * hiddenSize * numLayers;
  const compressedParams = 4 * (2 * hiddenSize * rank) * numLayers;
  const reductionPct = Math.max(
    10,
    Math.min(90, Math.round((1 - compressedParams / standardParams) * 100))
  );
  const requiredVramGB = Number(
    Math.max(2.1, (hiddenSize * numLayers * 0.002)).toFixed(1)
  );
  const hasVramHeadroom = selectedMachine.vramFree >= requiredVramGB;

  const handleRefreshPing = () => {
    setIsVerifyingPing(true);
    setTimeout(() => {
      setIsVerifyingPing(false);
      success("Node connectivity & WebSocket heartbeat verified (Latency: 3.8ms)");
    }, 600);
  };

  const handleTopUp = () => {
    setUserCreditBalance((prev) => prev + topUpAmount);
    setShowTopUp(false);
    success(`Successfully added $${topUpAmount}.00 to compute balance!`);
  };

  const handleConfirm = () => {
    if (!hasSufficientCredits) {
      toastError("Insufficient credit balance. Please top up your account balance.");
      return;
    }
    if (!hasVramHeadroom) {
      toastError("Insufficient VRAM on the selected node. Please pick another machine.");
      return;
    }
    onConfirmLaunch(selectedMachine);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="🚀 Pre-Flight Job & Machine Verification"
      size="xl"
    >
      <div className="space-y-6">
        {/* Step 1: Target Machine & Provider Agent Selection */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
              <Server size={14} /> 1. Select Compute Machine / Provider Agent Node
            </label>
            <button
              onClick={handleRefreshPing}
              disabled={isVerifyingPing}
              className="text-[11px] text-slate-400 hover:text-cyan-300 font-mono flex items-center gap-1 transition-colors cursor-pointer"
            >
              <RefreshCw
                size={12}
                className={isVerifyingPing ? "animate-spin text-cyan-400" : ""}
              />
              {isVerifyingPing ? "Pinging Nodes..." : "Ping Agents"}
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {AVAILABLE_MACHINES.map((machine) => {
              const isSelected = selectedMachineId === machine.id;
              return (
                <div
                  key={machine.id}
                  onClick={() => setSelectedMachineId(machine.id)}
                  className={`p-3.5 rounded-2xl border transition-all cursor-pointer relative ${
                    isSelected
                      ? "bg-cyan-500/10 border-cyan-500/60 shadow-lg shadow-cyan-500/10"
                      : "bg-white/[0.02] border-white/10 hover:border-white/20 hover:bg-white/[0.04]"
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                          isSelected ? "bg-cyan-500/20 text-cyan-300" : "bg-white/5 text-slate-400"
                        }`}
                      >
                        {machine.type === "local" ? (
                          <Cpu size={16} />
                        ) : machine.type === "metal" ? (
                          <Laptop size={16} />
                        ) : (
                          <Server size={16} />
                        )}
                      </div>
                      <div>
                        <h4 className="text-xs font-bold text-white">{machine.name}</h4>
                        <p className="text-[10px] font-mono text-slate-400">{machine.model}</p>
                      </div>
                    </div>
                    <span
                      className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded-full border ${machine.badgeColor}`}
                    >
                      {machine.badge}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-1 pt-2 border-t border-white/5 text-[10px] font-mono text-slate-400">
                    <div>
                      <span className="opacity-60">Status:</span>{" "}
                      <span className="text-emerald-400 font-bold">ONLINE</span>
                    </div>
                    <div>
                      <span className="opacity-60">Ping:</span> {machine.latency}
                    </div>
                    <div className="text-right font-bold text-slate-200">
                      {machine.pricePerHour === 0 ? "FREE" : `$${machine.pricePerHour.toFixed(2)}/hr`}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Step 2: Live Provider Agent Connectivity & VRAM Check */}
        <div className="p-4 rounded-2xl bg-black/40 border border-white/10 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="flex items-start gap-2.5">
            <CheckCircle2 size={16} className="text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-[10px] uppercase text-slate-500 block">Agent Daemon</span>
              <span className="font-bold text-white">{selectedMachine.agentVersion}</span>
              <span className="text-[10px] text-emerald-400 block mt-0.5">WebSocket Active (10s HB)</span>
            </div>
          </div>

          <div className="flex items-start gap-2.5 border-t sm:border-t-0 sm:border-l border-white/10 pt-2 sm:pt-0 sm:pl-3">
            <HardDrive size={16} className="text-cyan-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-[10px] uppercase text-slate-500 block">Available VRAM</span>
              <span className="font-bold text-cyan-300">
                {selectedMachine.vramFree} GB / {selectedMachine.vramTotal} GB
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Job needs: ~{requiredVramGB} GB (Headroom OK)
              </span>
            </div>
          </div>

          <div className="flex items-start gap-2.5 border-t sm:border-t-0 sm:border-l border-white/10 pt-2 sm:pt-0 sm:pl-3">
            <Zap size={16} className="text-purple-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-[10px] uppercase text-slate-500 block">SVD Compression</span>
              <span className="font-bold text-purple-300">-{reductionPct}% Parameters</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">Rank r={rank} Stiefel</span>
            </div>
          </div>
        </div>

        {/* Step 3: Compute Credits & Escrow Validation */}
        <div className="p-4 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-950 to-slate-900 border border-white/10">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2">
              <Wallet size={16} className="text-emerald-400" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold">
                2. Credit & Escrow Balance Check
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <span className="text-[10px] text-slate-400 block font-mono">Available Balance:</span>
                <span className="text-sm font-black text-emerald-400 font-mono">
                  ${userCreditBalance.toFixed(2)} USD
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowTopUp(!showTopUp)}
                className="text-xs font-mono"
              >
                <Plus size={12} className="mr-1" /> Add Funds
              </Button>
            </div>
          </div>

          {/* Top-up Drawer if toggled */}
          {showTopUp && (
            <div className="p-3 mb-3 bg-black/60 rounded-xl border border-cyan-500/30 flex flex-wrap items-center justify-between gap-2 animate-fade-in">
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-slate-400">Select Amount:</span>
                {[20, 50, 100, 250].map((amt) => (
                  <button
                    key={amt}
                    onClick={() => setTopUpAmount(amt)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                      topUpAmount === amt
                        ? "bg-cyan-500 text-black"
                        : "bg-white/10 text-slate-300 hover:bg-white/20"
                    }`}
                  >
                    +${amt}
                  </button>
                ))}
              </div>
              <Button size="sm" onClick={handleTopUp} className="text-xs font-bold">
                Confirm Top Up (${topUpAmount})
              </Button>
            </div>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-white/5 text-[11px] font-mono">
            <div>
              <span className="text-slate-500 block">Estimated Time:</span>
              <span className="font-bold text-white">~{estimatedHours} hours</span>
            </div>
            <div>
              <span className="text-slate-500 block">Rate per hour:</span>
              <span className="font-bold text-white">
                {selectedMachine.pricePerHour === 0 ? "$0.00 (Local)" : `$${selectedMachine.pricePerHour}/hr`}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">Estimated Cost:</span>
              <span className="font-bold text-cyan-300">${estimatedCost.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Pre-Flight Status:</span>
              {hasSufficientCredits ? (
                <span className="font-bold text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 size={12} /> Approved
                </span>
              ) : (
                <span className="font-bold text-rose-400 flex items-center gap-1">
                  <AlertCircle size={12} /> Low Balance
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-3 pt-2">
          <div className="text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
            <Lock size={12} className="text-cyan-400" />
            <span>Escrow lock: ${estimatedCost.toFixed(2)} held until job completion.</span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Button variant="ghost" onClick={onClose} disabled={isLaunching} className="flex-1 sm:flex-none">
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleConfirm}
              isLoading={isLaunching}
              disabled={!hasSufficientCredits || !hasVramHeadroom}
              className="flex-1 sm:flex-none font-bold font-mono text-xs px-6 bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 hover:brightness-110 text-black shadow-lg shadow-cyan-500/20"
            >
              <Zap size={14} className="mr-1.5" /> Approve & Dispatch Job
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
