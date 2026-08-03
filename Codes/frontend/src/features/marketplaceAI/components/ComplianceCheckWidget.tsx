import React from "react";
import { ComplianceCheck } from "../types";
import {
  ShieldCheck,
  ShieldAlert,
  Calendar,
  CheckCircle,
  XCircle,
} from "lucide-react";

interface ComplianceCheckWidgetProps {
  complianceUpdates: Record<string, ComplianceCheck>;
}

export function ComplianceCheckWidget({
  complianceUpdates,
}: ComplianceCheckWidgetProps) {
  const updates = Object.values(complianceUpdates);

  if (updates.length === 0) {
    return (
      <div className="glass p-6 rounded-lg text-center text-slate-400">
        <ShieldCheck className="mx-auto h-12 w-12 text-slate-600 mb-2" />
        <p className="font-semibold text-white">
          No Compliance Checks Performed Yet
        </p>
        <p className="text-sm mt-1">
          Request compliance checks on GPU listings to see audit results here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white mb-2">
        Compliance Audit Records
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {updates.map((check) => {
          const isCompliant = check.status === "compliant";
          return (
            <div
              key={check.id}
              className={`p-5 rounded-lg border bg-surface/50 backdrop-blur-md transition-all duration-300 hover:scale-[1.01] ${
                isCompliant
                  ? "border-green-500/30 shadow-[0_0_15px_rgba(34,197,94,0.05)]"
                  : "border-rose-500/30 shadow-[0_0_15px_rgba(244,63,94,0.05)]"
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                      isCompliant
                        ? "bg-green-500/10 text-green-400 border-green-500/20"
                        : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                    }`}
                  >
                    {isCompliant ? (
                      <ShieldCheck className="w-3.5 h-3.5 mr-1" />
                    ) : (
                      <ShieldAlert className="w-3.5 h-3.5 mr-1" />
                    )}
                    {isCompliant ? "COMPLIANT" : "NON-COMPLIANT"}
                  </span>
                  <h3 className="text-sm font-semibold text-slate-300 mt-2">
                    Listing ID: {check.listing_id.substring(0, 8)}...
                  </h3>
                </div>
                <div className="flex items-center text-xs text-slate-500">
                  <Calendar className="w-3.5 h-3.5 mr-1" />
                  {new Date(check.checked_at).toLocaleTimeString()}
                </div>
              </div>

              <div className="mt-4 space-y-3">
                {check.passed_checks.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-1">
                      Passed Checks
                    </h4>
                    <ul className="space-y-1">
                      {check.passed_checks.map((rule, idx) => (
                        <li
                          key={idx}
                          className="flex items-center text-xs text-slate-400"
                        >
                          <CheckCircle className="w-3.5 h-3.5 text-green-500/70 mr-1.5 shrink-0" />
                          {rule}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {check.failed_checks.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-rose-400 uppercase tracking-wider mb-1">
                      Failed Checks
                    </h4>
                    <ul className="space-y-1">
                      {check.failed_checks.map((rule, idx) => (
                        <li
                          key={idx}
                          className="flex items-center text-xs text-slate-400"
                        >
                          <XCircle className="w-3.5 h-3.5 text-rose-500/70 mr-1.5 shrink-0" />
                          {rule}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
