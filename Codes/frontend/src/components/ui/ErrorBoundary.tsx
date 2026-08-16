import React, { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "./Button";
import { AlertTriangle, RefreshCw, Home, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
  title?: string;
  className?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  reset = () => {
    this.props.onReset?.();
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div
          className={`flex flex-col items-center justify-center min-h-[350px] p-6 text-center animate-fade-in ${
            this.props.className || ""
          }`}
        >
          <div className="glass p-8 rounded-2xl max-w-lg w-full border border-red-500/20 shadow-2xl shadow-red-500/5">
            <div className="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto mb-4 text-red-400">
              <AlertTriangle size={24} />
            </div>

            <h2 className="text-xl font-bold text-white mb-2">
              {this.props.title || "Something went wrong"}
            </h2>

            <p className="text-slate-400 text-sm mb-6 leading-relaxed">
              {this.state.error?.message ||
                "An unexpected rendering error occurred in this view."}
            </p>

            {this.state.error?.stack && (
              <details className="mb-6 text-left bg-black/40 border border-white/5 rounded-xl p-3">
                <summary className="text-xs text-slate-400 font-mono cursor-pointer hover:text-white select-none">
                  View error details
                </summary>
                <pre className="mt-2 text-[11px] font-mono text-red-300/80 overflow-x-auto whitespace-pre-wrap max-h-36">
                  {this.state.error.stack}
                </pre>
              </details>
            )}

            <div className="flex flex-wrap items-center justify-center gap-3">
              <Button
                variant="primary"
                size="sm"
                icon={<RotateCcw size={14} />}
                onClick={this.reset}
              >
                Try Again
              </Button>
              <Button
                variant="secondary"
                size="sm"
                icon={<Home size={14} />}
                onClick={() => {
                  window.location.href = "/dashboard";
                }}
              >
                Dashboard
              </Button>
              <Button
                variant="ghost"
                size="sm"
                icon={<RefreshCw size={14} />}
                onClick={() => window.location.reload()}
              >
                Reload
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
