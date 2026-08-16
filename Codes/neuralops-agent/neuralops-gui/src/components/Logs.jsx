import React, { useEffect, useRef, useState } from 'react';
import { ScrollText, Trash2, Copy, Check, Download, ArrowDownCircle } from 'lucide-react';

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const [filter, setFilter] = useState('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  const logEndRef = useRef(null);

  useEffect(() => {
    let unLog = null;
    let unErr = null;

    if (window.neuralopsAPI) {
      unLog = window.neuralopsAPI.onAgentLog((msg) => {
        setLogs((prev) => [
          ...prev.slice(-2000),
          { text: msg, type: 'info', time: new Date().toLocaleTimeString() },
        ]);
      });

      unErr = window.neuralopsAPI.onAgentError((msg) => {
        setLogs((prev) => [
          ...prev.slice(-2000),
          { text: msg, type: 'error', time: new Date().toLocaleTimeString() },
        ]);
      });
    }

    return () => {
      if (unLog) unLog();
      if (unErr) unErr();
    };
  }, []);

  useEffect(() => {
    if (autoScroll) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const clearLogs = () => setLogs([]);

  const copyLogs = () => {
    const text = logs.map((l) => `[${l.time}] [${l.type.toUpperCase()}] ${l.text}`).join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const exportLogs = () => {
    const text = logs.map((l) => `[${l.time}] [${l.type.toUpperCase()}] ${l.text}`).join('\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuralops-agent-logs-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredLogs = logs.filter((l) => {
    if (filter === 'all') return true;
    return l.type === filter;
  });

  return (
    <div className="space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-primary-600/10 text-primary-400 border border-primary-500/20">
            <ScrollText size={20} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Agent Output Stream</h2>
            <p className="text-xs text-gray-400">Real-time stdout/stderr log feed from the provider core</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Level Filter */}
          <div className="flex bg-dark-card border border-dark-border rounded-xl p-1 text-xs">
            {['all', 'info', 'error'].map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilter(lvl)}
                className={`px-3 py-1 rounded-lg uppercase text-[11px] font-semibold transition-colors ${
                  filter === lvl
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            title={autoScroll ? 'Disable Auto-scroll' : 'Enable Auto-scroll'}
            className={`p-2 rounded-xl border text-xs flex items-center gap-1 transition-colors ${
              autoScroll
                ? 'border-primary-500/40 bg-primary-600/20 text-primary-300'
                : 'border-dark-border bg-dark-card text-gray-400 hover:text-white'
            }`}
          >
            <ArrowDownCircle size={15} />
          </button>

          <button
            onClick={copyLogs}
            disabled={logs.length === 0}
            title="Copy Logs"
            className="p-2 rounded-xl border border-dark-border bg-dark-card hover:bg-dark-border text-gray-300 hover:text-white transition-colors disabled:opacity-40"
          >
            {copied ? <Check size={15} className="text-green-400" /> : <Copy size={15} />}
          </button>

          <button
            onClick={exportLogs}
            disabled={logs.length === 0}
            title="Export Logs"
            className="p-2 rounded-xl border border-dark-border bg-dark-card hover:bg-dark-border text-gray-300 hover:text-white transition-colors disabled:opacity-40"
          >
            <Download size={15} />
          </button>

          <button
            onClick={clearLogs}
            disabled={logs.length === 0}
            title="Clear Stream"
            className="p-2 rounded-xl border border-dark-border bg-dark-card hover:bg-rose-500/20 text-gray-300 hover:text-rose-300 transition-colors disabled:opacity-40"
          >
            <Trash2 size={15} />
          </button>
        </div>
      </div>

      {/* Log Terminal Window */}
      <div className="bg-black/90 border border-dark-border rounded-2xl p-4 h-[560px] overflow-y-auto font-mono text-xs shadow-2xl space-y-1">
        {logs.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-gray-600">
            <ScrollText size={36} className="mb-2 opacity-40" />
            <p className="text-gray-400 font-medium">Log stream is empty</p>
            <p className="text-[11px] text-gray-500 mt-1">
              Start the agent from the Dashboard to stream live worker output.
            </p>
          </div>
        )}

        {filteredLogs.map((log, idx) => (
          <div
            key={idx}
            className={`flex items-start gap-3 leading-relaxed ${
              log.type === 'error'
                ? 'text-rose-400 bg-rose-950/20 px-2 py-0.5 rounded'
                : 'text-gray-300'
            }`}
          >
            <span className="text-gray-500 select-none shrink-0 font-mono text-[10px] pt-0.5">
              {log.time}
            </span>
            <span
              className={`select-none text-[10px] font-bold uppercase px-1 py-0.2 rounded shrink-0 ${
                log.type === 'error'
                  ? 'bg-rose-500/20 text-rose-300'
                  : 'bg-emerald-500/10 text-emerald-400'
              }`}
            >
              {log.type}
            </span>
            <span className="whitespace-pre-wrap break-all flex-1 font-mono">{log.text}</span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}