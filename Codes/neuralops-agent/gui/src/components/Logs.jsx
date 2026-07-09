import React, { useEffect, useRef, useState } from 'react';
import { ScrollText } from 'lucide-react';

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  useEffect(() => {
    if (window.neuralopsAPI) {
      window.neuralopsAPI.onAgentLog((msg) => {
        setLogs((prev) => [...prev.slice(-1000), { text: msg, type: 'info', time: new Date().toLocaleTimeString() }]);
      });
      window.neuralopsAPI.onAgentError((msg) => {
        setLogs((prev) => [...prev.slice(-1000), { text: msg, type: 'error', time: new Date().toLocaleTimeString() }]);
      });
    }
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <ScrollText size={24} />
        <h2 className="text-2xl font-bold">Agent Logs</h2>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-xl p-4 h-[500px] overflow-y-auto font-mono text-sm">
        {logs.length === 0 && <p className="text-gray-500">Agent is not running. Start the agent to see logs.</p>}
        {logs.map((log, i) => (
          <div key={i} className={`flex gap-2 py-0.5 ${log.type === 'error' ? 'text-red-400' : 'text-gray-300'}`}>
            <span className="text-gray-500 shrink-0">{log.time}</span>
            <span className="whitespace-pre-wrap">{log.text}</span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}