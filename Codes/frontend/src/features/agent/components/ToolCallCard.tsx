// ToolCallCard — renders an agent tool call in the chat with collapsible result
import { useState } from 'react';
import { ChevronDown, ChevronRight, Wrench, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import type { ToolCall } from '../types';

interface Props {
  toolCall: ToolCall;
}

export default function ToolCallCard({ toolCall }: Props) {
  const [expanded, setExpanded] = useState(false);
  const hasResult = toolCall.result !== undefined;
  const isError =
    hasResult &&
    typeof toolCall.result === 'object' &&
    toolCall.result !== null &&
    'error' in (toolCall.result as Record<string, unknown>);

  return (
    <div
      style={{
        background: 'rgba(99,102,241,0.08)',
        border: '1px solid rgba(99,102,241,0.2)',
        borderRadius: 10,
        overflow: 'hidden',
        marginBottom: 4,
        backdropFilter: 'blur(10px)',
      }}
    >
      {/* Header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        <Wrench size={13} color="#818cf8" />
        <span style={{ fontSize: 12, fontWeight: 700, color: '#818cf8', fontFamily: 'monospace', flex: 1 }}>
          {toolCall.name}
        </span>
        {!hasResult ? (
          <Loader2 size={13} color="#f59e0b" style={{ animation: 'spin 1s linear infinite' }} />
        ) : isError ? (
          <XCircle size={13} color="#ef4444" />
        ) : (
          <CheckCircle2 size={13} color="#22c55e" />
        )}
        {expanded ? (
          <ChevronDown size={13} color="#64748b" />
        ) : (
          <ChevronRight size={13} color="#64748b" />
        )}
      </button>

      {/* Body */}
      {expanded && (
        <div style={{ padding: '0 12px 10px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          {/* Args */}
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', letterSpacing: '0.08em', marginBottom: 4 }}>
              INPUT
            </div>
            <pre
              style={{
                background: 'rgba(0,0,0,0.3)',
                borderRadius: 6,
                padding: '6px 8px',
                fontSize: 11,
                color: '#c4b5fd',
                overflow: 'auto',
                maxHeight: 150,
                fontFamily: 'monospace',
                margin: 0,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {JSON.stringify(toolCall.args, null, 2)}
            </pre>
          </div>

          {/* Result */}
          {hasResult && (
            <div style={{ marginTop: 8 }}>
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color: isError ? '#ef4444' : '#64748b',
                  letterSpacing: '0.08em',
                  marginBottom: 4,
                }}
              >
                {isError ? 'ERROR' : 'OUTPUT'}
              </div>
              <pre
                style={{
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: 6,
                  padding: '6px 8px',
                  fontSize: 11,
                  color: isError ? '#fca5a5' : '#86efac',
                  overflow: 'auto',
                  maxHeight: 200,
                  fontFamily: 'monospace',
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {typeof toolCall.result === 'string'
                  ? toolCall.result
                  : JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
