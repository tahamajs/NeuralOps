import { useEffect, useRef } from 'react'

export const LiveLogViewer = ({ logs }: { logs: string[] }) => {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs])

  return (
    <div
      ref={ref}
      className="bg-cyber-800 rounded-lg p-3 text-green-400 font-mono text-xs h-64 overflow-y-auto scrollbar-thin"
    >
      {logs.length === 0 && <span className="text-slate-500">Waiting for logs...</span>}
      {logs.map((line, i) => (
        <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
      ))}
    </div>
  )
}