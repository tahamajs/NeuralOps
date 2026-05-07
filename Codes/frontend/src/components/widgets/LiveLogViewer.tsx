import { useEffect, useRef } from 'react'
import { ScrollText } from 'lucide-react'

export const LiveLogViewer = ({ logs }: { logs: string[] }) => {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs])

  return (
    <div className="relative">
      {logs.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="flex flex-col items-center text-slate-600">
            <ScrollText size={24} className="mb-1" />
            <span className="text-xs">Waiting for logs...</span>
          </div>
        </div>
      )}
      <div
        ref={ref}
        className="bg-cyber-800 rounded-lg p-3 text-green-400 font-mono text-xs h-64 overflow-y-auto scrollbar-thin"
      >
        {logs.length > 0 ? (
          logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">
              {line}
            </div>
          ))
        ) : (
          <div className="h-full" /> // empty spacer to keep the height
        )}
      </div>
    </div>
  )
}