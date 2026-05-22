import { useEffect, useRef } from 'react'

export function LogWidget({ data }: any) {
  const containerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight
  }, [data])
  return (
    <div ref={containerRef} className="h-40 overflow-y-auto font-mono text-xs text-slate-300 bg-black/30 rounded p-2">
      {data.map((line: any, i: number) => (
        <div key={i}>{line.timestamp} - {line.message}</div>
      ))}
    </div>
  )
}