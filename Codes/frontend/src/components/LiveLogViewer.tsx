import { useEffect, useRef } from 'react'

export default function LiveLogViewer({ logs }: { logs: string[] }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs])

  return (
    <div className="bg-gray-900 rounded-lg shadow p-4 text-green-400 font-mono text-sm h-64 overflow-y-auto" ref={ref}>
      {logs.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  )
}