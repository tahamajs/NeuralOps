import React from "react";

export function TableWidget({ data, config }: any) {
  if (!data || data.length === 0) {
    return <div className="text-slate-500 text-xs p-4">No data available</div>;
  }

  const keys = Object.keys(data[0] || {}).filter((k) => k !== "id");

  return (
    <div className="overflow-x-auto w-full h-full max-h-[220px]">
      <table className="min-w-full text-xs text-left text-slate-300">
        <thead className="bg-slate-900/50 uppercase tracking-wider text-[10px] text-slate-400">
          <tr>
            {keys.map((k) => (
              <th key={k} className="px-3 py-2 border-b border-slate-800">
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row: any, i: number) => (
            <tr key={i} className="hover:bg-slate-800/30">
              {keys.map((k) => (
                <td
                  key={k}
                  className="px-3 py-1.5 border-b border-slate-800/50"
                >
                  {typeof row[k] === "object"
                    ? JSON.stringify(row[k])
                    : String(row[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
