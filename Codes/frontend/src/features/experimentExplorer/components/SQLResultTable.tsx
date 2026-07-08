// src/features/experimentExplorer/components/SQLResultTable.tsx
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-alpine.css";

export function SQLResultTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Record<string, any>[];
}) {
  const colDefs = columns.map((col) => ({
    field: col,
    headerName: col,
    flex: 1,
    minWidth: 120,
    resizable: true,
  }));
  const rowData = rows.map((row, idx) => ({ ...row, id: idx }));

  return (
    <div className="h-96 w-full ag-theme-alpine-dark">
      <AgGridReact
        columnDefs={colDefs}
        rowData={rowData}
        pagination={true}
        paginationPageSize={50}
        domLayout="autoHeight"
      />
    </div>
  );
}
