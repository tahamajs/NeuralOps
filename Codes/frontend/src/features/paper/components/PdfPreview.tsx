import { useState, useEffect } from "react";
import { Document, Page } from "react-pdf";

interface PdfPreviewProps {
  pdfUrl: string | null;
  isLoading: boolean;
  error: string | null;
}

export function PdfPreview({ pdfUrl, isLoading, error }: PdfPreviewProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(1);

  if (isLoading) return <div className="text-center py-4">Loading PDF...</div>;
  if (error)
    return <div className="text-center py-4 text-red-400">{error}</div>;
  if (!pdfUrl)
    return (
      <div className="text-center py-4 text-slate-500">
        No PDF preview available
      </div>
    );

  return (
    <div className="flex flex-col items-center">
      <Document
        file={pdfUrl}
        onLoadSuccess={({ numPages }) => setNumPages(numPages)}
        className="w-full"
      >
        <Page
          pageNumber={pageNumber}
          renderTextLayer={false}
          renderAnnotationLayer={false}
          className="shadow-lg rounded-lg"
        />
      </Document>
      {numPages && numPages > 1 && (
        <div className="flex gap-2 mt-3">
          <button
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            disabled={pageNumber === 1}
            className="px-3 py-1 rounded bg-white/5 hover:bg-white/10 disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-slate-400">
            Page {pageNumber} of {numPages}
          </span>
          <button
            onClick={() => setPageNumber((p) => Math.min(numPages, p + 1))}
            disabled={pageNumber === numPages}
            className="px-3 py-1 rounded bg-white/5 hover:bg-white/10 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
