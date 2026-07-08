import React from "react";

interface PdfPreviewProps {
  pdfUrl: string | null;
  isLoading: boolean;
  error: string | null;
}

export function PdfPreview({ pdfUrl, isLoading, error }: PdfPreviewProps) {
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
    <div className="w-full h-full flex flex-col items-center">
      <iframe
        src={pdfUrl}
        className="w-full h-[600px] border border-white/10 rounded-lg shadow-lg bg-white"
        title="PDF Preview"
      />
    </div>
  );
}
