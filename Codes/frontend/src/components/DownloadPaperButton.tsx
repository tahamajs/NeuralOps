import { useState } from "react";

export default function DownloadPaperButton({ experimentId }: { experimentId: string }) {
  const [loading, setLoading] = useState(false);

  const downloadPaper = async () => {
    setLoading(true);
    try {
      const pdfUrl = `/api/experiments/${experimentId}/paper/`;

      // Try to fetch as blob to trigger download
      const response = await fetch(pdfUrl);
      if (!response.ok) throw new Error("Paper generation failed");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;

      // Determine file extension from content type
      const contentType = response.headers.get("content-type") || "";
      const ext = contentType.includes("pdf") ? "pdf" : "tex";
      link.download = `paper_${experimentId}.${ext}`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download paper:", error);
      alert("Paper generation failed. Ensure pdflatex is installed on the server, or check the logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Paper</h2>
      <button
        onClick={downloadPaper}
        disabled={loading}
        className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 disabled:opacity-50"
      >
        {loading ? "Generating..." : "📄 Generate Paper (LaTeX)"}
      </button>
      <p className="text-xs text-gray-500 mt-1">
        Compiles a PDF if pdflatex is installed; otherwise downloads a .tex file.
      </p>
    </div>
  );
}