import { FileText } from "lucide-react";
import { Button } from "../ui/Button";

export function DownloadPaperButton({ runId }: { runId?: string }) {
  const handleDownload = () => {
    if (runId) {
      window.open(`/api/runs/${runId}/paper/`, "_blank");
    }
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={handleDownload}
      disabled={!runId}
    >
      <FileText size={16} className="mr-1" />
      Download Paper
    </Button>
  );
}
