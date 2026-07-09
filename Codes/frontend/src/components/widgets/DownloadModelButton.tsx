import { Download } from "lucide-react";
import { Button } from "../ui/Button";

export function DownloadModelButton({ runId }: { runId?: string }) {
  const handleDownload = () => {
    if (runId) {
      window.open(`/runs/${runId}/model/`, "_blank");
    }
  };

  return (
    <Button
      variant="primary"
      size="sm"
      onClick={handleDownload}
      disabled={!runId}
    >
      <Download size={16} className="mr-1" />
      Download Model
    </Button>
  );
}
