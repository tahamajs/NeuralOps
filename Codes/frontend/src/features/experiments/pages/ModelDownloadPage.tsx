import { useParams } from "react-router-dom";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";

export default function ModelDownloadPage() {
  const { runId } = useParams<{ runId: string }>();
  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="text-center space-y-4 max-w-md w-full">
        <h1 className="text-2xl font-black text-white">
          MODEL <span className="text-neon-blue">DOWNLOAD</span>
        </h1>
        <p className="text-slate-400 text-sm">
          Download the trained model as a ZIP archive.
        </p>
        <a href={`/runs/${runId}/model/`} download>
          <Button>Download Model</Button>
        </a>
      </Card>
    </div>
  );
}
