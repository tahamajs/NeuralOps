import { useParams } from "react-router-dom";
import { FLDashboard } from "../components/FLDashboard";

export default function FLRunPage({ onClose }: { onClose?: () => void }) {
  const { runId } = useParams();
  if (!runId) return <div className="text-red-400">No run ID provided</div>;
  return <FLDashboard runId={runId} onClose={onClose} />;
}
