import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { communityAPI } from "../services/api";
import { Card, Spinner } from "../../../components/ui";
import { QuestionList } from "../components/QuestionList";

export default function TagDetailPage() {
  const { tagName } = useParams();
  const { data: tag, isLoading } = useQuery({
    queryKey: ["tag", tagName],
    queryFn: () => communityAPI.getTag(tagName!),
    enabled: !!tagName,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex items-center gap-3">
          <Badge className="text-lg px-4 py-2">#{tag?.name}</Badge>
          <span className="text-slate-400">{tag?.count} questions</span>
        </div>
        <p className="text-slate-300 mt-2">
          {tag?.description || "No description yet."}
        </p>
      </Card>
      <QuestionList tagFilter={tagName} />
    </div>
  );
}
