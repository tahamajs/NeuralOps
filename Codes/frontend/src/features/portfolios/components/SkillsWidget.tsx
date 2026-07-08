import { Card, Badge } from "../../../components/ui";
import type { Portfolio } from "../types";

interface SkillsWidgetProps {
  portfolio: Portfolio;
}

export function SkillsWidget({ portfolio }: SkillsWidgetProps) {
  return (
    <div className="space-y-6">
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          Areas of Expertise
        </h3>
        <div className="flex flex-wrap gap-2">
          {portfolio.expertise?.map((skill) => (
            <Badge key={skill}>{skill}</Badge>
          ))}
        </div>
      </Card>
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          Research Interests
        </h3>
        <div className="flex flex-wrap gap-2">
          {portfolio.research_interests?.map((interest) => (
            <Badge key={interest} variant="secondary">
              {interest}
            </Badge>
          ))}
        </div>
      </Card>
    </div>
  );
}
