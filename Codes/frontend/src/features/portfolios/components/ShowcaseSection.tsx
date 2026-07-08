import { Card, Badge } from "../../../components/ui";
import { GitBranch, FileText, Library, StickyNote, Star } from "lucide-react";
import type { ShowcaseItem } from "../types";

interface ShowcaseSectionProps {
  items: ShowcaseItem[];
}

export function ShowcaseSection({ items }: ShowcaseSectionProps) {
  if (!items.length)
    return (
      <div className="text-center text-slate-400 py-12">
        No showcased items yet.
      </div>
    );

  const getIcon = (type: ShowcaseItem["type"]) => {
    switch (type) {
      case "experiment":
        return <GitBranch size={20} className="text-neon-blue" />;
      case "paper":
        return <FileText size={20} className="text-pink-400" />;
      case "model":
        return <Library size={20} className="text-purple-400" />;
      case "note":
        return <StickyNote size={20} className="text-yellow-400" />;
      default:
        return null;
    }
  };

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
      {items.map((item) => (
        <Card key={item.id} hover className="overflow-hidden group">
          {item.thumbnail_url && (
            <img
              src={item.thumbnail_url}
              alt={item.title}
              className="w-full h-32 object-cover"
            />
          )}
          <div className="p-4">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                {getIcon(item.type)}
                <h3 className="font-semibold text-white group-hover:text-neon-blue transition-colors">
                  {item.title}
                </h3>
              </div>
              {item.stats?.stars && (
                <Star size={14} className="text-yellow-400" />
              )}
            </div>
            <p className="text-sm text-slate-400 mt-1 line-clamp-2">
              {item.description}
            </p>
            <div className="flex flex-wrap gap-1 mt-2">
              {item.tags?.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded bg-neon-blue/10 text-neon-blue text-xs"
                >
                  #{tag}
                </span>
              ))}
            </div>
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-block text-xs text-neon-blue hover:underline"
            >
              View Details →
            </a>
          </div>
        </Card>
      ))}
    </div>
  );
}
