import { useState } from "react";
import { Card, Select } from "../../../components/ui";

interface ContributionWeek {
  days: { date: string; count: number; level: 0 | 1 | 2 | 3 | 4 }[];
}

interface ActivityHeatmapProps {
  data?: {
    weeks: ContributionWeek[];
    total: number;
    range: { start: string; end: string };
  };
}

export function ActivityHeatmap({ data }: ActivityHeatmapProps) {
  const [year, setYear] = useState(new Date().getFullYear());
  if (!data)
    return (
      <div className="text-center text-slate-400 py-12">
        No contribution data available.
      </div>
    );

  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const getColor = (level: number) => {
    switch (level) {
      case 1:
        return "bg-green-900";
      case 2:
        return "bg-green-700";
      case 3:
        return "bg-green-500";
      case 4:
        return "bg-green-300";
      default:
        return "bg-white/5";
    }
  };

  return (
    <Card className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-bold text-white">Activity Calendar</h3>
        <Select
          options={[
            { value: "2024", label: "2024" },
            { value: "2025", label: "2025" },
          ]}
          value={String(year)}
          onChange={(e) => setYear(parseInt(e.target.value))}
          className="w-28"
        />
      </div>
      <div className="overflow-x-auto">
        <div className="flex gap-1">
          {data.weeks.map((week, weekIdx) => (
            <div key={weekIdx} className="flex flex-col gap-1">
              {week.days.map((day, dayIdx) => (
                <div
                  key={dayIdx}
                  className={`w-3 h-3 rounded-sm ${getColor(day.level)}`}
                  title={`${day.date}: ${day.count} contributions`}
                />
              ))}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-slate-500">
          {months.map((month) => (
            <span key={month}>{month}</span>
          ))}
        </div>
      </div>
      <div className="flex justify-end items-center gap-2 mt-4 text-xs text-slate-500">
        <span>Less</span>
        <div className="flex gap-1">
          <div className="w-3 h-3 rounded-sm bg-white/5" />
          <div className="w-3 h-3 rounded-sm bg-green-900" />
          <div className="w-3 h-3 rounded-sm bg-green-700" />
          <div className="w-3 h-3 rounded-sm bg-green-500" />
          <div className="w-3 h-3 rounded-sm bg-green-300" />
        </div>
        <span>More</span>
      </div>
      <div className="mt-3 text-center text-sm text-slate-400">
        {data.total.toLocaleString()} contributions in the last year
      </div>
    </Card>
  );
}
