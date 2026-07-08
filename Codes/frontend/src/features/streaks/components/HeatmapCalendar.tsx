import { useState, useEffect } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Tooltip } from "../../../components/ui/Tooltip";
import {
  format,
  startOfWeek,
  addDays,
  eachWeekOfInterval,
  subWeeks,
  getISOWeek,
} from "date-fns";
import { HeatmapData, ContributionDay } from "../types";
import { Calendar } from "lucide-react";

interface HeatmapCalendarProps {
  data: HeatmapData;
  year: number;
}

export function HeatmapCalendar({ data, year }: HeatmapCalendarProps) {
  const [weeks, setWeeks] = useState<ContributionDay[][]>([]);
  const [monthLabels, setMonthLabels] = useState<
    { month: string; colSpan: number }[]
  >([]);

  useEffect(() => {
    if (data?.weeks) {
      setWeeks(data.weeks);
      const firstDate = new Date(year, 0, 1);
      const firstSunday = startOfWeek(firstDate, { weekStartsOn: 0 });
      const weeksInYear = eachWeekOfInterval(
        { start: firstSunday, end: new Date(year, 11, 31) },
        { weekStartsOn: 0 },
      );
      const months = weeksInYear.reduce(
        (acc, weekStart, idx) => {
          const month = weekStart.toLocaleString("default", { month: "short" });
          if (acc.length === 0 || acc[acc.length - 1].month !== month) {
            const colSpan = weeksInYear
              .slice(idx)
              .findIndex(
                (w) =>
                  w.toLocaleString("default", { month: "short" }) !== month,
              );
            acc.push({
              month,
              colSpan: colSpan === -1 ? weeksInYear.length - idx : colSpan,
            });
          }
          return acc;
        },
        [] as { month: string; colSpan: number }[],
      );
      setMonthLabels(months);
    }
  }, [data, year]);

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
    <TerminalCard className="p-4 overflow-x-auto">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-1">
          <Calendar size={16} className="text-neon-mint" />
          <h3 className="text-sm font-bold text-white">
            {year} Activity Calendar
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Less</span>
          <div className="flex gap-1">
            <div className="w-3 h-3 rounded-sm bg-white/5" />
            <div className="w-3 h-3 rounded-sm bg-green-900" />
            <div className="w-3 h-3 rounded-sm bg-green-700" />
            <div className="w-3 h-3 rounded-sm bg-green-500" />
            <div className="w-3 h-3 rounded-sm bg-green-300" />
          </div>
          <span className="text-slate-400">More</span>
        </div>
      </div>
      <div>
        <div className="flex gap-1 mb-1 text-xs text-slate-500 justify-around">
          {monthLabels.map((m) => (
            <span
              key={m.month}
              style={{ width: `${m.colSpan * 12}px` }}
              className="text-center"
            >
              {m.month}
            </span>
          ))}
        </div>
        <div className="flex gap-1">
          {weeks.map((week, wIdx) => (
            <div key={wIdx} className="flex flex-col gap-1">
              {week.map((day, dIdx) => (
                <Tooltip
                  key={dIdx}
                  content={`${day.date}: ${day.count} contributions`}
                >
                  <div
                    className={`w-3 h-3 rounded-sm ${getColor(day.level)} cursor-help`}
                  />
                </Tooltip>
              ))}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-4 text-xs text-slate-500">
          <div>Mon</div>
          <div>Wed</div>
          <div>Fri</div>
        </div>
      </div>
    </TerminalCard>
  );
}
