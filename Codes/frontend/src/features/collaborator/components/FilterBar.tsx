import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { Slider } from "../../../components/ui/Slider";
import { MatchFilters } from "../types";

export function FilterBar({
  filters,
  onFilterChange,
  onApply,
}: {
  filters: MatchFilters;
  onFilterChange: (filters: MatchFilters) => void;
  onApply: () => void;
}) {
  const [localFilters, setLocalFilters] = useState(filters);
  return (
    <TerminalCard className="p-4">
      <div className="grid md:grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">
            Min Match Score
          </label>
          <Slider
            min={0}
            max={100}
            step={5}
            value={[localFilters.min_score || 0]}
            onValueChange={([val]) =>
              setLocalFilters({ ...localFilters, min_score: val })
            }
          />
        </div>
        <Input
          label="Expertise Tag"
          value={localFilters.expertise?.[0] || ""}
          onChange={(e) =>
            setLocalFilters({
              ...localFilters,
              expertise: e.target.value ? [e.target.value] : [],
            })
          }
        />
        <Select
          label="Availability"
          options={[
            { value: "", label: "All" },
            { value: "active", label: "Active" },
            { value: "busy", label: "Busy" },
            { value: "looking", label: "Looking" },
          ]}
          value={localFilters.availability || ""}
          onChange={(e) =>
            setLocalFilters({ ...localFilters, availability: e.target.value })
          }
        />
        <Select
          label="Sort by"
          options={[
            { value: "score", label: "Match Score" },
            { value: "name", label: "Name" },
            { value: "recent", label: "Most Recent" },
          ]}
          value={localFilters.sort_by || "score"}
          onChange={(e) =>
            setLocalFilters({ ...localFilters, sort_by: e.target.value as any })
          }
        />
      </div>
      <div className="flex gap-2 mt-4">
        <Button
          size="sm"
          onClick={() => {
            onFilterChange(localFilters);
            onApply();
          }}
        >
          Apply Filters
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setLocalFilters({});
            onFilterChange({});
            onApply();
          }}
        >
          Reset
        </Button>
      </div>
    </TerminalCard>
  );
}
