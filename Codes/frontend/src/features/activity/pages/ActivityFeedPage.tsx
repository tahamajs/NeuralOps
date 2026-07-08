import { useState } from "react";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Activity, Clock, FileText, FlaskConical, Play, Share2 } from "lucide-react";

export default function ActivityFeedPage() {
  const [filter, setFilter] = useState("all");

  const activities = [
    {
      id: 1,
      user: "System",
      action: "started training run",
      target: "bert-finetune-v2",
      time: "2 mins ago",
      type: "experiment",
      icon: <Play size={16} className="text-neon-blue" />
    },
    {
      id: 2,
      user: "Dr. Turing",
      action: "published a new note",
      target: "Attention Mechanisms Survey",
      time: "1 hour ago",
      type: "note",
      icon: <FileText size={16} className="text-purple-400" />
    },
    {
      id: 3,
      user: "Auto-Agent",
      action: "detected performance drift in",
      target: "prod-recommender-model",
      time: "3 hours ago",
      type: "alert",
      icon: <Activity size={16} className="text-red-400" />
    },
    {
      id: 4,
      user: "You",
      action: "completed evaluation on",
      target: "transformer-baseline",
      time: "Yesterday",
      type: "experiment",
      icon: <FlaskConical size={16} className="text-green-400" />
    }
  ];

  const filteredActivities = filter === "all" ? activities : activities.filter(a => a.type === filter);

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
            <Activity className="text-neon-blue" size={24} /> Global Activity Feed
          </h1>
          <p className="text-slate-400 mt-1">Real-time updates across the NeuralOps platform</p>
        </div>
        <div className="flex bg-white/5 rounded-lg p-1 border border-white/10">
          {["all", "experiment", "note", "alert"].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${filter === f ? 'bg-white/10 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <Card className="glass p-1">
        <div className="divide-y divide-white/5">
          {filteredActivities.map((item) => (
            <div key={item.id} className="p-4 hover:bg-white/5 transition-colors flex gap-4 items-start group">
              <div className="mt-1 w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
                {item.icon}
              </div>
              <div className="flex-1">
                <p className="text-sm text-slate-200">
                  <span className="font-semibold text-white">{item.user}</span>{" "}
                  {item.action}{" "}
                  <span className="font-mono text-neon-blue bg-neon-blue/10 px-1.5 py-0.5 rounded text-xs">{item.target}</span>
                </p>
                <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                  <span className="flex items-center gap-1"><Clock size={12} /> {item.time}</span>
                  <button className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity hover:text-white">
                    <Share2 size={12} /> Share
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
