import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { reviewsAPI } from "../services/api";
import { ReviewRequestCard } from "../components/ReviewRequestCard";
import {
  Card,
  Button,
  Input,
  Select,
  Spinner,
  Tabs,
  TabList,
  Tab,
  TabPanel,
} from "../../../components/ui";
import {
  Plus,
  Search,
  Filter,
  ListTodo,
  CheckCircle,
  Clock,
  Award,
  TrendingUp,
} from "lucide-react";

export default function ReviewRequestListPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<string>("newest");

  const { data: requestsData, isLoading: requestsLoading } = useQuery({
    queryKey: [
      "reviewRequests",
      { search, status: statusFilter, sort: sortBy },
    ],
    queryFn: () =>
      reviewsAPI.getRequests({
        search,
        status: statusFilter === "all" ? undefined : statusFilter,
        sort: sortBy,
      }),
  });

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["reviewStats"],
    queryFn: () => reviewsAPI.getStats(),
  });

  const acceptRequest = useMutation({
    mutationFn: (id: string) => reviewsAPI.openForReview(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["reviewRequests"] }),
  });

  const declineRequest = useMutation({
    mutationFn: (id: string) => reviewsAPI.deleteRequest(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["reviewRequests"] }),
  });

  if (requestsLoading || statsLoading) return <Spinner />;

  const requests = requestsData?.results || [];
  const statsData = stats || {};

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Review Requests</h1>
          <p className="text-sm text-slate-400 mt-1">
            Manage and track all review requests
          </p>
        </div>
        <Button onClick={() => navigate("/reviews/new")}>
          <Plus size={16} className="mr-1" /> New Request
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Requests"
          value={statsData.total_requests || 0}
          icon={<ListTodo size={20} />}
        />
        <StatCard
          label="Pending"
          value={statsData.pending_reviews || 0}
          icon={<Clock size={20} />}
          color="text-yellow-400"
        />
        <StatCard
          label="Completed"
          value={statsData.completed_reviews || 0}
          icon={<CheckCircle size={20} />}
          color="text-green-400"
        />
        <StatCard
          label="Avg Rating"
          value={statsData.average_rating?.toFixed(1) || "0.0"}
          icon={<Star size={20} />}
          color="text-yellow-400"
          suffix="/5"
        />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex flex-1 gap-2">
          <div className="relative flex-1 max-w-md">
            <Search
              size={16}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search requests..."
              className="pl-9"
            />
          </div>
          <Select
            options={[
              { value: "all", label: "All Status" },
              { value: "pending", label: "Pending" },
              { value: "in_progress", label: "In Progress" },
              { value: "completed", label: "Completed" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-36"
          />
          <Select
            options={[
              { value: "newest", label: "Newest First" },
              { value: "oldest", label: "Oldest First" },
              { value: "due_soon", label: "Due Soon" },
              { value: "priority", label: "Priority" },
            ]}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="w-36"
          />
        </div>
      </div>

      {/* Results */}
      {requests.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-slate-400">No review requests found.</p>
          <Button
            variant="ghost"
            className="mt-4"
            onClick={() => navigate("/reviews/new")}
          >
            Create your first request
          </Button>
        </Card>
      ) : (
        <div className="space-y-3">
          {requests.map((request) => (
            <ReviewRequestCard
              key={request.id}
              request={request}
              onAccept={(id) => acceptRequest.mutate(id)}
              onDecline={(id) => declineRequest.mutate(id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const StatCard = ({
  label,
  value,
  icon,
  color = "text-neon-blue",
  suffix = "",
}: any) => (
  <Card className="p-4">
    <div className="flex items-center justify-between">
      <div className={`${color}`}>{icon}</div>
      <div className="text-right">
        <div className="text-2xl font-bold text-white">
          {value}
          {suffix}
        </div>
        <div className="text-xs text-slate-400">{label}</div>
      </div>
    </div>
  </Card>
);
