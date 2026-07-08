import React, { useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { portfolioAPI } from "../services/api";
import { useAuthStore } from "../../auth/store/authStore";
import {
  Card,
  Button,
  Avatar,
  Tabs,
  TabList,
  Tab,
  TabPanel,
  Spinner,
  Badge,
} from "../../../components/ui";
import { ProfileHeader } from "../components/ProfileHeader";
import { StatsGrid } from "../components/StatsGrid";
import { ActivityHeatmap } from "../components/ActivityHeatmap";
import { ShowcaseSection } from "../components/ShowcaseSection";
import { SkillsWidget } from "../components/SkillsWidget";
import { useToast } from "../../../hooks/useToast";

export default function PublicPortfolioPage() {
  const { username } = useParams<{ username: string }>();
  const { user: currentUser } = useAuthStore();
  const { showToast } = useToast();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("showcase");

  const {
    data: portfolio,
    isLoading: portfolioLoading,
    error,
  } = useQuery({
    queryKey: ["portfolio", username],
    queryFn: () => portfolioAPI.getPortfolio(username!),
    enabled: !!username,
  });

  const { data: stats } = useQuery({
    queryKey: ["portfolio-stats", portfolio?.id],
    queryFn: () => portfolioAPI.getStats(portfolio.id),
    enabled: !!portfolio?.id,
  });

  const { data: showcase } = useQuery({
    queryKey: ["portfolio-showcase", portfolio?.id],
    queryFn: () => portfolioAPI.getShowcaseItems(portfolio.id),
    enabled: !!portfolio?.id,
  });

  const { data: contributions } = useQuery({
    queryKey: ["portfolio-contributions", portfolio?.id],
    queryFn: () => portfolioAPI.getContributions(portfolio.id),
    enabled: !!portfolio?.id,
  });

  const followMutation = useMutation({
    mutationFn: () =>
      portfolio?.is_following
        ? portfolioAPI.unfollow(username!)
        : portfolioAPI.follow(username!),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ["portfolio", username] });
      const previous = queryClient.getQueryData(["portfolio", username]);
      queryClient.setQueryData(["portfolio", username], (old: any) => ({
        ...old,
        is_following: !old?.is_following,
        followers_count: old?.is_following
          ? (old?.followers_count || 0) - 1
          : (old?.followers_count || 0) + 1,
      }));
      return { previous };
    },
    onError: (err, _, context) => {
      queryClient.setQueryData(["portfolio", username], context?.previous);
      showToast("Failed to update follow status", "error");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio", username] });
    },
  });

  if (portfolioLoading) return <Spinner />;
  if (error)
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="p-8 text-center">
          <h2 className="text-xl font-bold text-white mb-2">
            Researcher not found
          </h2>
          <p className="text-slate-400">
            The profile you're looking for doesn't exist or is private.
          </p>
          <Link to="/dashboard">
            <Button className="mt-4">Back to Dashboard</Button>
          </Link>
        </Card>
      </div>
    );
  if (!portfolio) return null;

  const isOwnProfile = currentUser?.username === portfolio.username;

  return (
    <div className="min-h-screen bg-surface">
      <ProfileHeader
        portfolio={portfolio}
        stats={stats}
        isOwnProfile={isOwnProfile}
        onFollow={followMutation.mutate}
        isFollowing={portfolio.is_following}
        isFollowPending={followMutation.isPending}
      />

      <div className="max-w-6xl mx-auto px-4 py-8">
        <Tabs
          defaultValue="showcase"
          value={activeTab}
          onValueChange={setActiveTab}
        >
          <TabList className="mb-6">
            <Tab value="showcase">Showcase</Tab>
            <Tab value="activity">Activity</Tab>
            <Tab value="stats">Statistics</Tab>
            <Tab value="skills">Skills</Tab>
          </TabList>

          <TabPanel value="showcase">
            <ShowcaseSection items={showcase || []} />
          </TabPanel>

          <TabPanel value="activity">
            <ActivityHeatmap data={contributions} />
          </TabPanel>

          <TabPanel value="stats">
            <StatsGrid stats={stats} />
          </TabPanel>

          <TabPanel value="skills">
            <SkillsWidget portfolio={portfolio} />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
