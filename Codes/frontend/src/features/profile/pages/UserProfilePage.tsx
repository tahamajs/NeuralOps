// src/features/profile/pages/UserProfilePage.tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "../../auth/store/authStore";
import {
  profileAPI,
  UserProfile,
  UserStats,
  UserActivity,
  PinnedItem,
  Badge,
  OrganizationMembership,
} from "../services/api";
import {
  Card,
  Tabs,
  TabList,
  Tab,
  TabPanel,
  Spinner,
  Button,
  Avatar,
} from "../../../components/ui";
import { ProfileHeader } from "../components/ProfileHeader";
import { ContributionHeatmap } from "../components/ContributionHeatmap";
import { formatDistanceToNow } from "date-fns";
import {
  GitBranch,
  FileText,
  StickyNote,
  MessageSquare,
  Award,
  Users,
  Activity,
  Pin,
  ExternalLink,
  Star,
  ThumbsUp,
} from "lucide-react";

export default function UserProfilePage() {
  const { username } = useParams<{ username?: string }>();
  const { user: currentUser } = useAuthStore();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("overview");
  const [activityPage, setActivityPage] = useState(1);

  const isOwnProfile = !username || username === currentUser?.username;

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["profile", username || "me"],
    queryFn: () => profileAPI.getProfile(username),
  });

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["profile-stats", profile?.id],
    queryFn: () => profileAPI.getUserStats(profile.id),
    enabled: !!profile?.id,
  });

  const { data: activity, isLoading: activityLoading } = useQuery({
    queryKey: ["profile-activity", profile?.id, activityPage],
    queryFn: () => profileAPI.getUserActivity(profile.id, activityPage),
    enabled: !!profile?.id,
  });

  const { data: pinnedItems } = useQuery({
    queryKey: ["profile-pins", profile?.id],
    queryFn: () => profileAPI.getPinnedItems(profile.id),
    enabled: !!profile?.id,
  });

  const { data: badges } = useQuery({
    queryKey: ["profile-badges", profile?.id],
    queryFn: () => profileAPI.getBadges(profile.id),
    enabled: !!profile?.id,
  });

  const { data: orgs } = useQuery({
    queryKey: ["profile-orgs", profile?.id],
    queryFn: () => profileAPI.getOrganizationMemberships(profile.id),
    enabled: !!profile?.id,
  });

  const updateProfile = useMutation({
    mutationFn: (data: Partial<UserProfile>) => profileAPI.updateProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["profile", username || "me"],
      });
    },
  });

  const uploadAvatar = useMutation({
    mutationFn: (file: File) => profileAPI.uploadAvatar(file),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["profile", username || "me"],
      });
    },
  });

  if (profileLoading || statsLoading) return <Spinner />;

  if (!profile && !profileLoading) {
    return (
      <div className="text-center py-12 text-red-400">User not found.</div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <ProfileHeader
        profile={profile}
        stats={stats}
        isOwnProfile={isOwnProfile}
        onEdit={() => navigate("/profile/settings")}
        onAvatarUpload={(file) => uploadAvatar.mutate(file)}
      />

      <Tabs defaultValue="overview" onValueChange={setActiveTab}>
        <TabList>
          <Tab value="overview">Overview</Tab>
          <Tab value="activity">Activity</Tab>
          <Tab value="badges">Badges</Tab>
          <Tab value="organizations">Organizations</Tab>
        </TabList>

        <TabPanel value="overview">
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Pin size={16} className="text-neon-blue" /> Pinned Items
              </h3>
              {pinnedItems?.length ? (
                pinnedItems.map((item: PinnedItem) => (
                  <PinnedItemCard key={item.id} item={item} />
                ))
              ) : (
                <Card className="text-center text-slate-500 py-8">
                  No pinned items yet.
                </Card>
              )}
              <div className="pt-4">
                <h3 className="text-sm font-bold text-white mb-3">
                  Contribution Calendar
                </h3>
                <Card className="p-4">
                  <ContributionHeatmap userId={profile.id} />
                </Card>
              </div>
            </div>
            <div className="space-y-4">
              <Card>
                <h3 className="text-sm font-bold text-white mb-2">
                  Top 10% Reputation
                </h3>
                <div className="flex items-center gap-2">
                  <Award size={24} className="text-yellow-400" />
                  <div>
                    <div className="text-2xl font-bold text-white">
                      {stats?.reputation_score}
                    </div>
                    <div className="text-xs text-slate-400">
                      Rank #{stats?.rank}
                    </div>
                  </div>
                </div>
              </Card>
              <Card>
                <h3 className="text-sm font-bold text-white mb-2">
                  Recent Badges
                </h3>
                {badges?.slice(0, 5).map((badge: Badge) => (
                  <div
                    key={badge.id}
                    className="flex items-center gap-2 py-2 border-b border-white/5"
                  >
                    <img
                      src={badge.icon_url}
                      alt={badge.name}
                      className="w-6 h-6"
                    />
                    <div>
                      <div className="text-sm text-white">{badge.name}</div>
                      <div className="text-xs text-slate-500">
                        {badge.description}
                      </div>
                    </div>
                  </div>
                ))}
              </Card>
            </div>
          </div>
        </TabPanel>

        <TabPanel value="activity">
          <Card>
            <div className="space-y-4">
              {(activity?.results || []).map((act: UserActivity) => (
                <ActivityItem key={act.id} activity={act} />
              ))}
              {activity?.next && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setActivityPage((p) => p + 1)}
                >
                  Load More
                </Button>
              )}
            </div>
          </Card>
        </TabPanel>

        <TabPanel value="badges">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {badges?.map((badge: Badge) => (
              <Card key={badge.id} className="text-center p-4">
                <img
                  src={badge.icon_url}
                  alt={badge.name}
                  className="w-12 h-12 mx-auto mb-2"
                />
                <div className="font-bold text-white text-sm">{badge.name}</div>
                <div className="text-xs text-slate-400">
                  {badge.description}
                </div>
              </Card>
            ))}
          </div>
        </TabPanel>

        <TabPanel value="organizations">
          <div className="grid md:grid-cols-2 gap-4">
            {orgs?.map((org: OrganizationMembership) => (
              <Card key={org.id} className="flex items-center gap-3 p-4">
                <Avatar src={org.avatar_url} size="md" />
                <div>
                  <div className="font-bold text-white">{org.name}</div>
                  <div className="text-xs text-slate-400 capitalize">
                    {org.role}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabPanel>
      </Tabs>
    </div>
  );
}

// Helper components (same as before)
const PinnedItemCard = ({ item }: { item: PinnedItem }) => {
  const Icon =
    item.type === "experiment"
      ? GitBranch
      : item.type === "paper"
        ? FileText
        : StickyNote;
  return (
    <Card hover>
      <div className="flex items-start gap-3">
        <Icon size={20} className="text-neon-blue shrink-0 mt-0.5" />
        <div>
          <div className="font-semibold text-white">{item.title}</div>
          <div className="text-sm text-slate-400 line-clamp-2">
            {item.description}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {formatDistanceToNow(new Date(item.created_at), {
              addSuffix: true,
            })}
          </div>
        </div>
      </div>
    </Card>
  );
};

const ActivityItem = ({ activity }: { activity: UserActivity }) => {
  let icon = null;
  let link = "";
  switch (activity.target_type) {
    case "experiment":
      icon = <GitBranch size={16} className="text-green-400" />;
      link = `/experiments/${activity.target_id}`;
      break;
    case "note":
      icon = <StickyNote size={16} className="text-yellow-400" />;
      link = `/notes/${activity.target_id}`;
      break;
    case "paper":
      icon = <FileText size={16} className="text-purple-400" />;
      link = `/paper/editor/${activity.target_id}`;
      break;
    case "question":
      icon = <MessageSquare size={16} className="text-blue-400" />;
      link = `/questions/${activity.target_id}`;
      break;
    case "answer":
      icon = <ThumbsUp size={16} className="text-orange-400" />;
      link = `/questions/${activity.target_id}`;
      break;
    default:
      icon = <Activity size={16} className="text-slate-400" />;
  }
  return (
    <div className="flex items-center gap-3 py-3 border-b border-white/5">
      {icon}
      <div className="flex-1">
        <div className="text-sm text-white">{activity.action}</div>
        <div className="text-xs text-slate-500">
          {formatDistanceToNow(new Date(activity.created_at), {
            addSuffix: true,
          })}
        </div>
      </div>
      <a href={link} className="text-neon-blue hover:underline">
        <ExternalLink size={14} />
      </a>
    </div>
  );
};
