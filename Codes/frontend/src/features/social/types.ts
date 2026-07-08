export interface DiscussionThread {
  id: string;
  title: string;
  author: string;
  author_avatar?: string;
  created_at: string;
  reply_count: number;
  like_count: number;
  last_activity: string;
}

export interface Discussion {
  id: string;
  title: string;
  content: string;
  author: string;
  author_avatar?: string;
  created_at: string;
  replies: DiscussionReply[];
  like_count: number;
  user_liked: boolean;
}

export interface DiscussionReply {
  id: string;
  content: string;
  author: string;
  author_avatar?: string;
  created_at: string;
  like_count: number;
}

export interface Alert {
  id: string;
  type: "mention" | "reply" | "follow" | "group_invite";
  message: string;
  read: boolean;
  created_at: string;
  link: string;
}

export interface SuggestedResearcher {
  id: string;
  username: string;
  avatar_url?: string;
  institution: string;
  expertise: string[];
  common_interests: string[];
  match_score: number;
  reason: string;
}
