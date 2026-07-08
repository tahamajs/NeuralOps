import { useMutation, useQueryClient } from "@tanstack/react-query";
import { communityAPI } from "../services/api";
import { ChevronUp, ChevronDown } from "lucide-react";
import { cn } from "../../../../lib/utils";
import type { VoteType } from "../types";

interface VoteButtonsProps {
  targetType: "question" | "answer";
  targetId: string;
  score: number;
  userVote: VoteType;
  size?: "sm" | "md" | "lg";
  onSuccess?: () => void;
}

export function VoteButtons({
  targetType,
  targetId,
  score,
  userVote,
  size = "md",
  onSuccess,
}: VoteButtonsProps) {
  const queryClient = useQueryClient();

  const voteMutation = useMutation({
    mutationFn: async (value: VoteType) => {
      if (targetType === "question") {
        return communityAPI.voteQuestion(targetId, value);
      }
      return communityAPI.voteAnswer(targetId, value);
    },
    onMutate: async (newVote) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({
        queryKey: [targetType === "question" ? "question" : "answers"],
      });
      const previousData = queryClient.getQueryData([
        targetType === "question" ? "question" : "answers",
        targetId,
      ]);

      // Optimistic update
      queryClient.setQueryData(
        [targetType === "question" ? "question" : "answers", targetId],
        (old: any) => {
          if (!old) return old;
          let newScore = score;
          if (userVote === "up" && newVote === null) newScore -= 1;
          else if (userVote === "down" && newVote === null) newScore += 1;
          else if (userVote === null && newVote === "up") newScore += 1;
          else if (userVote === null && newVote === "down") newScore -= 1;
          else if (userVote === "up" && newVote === "down") newScore -= 2;
          else if (userVote === "down" && newVote === "up") newScore += 2;
          return { ...old, vote_score: newScore, user_vote: newVote };
        },
      );
      return { previousData };
    },
    onError: (err, newVote, context) => {
      queryClient.setQueryData(
        [targetType === "question" ? "question" : "answers", targetId],
        context?.previousData,
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: [targetType === "question" ? "question" : "answers"],
      });
      onSuccess?.();
    },
  });

  const handleVote = (value: VoteType) => {
    const newVote = userVote === value ? null : value;
    voteMutation.mutate(newVote);
  };

  const iconSize = size === "sm" ? 16 : size === "lg" ? 24 : 20;

  return (
    <div className="flex flex-col items-center gap-0.5">
      <button
        onClick={() => handleVote("up")}
        className={cn(
          "p-1 rounded transition-colors",
          userVote === "up"
            ? "text-orange-400 bg-orange-500/10"
            : "text-slate-500 hover:text-orange-400 hover:bg-white/5",
        )}
      >
        <ChevronUp size={iconSize} />
      </button>
      <span
        className={cn(
          "font-bold text-sm min-w-[2rem] text-center",
          score > 0
            ? "text-green-400"
            : score < 0
              ? "text-red-400"
              : "text-white",
        )}
      >
        {score}
      </span>
      <button
        onClick={() => handleVote("down")}
        className={cn(
          "p-1 rounded transition-colors",
          userVote === "down"
            ? "text-blue-400 bg-blue-500/10"
            : "text-slate-500 hover:text-blue-400 hover:bg-white/5",
        )}
      >
        <ChevronDown size={iconSize} />
      </button>
    </div>
  );
}
