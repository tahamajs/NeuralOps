// src/features/attentionDashboard/components/GamificationSidebar.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Flame,
  Trophy,
  Award,
  DollarSign,
  HelpCircle,
  CheckCircle2,
  XCircle,
  Sparkles,
  ChevronRight,
  TrendingUp,
  Clock,
  Zap,
} from 'lucide-react';

export const GamificationSidebar: React.FC = () => {
  const {
    gamification,
    dailyPuzzle,
    submitPuzzleAnswer,
    badges,
    bounties,
    claimBounty,
  } = useAttentionStore();

  const [activeTab, setActiveTab] = useState<'puzzle' | 'badges' | 'leaderboard' | 'bounties'>('puzzle');
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);

  const xpProgressPercent = Math.min(
    100,
    Math.round((gamification.currentXP / gamification.nextLevelXP) * 100)
  );

  const handleAnswer = (index: number) => {
    if (dailyPuzzle.solved) return;
    setSelectedAnswer(index);
    submitPuzzleAnswer(index);
  };

  return (
    <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 space-y-6 shadow-2xl backdrop-blur-md">
      {/* User Tier & XP Progress Header */}
      <div className="space-y-3 pb-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">🏆</span>
            <div>
              <h4 className="text-sm font-bold text-white leading-none">{gamification.tierName}</h4>
              <span className="text-[11px] text-slate-400 font-mono">Level {gamification.currentLevel} Researcher</span>
            </div>
          </div>
          {/* Streak Flame */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-bold font-mono">
            <Flame className="w-4 h-4 text-amber-500 animate-pulse fill-amber-500" />
            <span>{gamification.streakCount}d Streak</span>
          </div>
        </div>

        {/* XP Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">{gamification.currentXP} XP</span>
            <span className="text-slate-500">{gamification.nextLevelXP} XP (Next Tier)</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${xpProgressPercent}%` }}
            />
          </div>
        </div>

        {/* 7-Day Mini Calendar */}
        <div className="flex items-center justify-between pt-1">
          {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, idx) => {
            const isCompleted = gamification.weekStreakHistory[idx];
            return (
              <div key={idx} className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-slate-500 font-mono">{day}</span>
                <span
                  className={`w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold ${
                    isCompleted
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                      : 'bg-slate-950 text-slate-600 border border-slate-800'
                  }`}
                >
                  {isCompleted ? '✓' : '·'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Gamification Sub-tabs */}
      <div className="flex items-center justify-between p-1 bg-slate-950 rounded-lg border border-slate-800 text-xs">
        <button
          onClick={() => setActiveTab('puzzle')}
          className={`flex-1 py-1.5 rounded-md font-semibold transition-all ${
            activeTab === 'puzzle' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
          }`}
        >
          Daily Puzzle
        </button>
        <button
          onClick={() => setActiveTab('badges')}
          className={`flex-1 py-1.5 rounded-md font-semibold transition-all ${
            activeTab === 'badges' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
          }`}
        >
          Badges ({badges.filter((b) => b.unlocked).length})
        </button>
        <button
          onClick={() => setActiveTab('leaderboard')}
          className={`flex-1 py-1.5 rounded-md font-semibold transition-all ${
            activeTab === 'leaderboard' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
          }`}
        >
          Rankings
        </button>
        <button
          onClick={() => setActiveTab('bounties')}
          className={`flex-1 py-1.5 rounded-md font-semibold transition-all ${
            activeTab === 'bounties' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
          }`}
        >
          Bounties
        </button>
      </div>

      {/* Tab 1: Daily Attention Puzzle */}
      {activeTab === 'puzzle' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[10px] font-mono text-amber-400 uppercase tracking-wider">Today's Mini-Challenge</span>
              <h5 className="text-sm font-bold text-white">{dailyPuzzle.title}</h5>
            </div>
            <Badge variant="warning">+{dailyPuzzle.xpReward} XP</Badge>
          </div>

          <p className="text-xs text-slate-300">{dailyPuzzle.description}</p>

          {/* Multiple Choice Options */}
          <div className="space-y-2">
            {dailyPuzzle.options.map((opt, idx) => {
              const isSelected = selectedAnswer === idx || dailyPuzzle.userAnswer === idx;
              const isCorrect = idx === dailyPuzzle.correctOptionIndex;
              let btnStyle = 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700';

              if (dailyPuzzle.solved) {
                if (isCorrect) btnStyle = 'bg-emerald-950/60 border-emerald-500 text-emerald-300 font-bold';
                else if (isSelected && !isCorrect) btnStyle = 'bg-rose-950/60 border-rose-500 text-rose-300';
              }

              return (
                <button
                  key={opt}
                  onClick={() => handleAnswer(idx)}
                  disabled={dailyPuzzle.solved}
                  className={`w-full p-2.5 rounded-lg border text-left text-xs font-mono transition-all flex items-center justify-between ${btnStyle}`}
                >
                  <span>{opt}</span>
                  {dailyPuzzle.solved && isCorrect && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                  {dailyPuzzle.solved && isSelected && !isCorrect && <XCircle className="w-4 h-4 text-rose-400" />}
                </button>
              );
            })}
          </div>

          {dailyPuzzle.solved && (
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-300 space-y-1">
              <strong className="text-amber-400 font-semibold block">Explanation:</strong>
              <p>{dailyPuzzle.explanation}</p>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Badges Showcase */}
      {activeTab === 'badges' && (
        <div className="space-y-3">
          <div className="text-xs text-slate-400">Unlock badges by exploring architectures and publishing insights:</div>
          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {badges.map((b) => (
              <div
                key={b.id}
                className={`p-3 rounded-xl border flex items-center justify-between transition-all ${
                  b.unlocked ? 'bg-slate-950/90 border-amber-500/40 shadow-sm' : 'bg-slate-950/40 border-slate-800/80 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{b.icon}</span>
                  <div>
                    <h6 className="text-xs font-bold text-white">{b.name}</h6>
                    <p className="text-[10px] text-slate-400">{b.description}</p>
                  </div>
                </div>
                {b.unlocked ? (
                  <Badge variant="success" className="text-[10px]">Unlocked</Badge>
                ) : (
                  <span className="text-[10px] font-mono text-slate-500">{b.progressPercent}%</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 3: Real-time Leaderboard */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-3">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span>Weekly Interpretability Leaderboard</span>
            <Badge variant="outline">Live Sync</Badge>
          </div>

          <div className="space-y-2">
            {[
              { rank: 1, name: 'Dr. Sarah Vance', tier: 'Grandmaster', xp: '12,450 XP', change: '+2' },
              { rank: 2, name: 'Alex Thorne', tier: 'Attention Sage', xp: '9,820 XP', change: '0' },
              { rank: 3, name: 'You (Current User)', tier: 'Head Hunter', xp: `${gamification.currentXP} XP`, change: '+3', isCurrent: true },
              { rank: 4, name: 'Marc Dupont', tier: 'Pattern Scout', xp: '2,150 XP', change: '-1' },
            ].map((entry) => (
              <div
                key={entry.rank}
                className={`p-2.5 rounded-lg border flex items-center justify-between text-xs font-mono ${
                  entry.isCurrent ? 'bg-amber-950/20 border-amber-500/50 ring-1 ring-amber-500/30' : 'bg-slate-950 border-slate-800'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-5 font-bold text-slate-500">#{entry.rank}</span>
                  <span className="text-white font-semibold">{entry.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-amber-400 font-bold">{entry.xp}</span>
                  <span className="text-[10px] text-emerald-400">{entry.change}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 4: Bounty Marketplace */}
      {activeTab === 'bounties' && (
        <div className="space-y-3">
          <div className="text-xs text-slate-400">Funded mechanistic analysis bounties:</div>
          <div className="space-y-2.5">
            {bounties.map((b) => (
              <div key={b.id} className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <h6 className="text-xs font-bold text-white">{b.title}</h6>
                  <span className="text-xs font-mono font-bold text-emerald-400">${b.rewardAmount} {b.rewardCurrency}</span>
                </div>
                <p className="text-[11px] text-slate-400">{b.description}</p>
                <div className="flex items-center justify-between pt-1">
                  <span className="text-[10px] text-slate-500 font-mono">⏰ {b.daysLeft} days left</span>
                  <Button
                    size="sm"
                    variant={b.status === 'in_progress' ? 'outline' : 'primary'}
                    onClick={() => claimBounty(b.id)}
                    disabled={b.status !== 'open'}
                    className="text-xs h-6 px-2"
                  >
                    {b.status === 'open' ? 'Claim (+20 XP)' : 'In Progress'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
