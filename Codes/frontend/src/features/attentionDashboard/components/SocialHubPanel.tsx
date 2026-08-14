// src/features/attentionDashboard/components/SocialHubPanel.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { ReactionType } from '../types';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Users,
  Swords,
  GitFork,
  MessageSquare,
  Share2,
  ThumbsUp,
  Award,
  Sparkles,
  CheckCircle2,
  Send,
  HelpCircle,
  Copy,
  ExternalLink,
} from 'lucide-react';

const REACTION_CONFIG: { type: ReactionType; icon: string; label: string; weight: string; color: string }[] = [
  { type: 'interesting', icon: '🔬', label: 'Interesting', weight: '+2 XP', color: 'hover:border-blue-500 hover:bg-blue-500/10' },
  { type: 'efficient', icon: '⚡', label: 'Efficient', weight: '+3 XP', color: 'hover:border-amber-500 hover:bg-amber-500/10' },
  { type: 'specialized', icon: '🎯', label: 'Specialized', weight: '+3 XP', color: 'hover:border-emerald-500 hover:bg-emerald-500/10' },
  { type: 'redundant', icon: '🔄', label: 'Redundant', weight: '-1 prune', color: 'hover:border-purple-500 hover:bg-purple-500/10' },
  { type: 'novel', icon: '🧩', label: 'Novel Discovery', weight: '+4 XP', color: 'hover:border-rose-500 hover:bg-rose-500/10' },
];

export const SocialHubPanel: React.FC = () => {
  const {
    reactions,
    userReactions,
    toggleReaction,
    arenaMatch,
    voteArena,
    revealArena,
    nextArenaMatch,
    arenaVotesCount,
    forkLineage,
    discussions,
    addDiscussionComment,
    selectedModel,
    selectedAttentionType,
    selectedLayer,
    selectedHead,
  } = useAttentionStore();

  const [commentText, setCommentText] = useState('');
  const [commentLatex, setCommentLatex] = useState('');
  const [activeSocialTab, setActiveSocialTab] = useState<'arena' | 'reactions' | 'lineage' | 'discussions' | 'share'>('arena');
  const [shareCopied, setShareCopied] = useState(false);

  const handlePostComment = (e: React.FormEvent) => {
    e.preventDefault();
    if (!commentText.trim()) return;
    addDiscussionComment(commentText, commentLatex || undefined, 'head', `Layer ${selectedLayer}, Head ${selectedHead}`);
    setCommentText('');
    setCommentLatex('');
  };

  const handleCopyShareLink = () => {
    const shareUrl = `${window.location.origin}/attention?model=${encodeURIComponent(selectedModel)}&attn=${encodeURIComponent(selectedAttentionType)}&layer=${selectedLayer}&head=${selectedHead}`;
    navigator.clipboard.writeText(shareUrl);
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/20">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Social & Community Intelligence Layer (Zone 7)</h3>
            <p className="text-xs text-slate-400">
              Blind pairwise attention arena, multi-dimensional peer reactions, fork lineage & threaded discussions
            </p>
          </div>
        </div>

        {/* Tab Buttons */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-950 border border-slate-800">
          <button
            onClick={() => setActiveSocialTab('arena')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
              activeSocialTab === 'arena' ? 'bg-pink-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Swords className="w-3.5 h-3.5" /> Attention Arena
          </button>
          <button
            onClick={() => setActiveSocialTab('reactions')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
              activeSocialTab === 'reactions' ? 'bg-pink-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" /> Reactions
          </button>
          <button
            onClick={() => setActiveSocialTab('lineage')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
              activeSocialTab === 'lineage' ? 'bg-pink-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <GitFork className="w-3.5 h-3.5" /> Fork Lineage
          </button>
          <button
            onClick={() => setActiveSocialTab('discussions')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
              activeSocialTab === 'discussions' ? 'bg-pink-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" /> Discussions ({discussions.length})
          </button>
          <button
            onClick={() => setActiveSocialTab('share')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
              activeSocialTab === 'share' ? 'bg-pink-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Share2 className="w-3.5 h-3.5" /> Share Cards
          </button>
        </div>
      </div>

      {/* Tab 1: Attention Arena */}
      {activeSocialTab === 'arena' && arenaMatch && (
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-6 space-y-6 shadow-2xl">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <h4 className="text-base font-bold text-white">Blind Pairwise Attention Arena (ComparIA-Style)</h4>
                <Badge variant="warning">Match #{arenaVotesCount + 1}</Badge>
              </div>
              <p className="text-xs text-slate-400 mt-1">{arenaMatch.criterion}</p>
            </div>
            <div className="text-xs font-mono text-slate-400">
              Total Community Votes Cast: <strong className="text-white">{arenaVotesCount}</strong>
            </div>
          </div>

          {/* Side-by-Side Arena Split Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pattern A */}
            <div
              className={`p-5 rounded-xl border transition-all duration-300 flex flex-col justify-between space-y-4 ${
                arenaMatch.revealed
                  ? 'bg-slate-900/90 border-blue-500/50 shadow-lg'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-bold font-mono text-blue-400">
                    {arenaMatch.revealed ? arenaMatch.patternA.model : 'Option A (Blind)'}
                  </span>
                  {arenaMatch.revealed && (
                    <Badge variant="outline" className="font-mono text-xs">
                      Elo: {arenaMatch.eloA}
                    </Badge>
                  )}
                </div>

                {/* Simulated Heatmap Box */}
                <div className="w-full h-44 bg-slate-950 rounded-lg border border-slate-800 flex items-center justify-center p-4">
                  <div className="grid grid-cols-6 gap-1 w-full h-full">
                    {Array.from({ length: 36 }, (_, i) => (
                      <div
                        key={i}
                        className="rounded"
                        style={{
                          backgroundColor: '#3b82f6',
                          opacity: Math.sin(i * 0.4) * 0.4 + 0.5,
                        }}
                      />
                    ))}
                  </div>
                </div>

                {arenaMatch.revealed && (
                  <p className="text-xs text-slate-300 mt-3 font-mono">
                    {arenaMatch.patternA.attentionType} · Layer {arenaMatch.patternA.layer}, Head {arenaMatch.patternA.head}
                  </p>
                )}
              </div>

              <Button
                variant="primary"
                className="w-full bg-blue-600 hover:bg-blue-500 font-semibold text-xs"
                onClick={() => voteArena('A')}
                disabled={arenaMatch.revealed}
              >
                Vote Option A as Superior (+10 XP)
              </Button>
            </div>

            {/* Pattern B */}
            <div
              className={`p-5 rounded-xl border transition-all duration-300 flex flex-col justify-between space-y-4 ${
                arenaMatch.revealed
                  ? 'bg-slate-900/90 border-purple-500/50 shadow-lg'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-bold font-mono text-purple-400">
                    {arenaMatch.revealed ? arenaMatch.patternB.model : 'Option B (Blind)'}
                  </span>
                  {arenaMatch.revealed && (
                    <Badge variant="outline" className="font-mono text-xs">
                      Elo: {arenaMatch.eloB}
                    </Badge>
                  )}
                </div>

                {/* Simulated Heatmap Box */}
                <div className="w-full h-44 bg-slate-950 rounded-lg border border-slate-800 flex items-center justify-center p-4">
                  <div className="grid grid-cols-6 gap-1 w-full h-full">
                    {Array.from({ length: 36 }, (_, i) => (
                      <div
                        key={i}
                        className="rounded"
                        style={{
                          backgroundColor: '#a855f7',
                          opacity: Math.cos(i * 0.5) * 0.4 + 0.5,
                        }}
                      />
                    ))}
                  </div>
                </div>

                {arenaMatch.revealed && (
                  <p className="text-xs text-slate-300 mt-3 font-mono">
                    {arenaMatch.patternB.attentionType} · Layer {arenaMatch.patternB.layer}, Head {arenaMatch.patternB.head}
                  </p>
                )}
              </div>

              <Button
                variant="primary"
                className="w-full bg-purple-600 hover:bg-purple-500 font-semibold text-xs"
                onClick={() => voteArena('B')}
                disabled={arenaMatch.revealed}
              >
                Vote Option B as Superior (+10 XP)
              </Button>
            </div>
          </div>

          {/* Delayed Reveal & Next Round */}
          <div className="flex items-center justify-between pt-4 border-t border-slate-800">
            <Button variant="outline" size="sm" onClick={revealArena} disabled={arenaMatch.revealed}>
              Reveal Identities (No Vote)
            </Button>
            <Button variant="primary" size="sm" onClick={nextArenaMatch} className="bg-emerald-600 hover:bg-emerald-500">
              Next Pairwise Match →
            </Button>
          </div>
        </div>
      )}

      {/* Tab 2: Multi-Dimensional Reactions */}
      {activeSocialTab === 'reactions' && (
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-6 space-y-6 shadow-2xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-base font-bold text-white">Multi-Dimensional Attention Signal Reactions</h4>
              <p className="text-xs text-slate-400 mt-1">
                Beyond binary likes: signal research insights, compression candidacy, and novel inductive biases
              </p>
            </div>
            <Badge variant="outline">Weighted Quality Scoring</Badge>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {REACTION_CONFIG.map((rc) => {
              const count = reactions[rc.type] || 0;
              const hasReacted = userReactions.has(rc.type);

              return (
                <button
                  key={rc.type}
                  onClick={() => toggleReaction(rc.type)}
                  className={`p-4 rounded-xl border text-center transition-all flex flex-col items-center justify-center space-y-2 ${
                    hasReacted
                      ? 'bg-slate-900 border-pink-500 ring-2 ring-pink-500/40 shadow-lg'
                      : `bg-slate-900/60 border-slate-800 ${rc.color}`
                  }`}
                >
                  <span className="text-3xl">{rc.icon}</span>
                  <span className="text-xs font-bold text-white">{rc.label}</span>
                  <div className="flex items-center gap-1.5 text-xs font-mono">
                    <span className="text-pink-400 font-bold">{count}</span>
                    <span className="text-[10px] text-slate-500">({rc.weight})</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 3: Fork Lineage */}
      {activeSocialTab === 'lineage' && (
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-6 space-y-4 shadow-2xl">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-bold text-white flex items-center gap-2">
              <GitFork className="w-5 h-5 text-purple-400" />
              Exploration Lineage DAG & Community Forks
            </h4>
            <Button size="sm" variant="primary" className="bg-purple-600 hover:bg-purple-500 text-xs">
              <GitFork className="w-3.5 h-3.5 mr-1" /> Fork This Exploration (+20 XP)
            </Button>
          </div>

          <div className="space-y-3 pt-2">
            {forkLineage.map((fn, idx) => (
              <div
                key={fn.id}
                className={`p-4 rounded-xl border transition-all flex items-center justify-between ${
                  fn.id === 'f3'
                    ? 'bg-purple-950/20 border-purple-500/60 ring-1 ring-purple-500/40'
                    : 'bg-slate-900/60 border-slate-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-mono text-xs text-purple-300">
                    {idx + 1}
                  </span>
                  <div>
                    <h5 className="text-sm font-semibold text-white">{fn.name}</h5>
                    <p className="text-xs text-slate-400">By {fn.author} · {fn.date} · {fn.attentionType}</p>
                  </div>
                </div>
                <Badge variant="outline" className="font-mono text-xs">
                  {fn.reactions} Reactions
                </Badge>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 4: Discussions */}
      {activeSocialTab === 'discussions' && (
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-6 space-y-6 shadow-2xl">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-bold text-white flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-blue-400" />
              Threaded Mathematical & Linguistic Discussions
            </h4>
            <Badge variant="outline">LaTeX & SVG Support</Badge>
          </div>

          {/* New Comment Input */}
          <form onSubmit={handlePostComment} className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
            <textarea
              placeholder="Share an insight, question, or hypothesis about this attention head..."
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
              className="w-full h-20 p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 resize-none"
            />
            <div className="flex flex-wrap items-center justify-between gap-3">
              <input
                type="text"
                placeholder="Optional LaTeX formula (e.g. W_{QK} = W_Q W_K^T)"
                value={commentLatex}
                onChange={(e) => setCommentLatex(e.target.value)}
                className="flex-1 min-w-[240px] px-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500"
              />
              <Button type="submit" size="sm" variant="primary" className="bg-blue-600 hover:bg-blue-500 text-xs">
                <Send className="w-3.5 h-3.5 mr-1" /> Post Discussion (+15 XP)
              </Button>
            </div>
          </form>

          {/* Discussions List */}
          <div className="space-y-4">
            {discussions.map((d) => (
              <div key={d.id} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{d.authorAvatar}</span>
                    <strong className="text-xs font-semibold text-white">{d.author}</strong>
                    <span className="text-[11px] text-slate-500">{d.timestamp}</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] font-mono">
                    {d.anchor.label}
                  </Badge>
                </div>
                <p className="text-xs text-slate-300">{d.content}</p>
                {d.latex && (
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 font-mono text-xs text-cyan-300">
                    ${d.latex}$
                  </div>
                )}

                {/* Replies */}
                {d.replies.length > 0 && (
                  <div className="pl-4 border-l-2 border-slate-800 space-y-2 pt-2">
                    {d.replies.map((r) => (
                      <div key={r.id} className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60 text-xs space-y-1">
                        <div className="flex items-center gap-2">
                          <span>{r.authorAvatar}</span>
                          <strong className="text-white text-[11px]">{r.author}</strong>
                          <span className="text-[10px] text-slate-500">{r.timestamp}</span>
                        </div>
                        <p className="text-slate-300">{r.content}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 5: Share Cards */}
      {activeSocialTab === 'share' && (
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-6 space-y-6 shadow-2xl">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-bold text-white flex items-center gap-2">
              <Share2 className="w-5 h-5 text-cyan-400" />
              One-Click Social Share Card Generator
            </h4>
            <Badge variant="outline">2x Retina Ready</Badge>
          </div>

          <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-950 to-blue-950/40 border border-slate-700/80 shadow-2xl space-y-4 max-w-xl mx-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono uppercase tracking-widest text-cyan-400">NeuralOps Attention Lab</span>
                <h3 className="text-lg font-bold text-white mt-0.5">{selectedModel}</h3>
              </div>
              <Badge variant="outline" className="font-mono text-xs">{selectedAttentionType}</Badge>
            </div>

            <p className="text-xs text-slate-300 font-mono">
              Examining Layer {selectedLayer}, Head {selectedHead} with low-rank SVD attention decomposition.
            </p>

            <div className="grid grid-cols-3 gap-2 p-3 bg-slate-950/80 rounded-xl border border-slate-800 font-mono text-center text-xs">
              <div>
                <span className="text-[10px] text-slate-500 block">Importance</span>
                <strong className="text-emerald-400">0.89</strong>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Stability</span>
                <strong className="text-cyan-400">0.94</strong>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Rank $r_{eff}$</span>
                <strong className="text-purple-400">6.42</strong>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 text-[11px] text-slate-500 font-mono">
              <span>https://neuralops.com/attention</span>
              <span>QR Link Embedded</span>
            </div>
          </div>

          <div className="flex justify-center gap-3">
            <Button variant="primary" onClick={handleCopyShareLink} className="bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold">
              {shareCopied ? <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> : <Copy className="w-3.5 h-3.5 mr-1" />}
              {shareCopied ? 'Link Copied to Clipboard!' : 'Copy Interactive Dashboard Link'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
