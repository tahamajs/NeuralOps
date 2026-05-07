import { Bot, User, FlaskConical, FileText, AlertTriangle, TrendingUp, Play, Search, BookOpen } from 'lucide-react'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { useNavigate } from 'react-router-dom'
import { ChatMessage } from '../types'

export default function ChatMessageBubble({ msg }: { msg: ChatMessage }) {
  const navigate = useNavigate()

  // Experiment card (draft or launch)
  if (msg.role === 'experiment_card') {
    const config = msg.metadata?.config || {}
    const isLaunched = msg.metadata?.type === 'experiment_launched'
    return (
      <div className="flex justify-center my-3 animate-scale-in">
        <Card className="w-full max-w-md border border-neon-blue/30 bg-neon-blue/5">
          <div className="flex items-center gap-2 mb-2">
            <FlaskConical size={18} className="text-neon-blue" />
            <span className="text-sm font-display font-bold text-white">
              {isLaunched ? '🚀 Experiment Launched!' : 'Experiment Configuration'}
            </span>
          </div>
          <div className="text-xs space-y-1 text-slate-300">
            <p>Model: <span className="text-white">{config.model_name}</span></p>
            <p>Task: <span className="text-white">{config.task_name?.toUpperCase()}</span></p>
            <p>Attention: <span className="text-white">{config.attention_type || 'full'}</span></p>
            <p>Method: <span className="text-white">{config.method || 'baseline'}</span></p>
            <p>Epochs: {config.epochs} · Batch: {config.batch_size} · LR: {config.learning_rate}</p>
          </div>
          {isLaunched && config.run_id && (
            <Button size="sm" className="mt-2" onClick={() => navigate(`/experiments/${config.run_id}`)}>
              <Play size={14} className="mr-1" /> Monitor
            </Button>
          )}
        </Card>
      </div>
    )
  }

  // Training analysis card
  if (msg.role === 'tool' && msg.metadata?.tool_name === 'analyze_training') {
    const analysis = parseJSON(msg.content)
    if (!analysis?.status) return null
    return (
      <div className="flex justify-center my-3 animate-scale-in">
        <Card className="w-full max-w-md border border-yellow-500/30 bg-yellow-500/5">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={18} className="text-yellow-400" />
            <span className="text-sm font-display font-bold text-white">Training Analysis</span>
            <Badge status={analysis.status === 'healthy' ? 'completed' : analysis.status === 'overfitting' ? 'failed' : 'queued'}>
              {analysis.status}
            </Badge>
          </div>
          {analysis.issues?.length > 0 && (
            <div className="text-xs text-slate-300 mb-2">
              <span className="text-slate-400">Issues: </span>
              {analysis.issues.map((issue: string, i: number) => (
                <span key={i} className="text-yellow-400">{issue}{i < analysis.issues.length - 1 ? ', ' : ''}</span>
              ))}
            </div>
          )}
          {analysis.suggestions?.length > 0 && (
            <div className="text-xs text-slate-300">
              <span className="text-slate-400">Suggestions:</span>
              <ul className="list-disc list-inside mt-1 space-y-0.5 text-green-400">
                {analysis.suggestions.map((s: string, i: number) => <li key={i}>{s}</li>)}
              </ul>
            </div>
          )}
        </Card>
      </div>
    )
  }

  // Paper search result card
  if (msg.role === 'tool' && msg.metadata?.tool_name === 'search_papers') {
    const data = parseJSON(msg.content)
    if (!data?.results) return null
    return (
      <div className="flex justify-center my-3">
        <Card className="w-full max-w-md border border-green-500/30 bg-green-500/5">
          <div className="flex items-center gap-2 mb-2">
            <Search size={18} className="text-green-400" />
            <span className="text-sm font-display font-bold text-white">Search Results ({data.results.length} papers)</span>
          </div>
          <div className="max-h-64 overflow-y-auto scrollbar-thin space-y-2">
            {data.results.slice(0, 5).map((paper: any, i: number) => (
              <div key={i} className="bg-cyber-800 p-2 rounded text-xs">
                <p className="text-white font-medium truncate">{paper.title}</p>
                <p className="text-slate-400">{paper.authors?.slice(0, 3).join(', ')} ({paper.year})</p>
                <p className="text-neon-blue text-[10px]">{paper.url}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  // Tool result (collapsed)
  if (msg.role === 'tool') {
    return (
      <div className="flex justify-start mb-2">
        <div className="bg-cyber-800/50 rounded-lg px-3 py-1.5 max-w-[80%] text-xs text-slate-500 font-mono">
          🔧 {msg.metadata?.tool_name || 'tool'}: {msg.content.slice(0, 200)}...
        </div>
      </div>
    )
  }

  // Regular chat bubble
  const isUser = msg.role === 'user'
  return (
    <div className={`flex gap-2 mb-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center flex-shrink-0">
          <Bot size={14} className="text-white" />
        </div>
      )}
      <div className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm ${
        isUser
          ? 'bg-neon-blue/20 border border-neon-blue/30 text-white'
          : 'glass border border-white/10 text-slate-200'
      }`}>
        <div className="whitespace-pre-wrap">{msg.content}</div>
        <div className={`text-[10px] mt-1 ${isUser ? 'text-neon-blue/60' : 'text-slate-500'}`}>
          {new Date(msg.created_at).toLocaleTimeString()}
        </div>
      </div>
      {isUser && (
        <div className="w-7 h-7 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
          <User size={14} className="text-white" />
        </div>
      )}
    </div>
  )
}

// Helper to safely parse JSON from tool content
function parseJSON(str: string): any {
  try { return JSON.parse(str) } catch { return null }
}