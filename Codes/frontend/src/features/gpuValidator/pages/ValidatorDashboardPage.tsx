import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthStore } from '../../../features/auth/store/authStore';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/Tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import { Spinner } from '../../../components/ui/Spinner';
import { Progress } from '../../../components/ui/Progress';
import { 
  Shield, 
  Activity, 
  Server, 
  Award, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  ExternalLink,
  RefreshCw,
  TrendingUp,
  DollarSign,
  Cpu,
  HardDrive,
  Wifi,
  Eye,
  ShieldCheck,
  Gavel,
  Coins,
  BarChart3,
  Database,
  Zap,
  Users,
  Star,
  Flame,
  Calendar,
  Lock,
  Unlock
} from 'lucide-react';

// Types for validator data
interface ValidatorStats {
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  total_staked: number;
  total_rewards: number;
  uptime: number;
  reputation_score: number;
  success_rate: number;
  average_verification_time: number;
  active_validators: number;
  total_verifications: number;
  slashings_last_30d: number;
}

interface ValidatorInfo {
  node_id: string;
  status: 'active' | 'inactive' | 'jailed' | 'unbonding';
  stake_amount: number;
  commission_rate: number;
  delegators: number;
  total_delegated: number;
  apr: number;
  version: string;
  location?: string;
  hardware_specs?: {
    cpu: string;
    memory: string;
    gpu: string;
    storage: string;
  };
}

interface VerificationTask {
  id: string;
  job_id: string;
  provider_username: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'disputed';
  created_at: string;
  deadline: string;
  reward: number;
  verification_type: 'compute' | 'storage' | 'bandwidth' | 'ai_model';
}

interface StakingPosition {
  amount: number;
  rewards_earned: number;
  status: 'active' | 'unbonding' | 'withdrawn';
  lockup_period: number;
  unbonding_ends_at?: string;
}

interface SlashingEvent {
  id: string;
  reason: string;
  amount: number;
  timestamp: string;
  evidence_hash: string;
  severity: 'low' | 'medium' | 'high';
}

interface Delegation {
  delegator: string;
  amount: number;
  rewards: number;
  since: string;
}

// API service for validator endpoints
const validatorAPI = {
  getStats: () => 
    api.get('/validators/me/stats/').then(res => res.data),
  
  getInfo: () => 
    api.get('/validators/me/').then(res => res.data),
  
  getTasks: (params?: { status?: string; page?: number }) => 
    api.get('/tasks/mine/', { params }).then(res => res.data),
  
  getStakingPosition: () => 
    api.get('/stakes/me/').then(res => res.data),
  
  getSlashingEvents: () => 
    api.get('/slashing/history/me/').then(res => res.data),
  
  getDelegations: () => 
    api.get('/stakes/delegators/').then(res => res.data),
  
  stakeTokens: (amount: number) => 
    api.post('/stakes/stake/', { amount }).then(res => res.data),
  
  unbondTokens: (amount: number) => 
    api.post('/stakes/unbond/', { amount }).then(res => res.data),
  
  claimRewards: () => 
    api.post('/stakes/claim/', {}).then(res => res.data),
  
  updateValidatorMetadata: (data: Partial<ValidatorInfo>) => 
    api.patch('/validators/me/update/', data).then(res => res.data),
  
  submitVerification: (taskId: string, result: any) => 
    api.post(`/tasks/${taskId}/submit/`, result).then(res => res.data),
  
  finalizeConsensus: (taskId: string) => 
    api.post(`/tasks/${taskId}/finalize/`, {}).then(res => res.data),
};

// Component for the active tasks list
const TasksList = ({ tasks, isLoading, onRefresh }: { tasks: VerificationTask[]; isLoading: boolean; onRefresh: () => void }) => {
  if (isLoading) return <div className="py-8 text-center"><Spinner /></div>;
  if (tasks.length === 0) {
    return <div className="py-12 text-center text-slate-400">No pending verification tasks</div>;
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending': return <Badge variant="warning">Pending</Badge>;
      case 'in_progress': return <Badge variant="info">In Progress</Badge>;
      case 'completed': return <Badge variant="success">Completed</Badge>;
      case 'failed': return <Badge variant="danger">Failed</Badge>;
      case 'disputed': return <Badge variant="warning">Disputed</Badge>;
      default: return <Badge>Unknown</Badge>;
    }
  };

  const getTaskIcon = (type: string) => {
    switch (type) {
      case 'compute': return <Cpu size={16} className="text-neon-blue" />;
      case 'storage': return <HardDrive size={16} className="text-green-400" />;
      case 'bandwidth': return <Wifi size={16} className="text-yellow-400" />;
      case 'ai_model': return <Brain size={16} className="text-purple-400" />;
      default: return <Activity size={16} />;
    }
  };

  return (
    <div className="divide-y divide-white/5">
      {tasks.map((task) => (
        <div key={task.id} className="py-4 first:pt-0 last:pb-0">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              {getTaskIcon(task.verification_type)}
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-white">
                    Job: {task.job_id.slice(0, 8)}...
                  </span>
                  {getStatusBadge(task.status)}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Provider: {task.provider_username} · Created: {new Date(task.created_at).toLocaleString()}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Deadline: {new Date(task.deadline).toLocaleString()}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-semibold text-neon-blue">{task.reward} tokens</div>
              {task.status === 'pending' && (
                <Button variant="ghost" size="sm" className="mt-1">Verify →</Button>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

// Main Dashboard Component
export default function ValidatorDashboardPage() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch validator data
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['validator-stats'],
    queryFn: () => validatorAPI.getStats(),
  });

  const { data: validatorInfo, isLoading: infoLoading } = useQuery({
    queryKey: ['validator-info'],
    queryFn: () => validatorAPI.getInfo(),
  });

  const { data: tasksData, isLoading: tasksLoading } = useQuery({
    queryKey: ['validator-tasks', activeTab],
    queryFn: () => validatorAPI.getTasks({ status: activeTab === 'tasks' ? undefined : 'pending' }),
  });

  const { data: staking, isLoading: stakingLoading } = useQuery({
    queryKey: ['validator-staking'],
    queryFn: () => validatorAPI.getStakingPosition(),
  });

  const { data: slashingEvents, isLoading: slashingLoading } = useQuery({
    queryKey: ['validator-slashing'],
    queryFn: () => validatorAPI.getSlashingEvents(),
  });

  const { data: delegations, isLoading: delegationsLoading } = useQuery({
    queryKey: ['validator-delegations'],
    queryFn: () => validatorAPI.getDelegations(),
  });

  // Mutations
  const claimRewards = useMutation({
    mutationFn: () => validatorAPI.claimRewards(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['validator-stats'] }),
  });

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['validator-stats'] });
    queryClient.invalidateQueries({ queryKey: ['validator-info'] });
    queryClient.invalidateQueries({ queryKey: ['validator-tasks'] });
    queryClient.invalidateQueries({ queryKey: ['validator-staking'] });
    queryClient.invalidateQueries({ queryKey: ['validator-slashing'] });
  };

  if (statsLoading || infoLoading || stakingLoading) return <Spinner />;

  const tasks = tasksData?.results || [];
  const pendingTasks = tasks.filter((t: VerificationTask) => t.status === 'pending');

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Validator Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">Monitor your validator node performance and manage tasks</p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={refreshAll}>
            <RefreshCw size={14} className="mr-1" /> Refresh
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Staked"
          value={`${stats?.total_staked?.toLocaleString() || 0} tokens`}
          icon={<Coins size={20} />}
          trend={stats?.staked_change}
          color="text-neon-blue"
        />
        <StatCard
          title="Total Rewards"
          value={`${stats?.total_rewards?.toLocaleString() || 0} tokens`}
          icon={<DollarSign size={20} />}
          trend={stats?.rewards_change}
          color="text-green-400"
        />
        <StatCard
          title="Success Rate"
          value={`${stats?.success_rate || 0}%`}
          icon={<CheckCircle size={20} />}
          color="text-green-400"
        />
        <StatCard
          title="Uptime"
          value={`${stats?.uptime || 0}%`}
          icon={<Activity size={20} />}
          color={stats?.uptime > 99 ? 'text-green-400' : 'text-yellow-400'}
        />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="glass p-1 rounded-xl w-full justify-start overflow-x-auto">
          <TabsTrigger value="overview" className="data-[state=active]:bg-neon-blue data-[state=active]:text-white">
            Overview
          </TabsTrigger>
          <TabsTrigger value="tasks" className="data-[state=active]:bg-neon-blue data-[state=active]:text-white">
            Verification Tasks
            {pendingTasks.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-yellow-500 text-black">{pendingTasks.length}</span>
            )}
          </TabsTrigger>
          <TabsTrigger value="staking" className="data-[state=active]:bg-neon-blue data-[state=active]:text-white">
            Staking
          </TabsTrigger>
          <TabsTrigger value="slashing" className="data-[state=active]:bg-neon-blue data-[state=active]:text-white">
            Slashing Events
          </TabsTrigger>
          <TabsTrigger value="delegations" className="data-[state=active]:bg-neon-blue data-[state=active]:text-white">
            Delegations
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-white">Validator Status</CardTitle>
                <CardDescription>Current node status and configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-slate-400">Node ID</span>
                  <span className="font-mono text-white">{validatorInfo?.node_id?.slice(0, 16)}...</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Status</span>
                  <ValidatorStatusBadge status={validatorInfo?.status} />
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Commission Rate</span>
                  <span className="text-white">{validatorInfo?.commission_rate}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">APR</span>
                  <span className="text-green-400">{validatorInfo?.apr}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Version</span>
                  <span className="text-white">{validatorInfo?.version}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-white">Hardware Specifications</CardTitle>
                <CardDescription>Node hardware configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3">
                  <Cpu size={20} className="text-neon-blue" />
                  <div>
                    <div className="text-sm text-white">{validatorInfo?.hardware_specs?.cpu || 'N/A'}</div>
                    <div className="text-xs text-slate-500">Processor</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Database size={20} className="text-neon-blue" />
                  <div>
                    <div className="text-sm text-white">{validatorInfo?.hardware_specs?.memory || 'N/A'}</div>
                    <div className="text-xs text-slate-500">Memory</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Cpu size={20} className="text-purple-400" />
                  <div>
                    <div className="text-sm text-white">{validatorInfo?.hardware_specs?.gpu || 'N/A'}</div>
                    <div className="text-xs text-slate-500">GPU</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="text-white">Recent Tasks Activity</CardTitle>
                <CardDescription>Latest verification tasks handled by this validator</CardDescription>
              </CardHeader>
              <CardContent>
                <TasksList tasks={tasks.slice(0, 5)} isLoading={tasksLoading} onRefresh={refreshAll} />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Tasks Tab */}
        <TabsContent value="tasks">
          <Card>
            <CardHeader>
              <CardTitle className="text-white">Verification Tasks</CardTitle>
              <CardDescription>Manage and complete verification tasks assigned to this validator</CardDescription>
            </CardHeader>
            <CardContent>
              <TasksList tasks={tasks} isLoading={tasksLoading} onRefresh={refreshAll} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Staking Tab */}
        <TabsContent value="staking">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-white">Your Stake</CardTitle>
                <CardDescription>Manage your staking position</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-slate-400">Staked Amount</span>
                  <span className="text-white font-bold">{staking?.amount?.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Unclaimed Rewards</span>
                  <span className="text-green-400">{staking?.rewards_earned?.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Status</span>
                  <StakingStatusBadge status={staking?.status} />
                </div>
                {staking?.unbonding_ends_at && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Unbonding Ends</span>
                    <span className="text-yellow-400">{new Date(staking.unbonding_ends_at).toLocaleString()}</span>
                  </div>
                )}
                <div className="flex gap-3 pt-4">
                  <Button variant="primary" className="flex-1" onClick={() => claimRewards.mutate()}>
                    Claim Rewards
                  </Button>
                  <Button variant="secondary" className="flex-1">Stake More</Button>
                  <Button variant="ghost" className="flex-1">Unbond</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-white">Staking Performance</CardTitle>
                <CardDescription>Historical staking metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Performance Score</span>
                    <span className="text-white">{stats?.reputation_score || 0}/100</span>
                  </div>
                  <Progress value={stats?.reputation_score || 0} className="h-2" />
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Total Tasks Processed</span>
                  <span className="text-white">{stats?.total_tasks || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Completed Tasks</span>
                  <span className="text-green-400">{stats?.completed_tasks || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Average Verification Time</span>
                  <span className="text-white">{stats?.average_verification_time || 0}s</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Slashing Events Tab */}
        <TabsContent value="slashing">
          <Card>
            <CardHeader>
              <CardTitle className="text-white">Slashing Events</CardTitle>
              <CardDescription>Historical slashing incidents affecting this validator</CardDescription>
            </CardHeader>
            <CardContent>
              {slashingEvents?.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <ShieldCheck size={40} className="mx-auto mb-2 opacity-30" />
                  <p>No slashing events recorded</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {slashingEvents?.map((event: SlashingEvent) => (
                    <SlashingEventCard key={event.id} event={event} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Delegations Tab */}
        <TabsContent value="delegations">
          <Card>
            <CardHeader>
              <CardTitle className="text-white">Delegations</CardTitle>
              <CardDescription>Users who have delegated tokens to this validator</CardDescription>
            </CardHeader>
            <CardContent>
              {delegations?.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Users size={40} className="mx-auto mb-2 opacity-30" />
                  <p>No delegations yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {delegations?.map((delegation: Delegation) => (
                    <DelegationCard key={delegation.delegator} delegation={delegation} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Helper Components
const StatCard = ({ title, value, icon, trend, color }: { title: string; value: string; icon: React.ReactNode; trend?: number; color?: string }) => (
  <Card>
    <CardContent className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider">{title}</p>
          <p className={`text-2xl font-bold text-white mt-1 ${color || ''}`}>{value}</p>
          {trend !== undefined && (
            <p className={`text-xs mt-1 ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {trend > 0 ? `+${trend}` : trend}%
            </p>
          )}
        </div>
        <div className={`p-2 rounded-lg ${color?.replace('text-', 'bg-')?.replace('-400', '/10') || 'bg-white/5'}`}>
          {icon}
        </div>
      </div>
    </CardContent>
  </Card>
);

const ValidatorStatusBadge = ({ status }: { status: string }) => {
  const config = {
    active: { label: 'Active', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
    inactive: { label: 'Inactive', color: 'bg-slate-500/20 text-slate-400 border-slate-500/30' },
    jailed: { label: 'Jailed', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
    unbonding: { label: 'Unbonding', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  };
  const { label, color } = config[status as keyof typeof config] || config.inactive;
  return <span className={`px-2 py-1 rounded text-xs font-medium ${color}`}>{label}</span>;
};

const StakingStatusBadge = ({ status }: { status: string }) => {
  const config = {
    active: { label: 'Active', icon: <Lock size={12} className="mr-1" /> },
    unbonding: { label: 'Unbonding', icon: <Clock size={12} className="mr-1" /> },
    withdrawn: { label: 'Withdrawn', icon: <Unlock size={12} className="mr-1" /> },
  };
  const { label, icon } = config[status as keyof typeof config] || config.active;
  return <span className="flex items-center text-white">{icon}{label}</span>;
};

const SlashingEventCard = ({ event }: { event: SlashingEvent }) => (
  <div className="flex items-start justify-between p-3 rounded-lg bg-red-500/5 border border-red-500/20">
    <div className="flex items-start gap-3">
      <Gavel size={18} className="text-red-400 mt-0.5" />
      <div>
        <div className="font-medium text-white">{event.reason}</div>
        <div className="text-xs text-slate-400 mt-1">{new Date(event.timestamp).toLocaleString()}</div>
        <div className="text-xs text-slate-500 font-mono mt-1">Evidence: {event.evidence_hash.slice(0, 16)}...</div>
      </div>
    </div>
    <div className="text-right">
      <div className="text-red-400 font-bold">-{event.amount} tokens</div>
      <SlashingSeverityBadge severity={event.severity} />
    </div>
  </div>
);

const SlashingSeverityBadge = ({ severity }: { severity: string }) => {
  const config = {
    low: { label: 'Low', color: 'text-yellow-400 bg-yellow-500/10' },
    medium: { label: 'Medium', color: 'text-orange-400 bg-orange-500/10' },
    high: { label: 'High', color: 'text-red-400 bg-red-500/10' },
  };
  const { label, color } = config[severity as keyof typeof config] || config.low;
  return <span className={`text-xs px-2 py-0.5 rounded mt-1 inline-block ${color}`}>{label}</span>;
};

const DelegationCard = ({ delegation }: { delegation: Delegation }) => (
  <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
    <div>
      <div className="font-medium text-white">{delegation.delegator}</div>
      <div className="text-xs text-slate-400 mt-1">Delegated since {new Date(delegation.since).toLocaleDateString()}</div>
    </div>
    <div className="text-right">
      <div className="text-neon-blue font-bold">{delegation.amount} tokens</div>
      <div className="text-xs text-green-400">+{delegation.rewards} rewards</div>
    </div>
  </div>
);