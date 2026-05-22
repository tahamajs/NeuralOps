import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Progress } from '../../../components/ui/Progress'
import { Cpu, HardDrive, Zap, Database } from 'lucide-react'

export function SystemResourceWidget({ resources }: any) {
  if (!resources) return <div className="text-center text-slate-500 py-4">No resource data</div>
  return <div className="space-y-3"><div><div className="flex justify-between text-xs"><span>CPU Usage</span><span>{resources.cpu_usage_percent}%</span></div><Progress value={resources.cpu_usage_percent} className="h-1" /></div><div><div className="flex justify-between text-xs"><span>Memory</span><span>{resources.memory_usage_gb} / {resources.memory_total_gb} GB</span></div><Progress value={(resources.memory_usage_gb / resources.memory_total_gb) * 100} className="h-1" /></div><div><div className="flex justify-between text-xs"><span>Disk</span><span>{resources.disk_usage_percent}%</span></div><Progress value={resources.disk_usage_percent} className="h-1" /></div><div><div className="flex justify-between text-xs"><span>GPU Utilization</span><span>{resources.gpu_utilization}%</span></div><Progress value={resources.gpu_utilization} className="h-1" /></div></div>
}