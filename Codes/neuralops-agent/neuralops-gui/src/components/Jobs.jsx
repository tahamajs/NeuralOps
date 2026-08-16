import React, { useEffect, useState } from 'react';
import {
  Clock,
  CheckCircle,
  XCircle,
  Loader,
  RefreshCw,
  Terminal,
  Square,
  Pause,
  Play,
  Filter,
} from 'lucide-react';
import { api } from '../services/api';

const jobStatusIcon = {
  completed: CheckCircle,
  running: Loader,
  failed: XCircle,
  queued: Clock,
  paused: Pause,
};

const jobStatusColor = {
  completed: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  running: 'text-blue-400 bg-blue-400/10 border-blue-400/20 animate-pulse',
  failed: 'text-rose-400 bg-rose-400/10 border-rose-400/20',
  queued: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
  paused: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
};

export default function Jobs() {
  const [jobs, setJobs] = useState([]);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [selectedLogs, setSelectedLogs] = useState(null);
  const [logContent, setLogContent] = useState('');

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const resp = await api.getJobs({});
      setJobs(Array.isArray(resp) ? resp : resp?.results || []);
    } catch (e) {
      console.warn('Failed to fetch jobs:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleContainerAction = async (job, action) => {
    if (!window.neuralopsAPI || !job.container_id) return;
    try {
      if (action === 'pause') await window.neuralopsAPI.containerPause(job.container_id);
      if (action === 'unpause') await window.neuralopsAPI.containerUnpause(job.container_id);
      if (action === 'stop') await window.neuralopsAPI.containerStop(job.container_id);
      fetchJobs();
    } catch (e) {
      console.error(`Error executing ${action} on container:`, e);
    }
  };

  const handleViewLogs = async (job) => {
    setSelectedLogs(job);
    setLogContent('Loading container output...');
    if (window.neuralopsAPI && job.container_id) {
      try {
        const logs = await window.neuralopsAPI.containerLogs(job.container_id, 300);
        setLogContent(logs || 'No output recorded yet.');
      } catch (e) {
        setLogContent(`Error fetching logs: ${e.message}`);
      }
    } else {
      setLogContent(job.logs || 'Execution logs are streaming to backend...');
    }
  };

  const filteredJobs = jobs.filter((j) => {
    if (filter === 'all') return true;
    return j.status === filter;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Compute Jobs & Containers</h2>
          <p className="text-sm text-gray-400 mt-1">
            Track running container executions, resource allocations, and job payouts.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Status Filter */}
          <div className="flex items-center bg-dark-card border border-dark-border rounded-xl p-1 text-xs">
            {['all', 'running', 'completed', 'failed'].map((st) => (
              <button
                key={st}
                onClick={() => setFilter(st)}
                className={`px-3 py-1.5 rounded-lg capitalize font-medium transition-colors ${
                  filter === st
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {st}
              </button>
            ))}
          </div>

          <button
            onClick={fetchJobs}
            disabled={loading}
            className="p-2.5 rounded-xl border border-dark-border bg-dark-card hover:bg-dark-border text-gray-300 hover:text-white transition-colors"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-dark-border bg-dark-bg/60 text-xs font-semibold text-gray-400">
                <th className="py-3.5 px-4">Job ID</th>
                <th className="py-3.5 px-4">Model / Framework</th>
                <th className="py-3.5 px-4">Rate</th>
                <th className="py-3.5 px-4">Consumed Time</th>
                <th className="py-3.5 px-4">Payout</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-border">
              {filteredJobs.map((job) => {
                const StatusIcon = jobStatusIcon[job.status] || Clock;
                const statusBadge = jobStatusColor[job.status] || 'text-gray-400 bg-gray-500/10';

                return (
                  <tr key={job.id} className="hover:bg-dark-border/20 transition-colors">
                    <td className="py-3.5 px-4 font-mono text-xs text-primary-300 font-semibold">
                      {job.id.slice(0, 8)}
                    </td>
                    <td className="py-3.5 px-4 text-white font-medium">
                      {job.listing_gpu_model || job.model_name || 'Standard Compute'}
                    </td>
                    <td className="py-3.5 px-4 text-gray-300">
                      ${parseFloat(job.price_per_hour || 0.5).toFixed(2)}/hr
                    </td>
                    <td className="py-3.5 px-4 text-gray-300">
                      {job.hours_consumed ? `${job.hours_consumed.toFixed(2)}h` : 'Active'}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-emerald-400">
                      ${parseFloat(job.total_cost_usd || 0).toFixed(2)}
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${statusBadge}`}
                      >
                        <StatusIcon size={13} className={job.status === 'running' ? 'animate-spin' : ''} />
                        <span className="capitalize">{job.status}</span>
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => handleViewLogs(job)}
                          title="View Execution Logs"
                          className="p-1.5 hover:bg-dark-border rounded-lg text-gray-400 hover:text-white transition-colors"
                        >
                          <Terminal size={15} />
                        </button>
                        {job.status === 'running' && job.container_id && (
                          <>
                            <button
                              onClick={() => handleContainerAction(job, 'pause')}
                              title="Pause Container"
                              className="p-1.5 hover:bg-dark-border rounded-lg text-amber-400 hover:text-amber-300"
                            >
                              <Pause size={15} />
                            </button>
                            <button
                              onClick={() => handleContainerAction(job, 'stop')}
                              title="Stop Container"
                              className="p-1.5 hover:bg-dark-border rounded-lg text-rose-400 hover:text-rose-300"
                            >
                              <Square size={15} />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filteredJobs.length === 0 && (
          <div className="py-12 px-4 text-center">
            <Clock size={36} className="mx-auto text-gray-600 mb-2 opacity-60" />
            <p className="text-gray-400 font-medium">No compute jobs found</p>
            <p className="text-xs text-gray-500 mt-1">
              Ensure your agent is running and online to accept inbound marketplace jobs.
            </p>
          </div>
        )}
      </div>

      {/* Container Logs Modal */}
      {selectedLogs && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={() => setSelectedLogs(null)}
        >
          <div
            className="bg-dark-card border border-dark-border rounded-2xl w-full max-w-3xl overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-dark-border bg-dark-bg/80">
              <div className="flex items-center gap-2">
                <Terminal size={18} className="text-primary-400" />
                <h3 className="font-semibold text-white text-sm">
                  Container Logs — Job #{selectedLogs.id.slice(0, 8)}
                </h3>
              </div>
              <button
                onClick={() => setSelectedLogs(null)}
                className="text-gray-400 hover:text-white text-sm px-2 py-1 rounded-lg hover:bg-dark-border"
              >
                ✕
              </button>
            </div>
            <div className="p-4 bg-black/90 font-mono text-xs text-emerald-400 max-h-[60vh] overflow-y-auto whitespace-pre-wrap">
              {logContent}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}