import React, { useEffect, useState } from 'react';
import { Clock, CheckCircle, XCircle, Loader } from 'lucide-react';
import { api } from '../services/api';

const jobStatusIcon = { completed: CheckCircle, running: Loader, failed: XCircle, queued: Clock };
const jobStatusColor = { completed: 'text-green-400', running: 'text-blue-400 animate-spin', failed: 'text-red-400', queued: 'text-gray-400' };

export default function Jobs() {
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    const fetchJobs = () => api.getJobs({}).then(d => setJobs(d.results || []));
    fetchJobs();
    const t = setInterval(fetchJobs, 10000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Job History</h2>
      <div className="bg-dark-card border border-dark-border rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-dark-border text-left text-gray-400 text-sm">
              <th className="py-3 px-4">Job ID</th>
              <th className="py-3 px-4">Model</th>
              <th className="py-3 px-4">Price/hr</th>
              <th className="py-3 px-4">Duration</th>
              <th className="py-3 px-4">Earnings</th>
              <th className="py-3 px-4">Status</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(job => {
              const Icon = jobStatusIcon[job.status] || Clock;
              return (
                <tr key={job.id} className="border-b border-dark-border hover:bg-dark-border/30 transition-colors">
                  <td className="py-3 px-4 font-mono text-xs">{job.id.slice(0,8)}</td>
                  <td className="py-3 px-4">{job.listing_gpu_model || 'N/A'}</td>
                  <td className="py-3 px-4">${job.price_per_hour}</td>
                  <td className="py-3 px-4">{job.hours_consumed?.toFixed(2) || '—'}h</td>
                  <td className="py-3 px-4">${job.total_cost_usd || '0.00'}</td>
                  <td className="py-3 px-4"><span className={`flex items-center gap-1 ${jobStatusColor[job.status]}`}><Icon size={14} />{job.status}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {jobs.length === 0 && <p className="p-4 text-center text-gray-500">No jobs yet</p>}
      </div>
    </div>
  );
}