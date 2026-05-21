import React, { useState } from 'react';
import { Clock, CheckCircle, XCircle, Loader } from 'lucide-react';

const jobStatusIcon = {
  completed: CheckCircle,
  running: Loader,
  failed: XCircle,
  queued: Clock,
};
const jobStatusColor = {
  completed: 'text-green-400',
  running: 'text-blue-400 animate-spin',
  failed: 'text-red-400',
  queued: 'text-gray-400',
};

const sampleJobs = [
  { id: 'job-001', status: 'running', model: 'ResNet-50', earnings: '$0.45', duration: '12 min' },
  { id: 'job-002', status: 'completed', model: 'BERT-Base', earnings: '$1.20', duration: '45 min' },
  { id: 'job-003', status: 'failed', model: 'GPT-2', earnings: '$0.00', duration: '2 min' },
  { id: 'job-004', status: 'queued', model: 'Llama-2-7B', earnings: '--', duration: '--' },
];

export default function Jobs() {
  const [jobs] = useState(sampleJobs);

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Job History</h2>
      <div className="bg-dark-card border border-dark-border rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-dark-border text-left text-gray-400 text-sm">
              <th className="py-3 px-4">Job ID</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4">Model</th>
              <th className="py-3 px-4">Earnings</th>
              <th className="py-3 px-4">Duration</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const Icon = jobStatusIcon[job.status];
              return (
                <tr key={job.id} className="border-b border-dark-border hover:bg-dark-border/50 transition-colors">
                  <td className="py-3 px-4 font-mono text-sm">{job.id}</td>
                  <td className="py-3 px-4">
                    <span className={`flex items-center gap-2 ${jobStatusColor[job.status]}`}>
                      <Icon size={16} /> {job.status}
                    </span>
                  </td>
                  <td className="py-3 px-4">{job.model}</td>
                  <td className="py-3 px-4">{job.earnings}</td>
                  <td className="py-3 px-4">{job.duration}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}