import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, DollarSign, Briefcase, FileText, Settings, ChevronLeft, ChevronRight } from 'lucide-react';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/earnings', icon: DollarSign, label: 'Earnings' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs' },
  { to: '/logs', icon: FileText, label: 'Logs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`${collapsed ? 'w-20' : 'w-64'} bg-dark-card border-r border-dark-border flex flex-col transition-all duration-300`}>
      <div className="flex items-center justify-between p-4 border-b border-dark-border">
        {!collapsed && <h1 className="text-xl font-bold bg-gradient-to-r from-primary-400 to-primary-600 bg-clip-text text-transparent">NeuralOps</h1>}
        <button onClick={onToggle} className="p-1 hover:bg-dark-border rounded-lg text-gray-400">
          {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>
      <nav className="flex-1 py-4">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 mx-2 rounded-lg transition-colors ${isActive ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-dark-border hover:text-white'} ${collapsed ? 'justify-center' : ''}`
            }
          >
            <Icon size={20} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}