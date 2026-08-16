import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  DollarSign,
  Briefcase,
  FileText,
  Settings,
  Cpu,
  Info,
  Activity,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useNotifications } from '../context/NotificationContext';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/live', icon: Activity, label: 'Live Telemetry' },
  { to: '/gpu', icon: Cpu, label: 'GPU Details' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs & Containers' },
  { to: '/earnings', icon: DollarSign, label: 'Earnings' },
  { to: '/logs', icon: FileText, label: 'Logs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
  { to: '/about', icon: Info, label: 'About' },
];

export default function Sidebar({ collapsed, onToggle }) {
  const { unreadCount } = useNotifications();

  return (
    <aside
      className={`${
        collapsed ? 'w-20' : 'w-64'
      } bg-dark-card border-r border-dark-border flex flex-col transition-all duration-300 select-none z-20`}
    >
      {/* Header / Brand */}
      <div className="flex items-center justify-between p-4 border-b border-dark-border h-16">
        {!collapsed ? (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-primary-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-primary-500/20">
              N
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-primary-400 via-purple-300 to-indigo-400 bg-clip-text text-transparent leading-none">
                NeuralOps
              </h1>
              <span className="text-[10px] text-gray-400 font-mono tracking-wider">GPU AGENT</span>
            </div>
          </div>
        ) : (
          <div className="w-8 h-8 mx-auto rounded-lg bg-gradient-to-tr from-primary-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg">
            N
          </div>
        )}
        <button
          onClick={onToggle}
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          className="p-1.5 hover:bg-dark-border rounded-lg text-gray-400 hover:text-white transition-colors"
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      {/* Nav Menu */}
      <nav className="flex-1 py-4 space-y-1 overflow-y-auto px-2">
        {navItems.map(({ to, icon: NavIcon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 ${
                isActive
                  ? 'bg-primary-600 text-white font-medium shadow-md shadow-primary-600/30'
                  : 'text-gray-400 hover:bg-dark-border/60 hover:text-gray-100'
              } ${collapsed ? 'justify-center' : ''}`
            }
          >
            <NavIcon size={18} className="shrink-0" />
            {!collapsed && <span className="truncate text-sm">{label}</span>}
            {!collapsed && to === '/jobs' && unreadCount > 0 && (
              <span className="ml-auto bg-red-500 text-white text-[11px] font-bold px-1.5 py-0.5 rounded-full leading-none">
                {unreadCount}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer info */}
      <div className="p-4 border-t border-dark-border bg-dark-bg/30">
        {!collapsed ? (
          <div className="text-center">
            <p className="text-[11px] text-gray-400 font-medium">NeuralOps GPU Provider</p>
            <p className="text-[10px] text-gray-500 font-mono mt-0.5">v1.0.0 · Core Suite</p>
          </div>
        ) : (
          <div className="w-2 h-2 rounded-full bg-green-500 mx-auto" title="Agent Ready" />
        )}
      </div>
    </aside>
  );
}