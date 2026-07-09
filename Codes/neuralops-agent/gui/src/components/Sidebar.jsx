import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, DollarSign, Briefcase, FileText, Settings, Cpu, Info, ChevronLeft, ChevronRight } from 'lucide-react';
import { useNotifications } from '../context/NotificationContext';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/earnings', icon: DollarSign, label: 'Earnings' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs' },
  { to: '/logs', icon: FileText, label: 'Logs' },
  { to: '/gpu', icon: Cpu, label: 'GPU Details' },
  { to: '/settings', icon: Settings, label: 'Settings' },
  { to: '/about', icon: Info, label: 'About' },
];

export default function Sidebar({ collapsed, onToggle }) {
  const { unreadCount } = useNotifications();

  return (
    <aside className={`${collapsed ? 'w-20' : 'w-64'} bg-dark-card border-r border-dark-border flex flex-col transition-all duration-300`}>
      <div className="flex items-center justify-between p-4 border-b border-dark-border">
        {!collapsed && <h1 className="text-xl font-bold bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">NeuralOps</h1>}
        <button onClick={onToggle} className="p-1 hover:bg-dark-border rounded-lg text-gray-400"><Icon size={20} /></button>
      </div>
      <nav className="flex-1 py-4">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) =>
            `flex items-center gap-3 px-4 py-3 mx-2 rounded-lg transition-colors ${isActive ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-dark-border hover:text-white'} ${collapsed ? 'justify-center' : ''}`
          }>
            <Icon size={20} />
            {!collapsed && <span>{label}</span>}
            {!collapsed && to === '/jobs' && unreadCount > 0 && (
              <span className="ml-auto bg-red-500 text-white text-xs font-bold px-1.5 py-0.5 rounded-full">{unreadCount}</span>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-dark-border">
        {!collapsed && <p className="text-xs text-gray-500 text-center">NeuralOps GPU Agent v1.0.0</p>}
      </div>
    </aside>
  );
}