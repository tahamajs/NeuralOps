import React, { useState, useRef, useEffect } from 'react';
import { Bell, BellRing, X, CheckCheck, DollarSign, AlertCircle, Info } from 'lucide-react';
import { useNotifications } from '../context/NotificationContext';

const typeConfig = {
  earnings: { icon: DollarSign, color: 'text-green-400', bg: 'bg-green-400/10' },
  warning: { icon: AlertCircle, color: 'text-orange-400', bg: 'bg-orange-400/10' },
  info: { icon: Info, color: 'text-blue-400', bg: 'bg-blue-400/10' },
};

export default function Notifications() {
  const { notifications, unreadCount, markAllRead, markRead } = useNotifications();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handleClick = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(!open)} className="relative p-2 hover:bg-dark-border rounded-lg transition-colors">
        {unreadCount > 0 ? <BellRing size={20} className="text-primary-400" /> : <Bell size={20} className="text-gray-400" />}
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">{unreadCount > 9 ? '9+' : unreadCount}</span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-12 w-96 bg-dark-card border border-dark-border rounded-xl shadow-2xl z-50 max-h-[500px] overflow-hidden">
          <div className="flex items-center justify-between p-4 border-b border-dark-border">
            <h3 className="font-semibold">Notifications</h3>
            <button onClick={markAllRead} className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1"><CheckCheck size={14} /> Mark all read</button>
          </div>
          <div className="overflow-y-auto max-h-[400px]">
            {notifications.length === 0 ? (
              <div className="p-8 text-center text-gray-500"><Bell size={32} className="mx-auto mb-2 opacity-30" /><p>No notifications yet</p></div>
            ) : (
              notifications.map(n => {
                const cfg = typeConfig[n.type] || typeConfig.info;
                const Icon = cfg.icon;
                return (
                  <div key={n.id} onClick={() => markRead(n.id)}
                    className={`flex gap-3 p-4 border-b border-dark-border hover:bg-dark-border/50 cursor-pointer transition-colors ${!n.read ? 'bg-primary-600/5' : ''}`}>
                    <div className={`p-2 rounded-lg shrink-0 ${cfg.bg}`}><Icon size={16} className={cfg.color} /></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{n.title}</p>
                      <p className="text-xs text-gray-400 mt-0.5 truncate">{n.message}</p>
                      <p className="text-xs text-gray-500 mt-1">{new Date(n.timestamp).toLocaleTimeString()}</p>
                    </div>
                    {!n.read && <div className="w-2 h-2 bg-primary-500 rounded-full shrink-0 mt-1" />}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}