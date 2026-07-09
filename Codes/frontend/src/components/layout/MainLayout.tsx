import { useState, Suspense } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuthStore } from "../../features/auth/store/authStore";
import ConnectionStatus from "./ConnectionStatus";
import { useWindowStore } from "../../store/windowStore";
import { AVAILABLE_WINDOWS } from "../WindowLauncher";
import { Spinner } from "../ui/Spinner";
import {
  Bell,
  Search,
  Sun,
  Moon,
  ChevronLeft,
  ChevronRight,
  User,
  Settings,
  LogOut,
  LayoutDashboard,
  FlaskConical,
  Rocket,
  BookOpen,
  BarChart3,
  FileText,
  KeyRound,
  ChevronDown,
  Terminal,
  Monitor,
} from "lucide-react";

export const MainLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [notifications] = useState([
    {
      id: 1,
      text: 'Experiment "bert-mrpc" completed',
      time: "5 min ago",
      read: false,
    },
    { id: 2, text: "New note shared with you", time: "1 hour ago", read: true },
  ]);
  const [showNotifications, setShowNotifications] = useState(false);
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const { addWindow } = useWindowStore();

  const navItems = [
    {
      label: "Dashboard",
      path: "/dashboard",
      icon: <LayoutDashboard size={18} />,
    },
    {
      label: "Experiments",
      path: "/experiments",
      icon: <FlaskConical size={18} />,
    },
    { label: "Launcher", path: "/launcher", icon: <Rocket size={18} /> },
    { label: "Notes", path: "/notes", icon: <BookOpen size={18} /> },
    { label: "Analysis", path: "/analysis", icon: <BarChart3 size={18} /> },
    { label: "Papers", path: "/paper/editor", icon: <FileText size={18} /> },
    { label: "AI Agent", path: "/agent", icon: <Terminal size={18} /> },
    { label: "Marketplace", path: "/marketplace", icon: <Monitor size={18} /> },
    { label: "Support", path: "/support", icon: <BookOpen size={18} /> },
    { label: "System", path: "/system", icon: <Monitor size={18} /> },
  ];

  return (
    <>
      <ConnectionStatus />
      <div
        className={`min-h-screen flex ${darkMode ? "bg-surface" : "bg-gray-50"}`}
      >
        {/* Sidebar */}
        <aside
          className={`${
            sidebarCollapsed ? "w-20" : "w-64"
          } glass border-r border-white/5 flex flex-col transition-all duration-300 sticky top-0 h-screen z-40`}
        >
          <div className="p-4 flex items-center justify-between shrink-0">
            {!sidebarCollapsed && (
              <NavLink to="/dashboard" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-neon-blue to-purple-500 rounded-lg flex items-center justify-center">
                  <img src="/logo.jpg" alt="NeuralOps" className="w-full h-full object-cover rounded-md" />
                </div>
                <span className="font-bold text-white tracking-tight">
                  NeuralOps
                </span>
              </NavLink>
            )}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="text-slate-400 hover:text-white p-1 rounded-lg shrink-0"
            >
              {sidebarCollapsed ? (
                <ChevronRight size={18} />
              ) : (
                <ChevronLeft size={18} />
              )}
            </button>
          </div>

          <nav className="flex-1 px-3 space-y-6 overflow-y-auto pb-6 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
            <div>
              <p className={`text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 ${sidebarCollapsed ? 'text-center' : 'px-3'}`}>
                Core Apps
              </p>
              <div className="space-y-1">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                        isActive
                          ? "bg-neon-blue/10 text-neon-blue"
                          : "text-slate-400 hover:text-white hover:bg-white/5"
                      } ${sidebarCollapsed ? "justify-center" : ""}`
                    }
                    title={sidebarCollapsed ? item.label : undefined}
                  >
                    {item.icon}
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </NavLink>
                ))}
              </div>
            </div>

            <div>
              <p className={`text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 ${sidebarCollapsed ? 'text-center' : 'px-3'}`}>
                Features & Tools
              </p>
              <div className="space-y-1">
                {AVAILABLE_WINDOWS.map((win) => (
                  <button
                    key={win.type}
                    onClick={() => addWindow(win.type, win.title, win.icon)}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-white hover:bg-white/5 transition-all w-full ${sidebarCollapsed ? "justify-center" : "justify-start"}`}
                    title={sidebarCollapsed ? win.title : undefined}
                  >
                    <span className="text-lg w-5 text-center flex-shrink-0">{win.icon}</span>
                    {!sidebarCollapsed && <span className="truncate">{win.title}</span>}
                  </button>
                ))}
              </div>
            </div>
          </nav>

          <div className="p-4 border-t border-white/5">
            <div
              className={`flex items-center gap-3 ${sidebarCollapsed ? "justify-center" : ""}`}
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                {user?.username?.[0]?.toUpperCase() || "U"}
              </div>
              {!sidebarCollapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">
                    {user?.username || "User"}
                  </p>
                  <p className="text-[10px] text-slate-500">Researcher</p>
                </div>
              )}
            </div>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          <header
            className={`sticky top-0 z-30 ${
              darkMode ? "bg-surface/80" : "bg-white/80"
            } backdrop-blur-xl border-b border-white/5`}
          >
            <div className="px-6 py-3 flex items-center justify-between gap-4">
              <div className="flex-1 max-w-xl relative">
                <Search
                  size={16}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                />
                <input
                  type="text"
                  placeholder="Search experiments, notes, papers..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-sm text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      navigate(
                        `/search?q=${encodeURIComponent((e.target as HTMLInputElement).value)}`,
                      );
                    }
                  }}
                />
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                >
                  {darkMode ? <Sun size={18} /> : <Moon size={18} />}
                </button>

                <div className="relative">
                  <button
                    onClick={() => setShowNotifications(!showNotifications)}
                    className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors relative"
                  >
                    <Bell size={18} />
                    {notifications.filter((n) => !n.read).length > 0 && (
                      <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                    )}
                  </button>
                  {showNotifications && (
                    <div className="absolute right-0 top-12 w-80 glass border border-white/10 rounded-2xl shadow-2xl p-4 z-50">
                      <h3 className="text-sm font-bold text-white mb-3">
                        Notifications
                      </h3>
                      {notifications.map((n) => (
                        <div
                          key={n.id}
                          className={`py-2 border-b border-white/5 last:border-0 ${n.read ? "opacity-60" : ""}`}
                        >
                          <p className="text-xs text-white">{n.text}</p>
                          <p className="text-[10px] text-slate-500">{n.time}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="relative group">
                  <button className="flex items-center gap-2 glass py-1.5 px-3 rounded-xl">
                    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                      {user?.username?.[0]?.toUpperCase() || "U"}
                    </div>
                    <ChevronDown size={14} className="text-slate-400" />
                  </button>
                  <div className="absolute right-0 mt-2 w-48 glass border border-white/10 rounded-xl overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                    <NavLink
                      to="/profile"
                      className="flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white"
                    >
                      <User size={14} /> Profile
                    </NavLink>
                    <NavLink
                      to="/profile/change-password"
                      className="flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white"
                    >
                      <KeyRound size={14} /> Password
                    </NavLink>
                    <hr className="border-white/5" />
                    <button
                      onClick={logout}
                      className="w-full flex items-center gap-2 px-4 py-2 text-xs text-red-400 hover:bg-white/5"
                    >
                      <LogOut size={14} /> Logout
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </header>

          <main className="flex-1 p-6 md:p-8">
            <Suspense fallback={<Spinner className="py-8" />}>
              <Outlet />
            </Suspense>
          </main>
        </div>
      </div>
    </>
  );
};
