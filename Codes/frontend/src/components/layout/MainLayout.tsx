import { useState, useEffect, useRef, Suspense } from "react";
import { NavLink, Outlet, useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../../features/auth/store/authStore";
import ConnectionStatus from "./ConnectionStatus";
import { useWindowStore } from "../../store/windowStore";
import { AVAILABLE_WINDOWS } from "../WindowLauncher";
import { Spinner } from "../ui/Spinner";
import { ThemeToggle } from "../ui/ThemeToggle";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { CommandPalette } from "../ui/CommandPalette";
import { KeyboardShortcutsHelp } from "../ui/KeyboardShortcuts";
import {
  Bell,
  Search,
  ChevronLeft,
  ChevronRight,
  User,
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
  Menu,
  X,
  Sparkles,
  Layers,
  HelpCircle,
  Activity,
  CheckCircle2,
} from "lucide-react";

export const MainLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const notificationRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);

  const NOTIFICATIONS_STORAGE_KEY = "svd_platform_notifications_v2";

  const defaultNotifications = [
    {
      id: "1",
      title: "SVD Compression Completed",
      text: 'Experiment "Griffin-SSD MRPC" reached target 0.865 accuracy with 42% parameter reduction.',
      time: "2 min ago",
      read: false,
      type: "experiment",
      link: "/experiments/1c615fbe-7d0e-42ea-b195-27ee5e049867",
    },
    {
      id: "2",
      title: "Local GPU Cluster Active",
      text: "Triton CUDA Engine initialized: 4 nodes online (395.2 TFLOPS).",
      time: "15 min ago",
      read: false,
      type: "system",
      link: "/system-status",
    },
    {
      id: "3",
      title: "New AI Hypothesis Generated",
      text: "AI Co-Scientist synthesized: Dynamic Entropy Rank Scheduling on LLaMA-3.",
      time: "1 hour ago",
      read: true,
      type: "agent",
      link: "/agent",
    },
  ];

  const [notifications, setNotifications] = useState(() => {
    try {
      const saved = localStorage.getItem(NOTIFICATIONS_STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch {
      // Fallback
    }
    return defaultNotifications;
  });

  const [notificationFilter, setNotificationFilter] = useState<"all" | "unread">("all");

  const saveNotifications = (updated: typeof notifications) => {
    setNotifications(updated);
    try {
      localStorage.setItem(NOTIFICATIONS_STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // Ignore
    }
  };

  const markAllNotificationsRead = () => {
    saveNotifications(notifications.map((n) => ({ ...n, read: true })));
  };

  const clearAllNotifications = () => {
    saveNotifications([]);
  };

  const dismissNotification = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    saveNotifications(notifications.filter((n) => n.id !== id));
  };

  const markNotificationRead = (id: string, link?: string) => {
    saveNotifications(notifications.map((n) => (n.id === id ? { ...n, read: true } : n)));
    if (link) {
      navigate(link);
      setShowNotifications(false);
    }
  };

  const handleToggleNotifications = () => {
    const nextState = !showNotifications;
    setShowNotifications(nextState);
    if (nextState) {
      // When opened/seen, automatically mark all as read so the badge vanishes immediately
      markAllNotificationsRead();
    }
  };

  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  const { addWindow } = useWindowStore();

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Click outside listener for dropdowns
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        notificationRef.current &&
        !notificationRef.current.contains(e.target as Node)
      ) {
        setShowNotifications(false);
      }
      if (
        userMenuRef.current &&
        !userMenuRef.current.contains(e.target as Node)
      ) {
        setShowUserMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const coreNavItems = [
    {
      label: "Dashboard",
      path: "/dashboard",
      icon: <LayoutDashboard size={18} />,
    },
    {
      label: "Attention Canvas",
      path: "/attention-canvas",
      icon: <Sparkles size={18} />,
    },
    {
      label: "Transformer Studios",
      path: "/studios/representation",
      icon: <Layers size={18} />,
    },
    {
      label: "Experiments",
      path: "/experiments",
      icon: <FlaskConical size={18} />,
    },
    { label: "Launcher", path: "/launcher", icon: <Rocket size={18} /> },
    { label: "Notes", path: "/notes", icon: <BookOpen size={18} /> },
    { label: "AI Co-pilot", path: "/agent", icon: <Terminal size={18} /> },
  ];

  const platformNavItems = [
    { label: "Marketplace", path: "/marketplace", icon: <Monitor size={18} /> },
    { label: "Analysis", path: "/analysis", icon: <BarChart3 size={18} /> },
    { label: "System Status", path: "/system", icon: <Activity size={18} /> },
    { label: "Support Hub", path: "/support", icon: <HelpCircle size={18} /> },
  ];

  const renderSidebarContent = (isMobile = false) => (
    <div className="flex flex-col h-full">
      {/* Brand Header */}
      <div className="p-4 flex items-center justify-between shrink-0 border-b border-white/5">
        {(!sidebarCollapsed || isMobile) && (
          <NavLink to="/dashboard" className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-gradient-to-br from-neon-blue to-purple-500 rounded-xl flex items-center justify-center shadow-lg shadow-neon-blue/20">
              <img
                src="/logo.jpg"
                alt="NeuralOps"
                className="w-full h-full object-cover rounded-xl"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                }}
              />
            </div>
            <div className="flex flex-col">
              <span className="font-extrabold text-white tracking-tight text-sm">
                NeuralOps
              </span>
              <span className="text-[10px] text-neon-blue font-mono tracking-wider">
                STUDIO v2.0
              </span>
            </div>
          </NavLink>
        )}

        {isMobile ? (
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10"
          >
            <X size={18} />
          </button>
        ) : (
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-white/5 transition-colors shrink-0"
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? (
              <ChevronRight size={18} />
            ) : (
              <ChevronLeft size={18} />
            )}
          </button>
        )}
      </div>

      {/* Nav Menu */}
      <nav className="flex-1 px-3 py-4 space-y-6 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
        <div>
          <p
            className={`text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 ${
              sidebarCollapsed && !isMobile ? "text-center" : "px-3"
            }`}
          >
            Core Apps
          </p>
          <div className="space-y-1">
            {coreNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-neon-blue/15 text-neon-blue font-semibold shadow-sm"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  } ${sidebarCollapsed && !isMobile ? "justify-center" : ""}`
                }
                title={sidebarCollapsed && !isMobile ? item.label : undefined}
              >
                <span className="shrink-0">{item.icon}</span>
                {(!sidebarCollapsed || isMobile) && (
                  <span className="truncate">{item.label}</span>
                )}
              </NavLink>
            ))}
          </div>
        </div>

        <div>
          <p
            className={`text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 ${
              sidebarCollapsed && !isMobile ? "text-center" : "px-3"
            }`}
          >
            Platform & Tools
          </p>
          <div className="space-y-1">
            {platformNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-neon-blue/15 text-neon-blue font-semibold shadow-sm"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  } ${sidebarCollapsed && !isMobile ? "justify-center" : ""}`
                }
                title={sidebarCollapsed && !isMobile ? item.label : undefined}
              >
                <span className="shrink-0">{item.icon}</span>
                {(!sidebarCollapsed || isMobile) && (
                  <span className="truncate">{item.label}</span>
                )}
              </NavLink>
            ))}
          </div>
        </div>

        <div>
          <p
            className={`text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 ${
              sidebarCollapsed && !isMobile ? "text-center" : "px-3"
            }`}
          >
            Floating Workspaces
          </p>
          <div className="space-y-1">
            {AVAILABLE_WINDOWS.slice(0, 6).map((win) => (
              <button
                key={win.type}
                onClick={() => addWindow(win.type, win.title, win.icon)}
                className={`flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-white/5 transition-all w-full ${
                  sidebarCollapsed && !isMobile
                    ? "justify-center"
                    : "justify-start"
                }`}
                title={sidebarCollapsed && !isMobile ? win.title : undefined}
              >
                <span className="text-sm w-5 text-center flex-shrink-0">
                  {win.icon}
                </span>
                {(!sidebarCollapsed || isMobile) && (
                  <span className="truncate">{win.title}</span>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* User Footer */}
      <div className="p-3 border-t border-white/5 shrink-0 bg-white/[0.02]">
        <div
          className={`flex items-center gap-3 ${
            sidebarCollapsed && !isMobile ? "justify-center" : ""
          }`}
        >
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold shadow-md">
            {user?.username?.[0]?.toUpperCase() || "U"}
          </div>
          {(!sidebarCollapsed || isMobile) && (
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-white truncate">
                {user?.username || "Researcher"}
              </p>
              <p className="text-[10px] text-slate-500 truncate">
                {user?.email || "online"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      <ConnectionStatus />
      <div className="min-h-screen flex bg-surface transition-colors duration-300">
        {/* Desktop Sticky Sidebar */}
        <aside
          className={`hidden md:flex flex-col glass border-r border-white/5 transition-all duration-300 sticky top-0 h-screen z-40 ${
            sidebarCollapsed ? "w-20" : "w-64"
          }`}
        >
          {renderSidebarContent(false)}
        </aside>

        {/* Mobile Sidebar Overlay Drawer */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-50 md:hidden flex">
            <div
              className="fixed inset-0 bg-black/70 backdrop-blur-sm animate-fade-in"
              onClick={() => setMobileMenuOpen(false)}
            />
            <div className="relative w-72 max-w-[85vw] h-full glass border-r border-white/10 shadow-2xl flex flex-col z-10 animate-scale-in">
              {renderSidebarContent(true)}
            </div>
          </div>
        )}

        {/* Main content Area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Topbar Header */}
          <header className="sticky top-0 z-30 bg-surface/80 backdrop-blur-xl border-b border-white/5">
            <div className="px-4 md:px-6 py-3 flex items-center justify-between gap-3 md:gap-4">
              {/* Mobile hamburger & search */}
              <div className="flex items-center gap-3 flex-1 max-w-xl">
                <button
                  onClick={() => setMobileMenuOpen(true)}
                  className="md:hidden p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
                  aria-label="Open Navigation Menu"
                >
                  <Menu size={20} />
                </button>

                <div className="flex-1 relative">
                  <Search
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
                  />
                  <input
                    type="text"
                    placeholder="Search experiments, models, notes..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-xs md:text-sm text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        navigate(
                          `/search?q=${encodeURIComponent(
                            (e.target as HTMLInputElement).value,
                          )}`,
                        );
                      }
                    }}
                  />
                </div>
              </div>

              {/* Right Controls (Theme, Notifications, Profile) */}
              <div className="flex items-center gap-2 md:gap-3 shrink-0">
                {/* Theme Toggle Component */}
                <ThemeToggle variant="button" />

                {/* Notifications Dropdown */}
                <div className="relative" ref={notificationRef}>
                  <button
                    onClick={handleToggleNotifications}
                    className="p-2 rounded-xl hover:bg-white/5 text-slate-400 hover:text-white transition-colors relative cursor-pointer"
                    aria-label="Notifications"
                    title="Notifications"
                  >
                    <Bell size={18} />
                    {notifications.filter((n) => !n.read).length > 0 && (
                      <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_#38bdf8] ring-2 ring-[#080d19]" />
                    )}
                  </button>

                  {showNotifications && (
                    <div className="absolute right-0 top-12 w-88 sm:w-96 glass border border-white/10 rounded-2xl shadow-2xl p-4 z-50 animate-scale-in bg-[#0c1222]/95 backdrop-blur-2xl">
                      {/* Header */}
                      <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2.5">
                        <div className="flex items-center gap-2">
                          <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                            Notifications
                          </h3>
                          {notifications.filter((n) => !n.read).length > 0 ? (
                            <span className="px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-mono font-bold">
                              {notifications.filter((n) => !n.read).length} new
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full bg-white/5 text-slate-400 text-[10px] font-mono">
                              0 unread
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={markAllNotificationsRead}
                            className="text-[10px] text-slate-400 hover:text-cyan-300 transition-colors font-medium cursor-pointer"
                          >
                            Mark all read
                          </button>
                          <span className="text-slate-600 text-xs">·</span>
                          <button
                            onClick={clearAllNotifications}
                            className="text-[10px] text-slate-400 hover:text-red-400 transition-colors font-medium cursor-pointer"
                          >
                            Clear all
                          </button>
                        </div>
                      </div>

                      {/* Filter Pills */}
                      <div className="flex items-center justify-between gap-1.5 mb-3">
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => setNotificationFilter("all")}
                            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all cursor-pointer ${
                              notificationFilter === "all"
                                ? "bg-white/10 text-white border border-white/20"
                                : "text-slate-400 hover:text-white"
                            }`}
                          >
                            All ({notifications.length})
                          </button>
                          <button
                            onClick={() => setNotificationFilter("unread")}
                            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all cursor-pointer ${
                              notificationFilter === "unread"
                                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                                : "text-slate-400 hover:text-white"
                            }`}
                          >
                            Unread ({notifications.filter((n) => !n.read).length})
                          </button>
                        </div>
                        {notifications.length > 0 && (
                          <button
                            onClick={clearAllNotifications}
                            className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors cursor-pointer"
                          >
                            Dismiss all
                          </button>
                        )}
                      </div>

                      {/* Notification Items */}
                      <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin pr-1">
                        {notifications
                          .filter((n) => notificationFilter === "all" || !n.read)
                          .map((n) => (
                            <div
                              key={n.id}
                              onClick={() => markNotificationRead(n.id, n.link)}
                              className={`group relative p-3 rounded-xl transition-all cursor-pointer border ${
                                n.read
                                  ? "bg-white/[0.02] border-white/5 opacity-70 hover:opacity-100"
                                  : "bg-cyan-950/20 border-cyan-500/30 hover:border-cyan-500/60 shadow-lg shadow-cyan-500/5"
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1 pr-5">
                                <span className="text-xs font-bold text-slate-200">
                                  {n.title}
                                </span>
                                <span className="text-[10px] text-slate-500 font-mono">
                                  {n.time}
                                </span>
                              </div>
                              <p className="text-xs text-slate-300 leading-relaxed pr-4">
                                {n.text}
                              </p>
                              {/* Dismiss X button */}
                              <button
                                onClick={(e) => dismissNotification(n.id, e)}
                                className="absolute top-2.5 right-2.5 p-1 text-slate-500 hover:text-red-400 rounded-md hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-all cursor-pointer"
                                title="Dismiss notification"
                              >
                                <X size={12} />
                              </button>
                            </div>
                          ))}
                        {notifications.filter((n) => notificationFilter === "all" || !n.read).length === 0 && (
                          <div className="py-8 text-center text-xs text-slate-500 font-mono space-y-1">
                            <CheckCircle2 size={24} className="mx-auto text-slate-600 mb-2" />
                            <p>No notifications to display</p>
                            <p className="text-[10px] text-slate-600">All caught up</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* User Profile Menu */}
                <div className="relative" ref={userMenuRef}>
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center gap-2 glass py-1.5 px-2.5 rounded-xl hover:border-white/20 transition-all"
                  >
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                      {user?.username?.[0]?.toUpperCase() || "U"}
                    </div>
                    <ChevronDown
                      size={14}
                      className={`text-slate-400 transition-transform duration-200 hidden sm:block ${
                        showUserMenu ? "rotate-180" : ""
                      }`}
                    />
                  </button>

                  {showUserMenu && (
                    <div className="absolute right-0 mt-2 w-48 glass border border-white/10 rounded-xl overflow-hidden shadow-2xl z-50 animate-scale-in">
                      <div className="px-4 py-2.5 border-b border-white/5 bg-white/[0.02]">
                        <p className="text-xs font-bold text-white truncate">
                          {user?.username || "Researcher"}
                        </p>
                        <p className="text-[10px] text-slate-400 truncate">
                          {user?.email || "Workspace User"}
                        </p>
                      </div>
                      <NavLink
                        to="/profile"
                        onClick={() => setShowUserMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-xs text-slate-300 hover:bg-white/5 hover:text-white transition-colors"
                      >
                        <User size={14} /> Profile
                      </NavLink>
                      <NavLink
                        to="/profile/change-password"
                        onClick={() => setShowUserMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-xs text-slate-300 hover:bg-white/5 hover:text-white transition-colors"
                      >
                        <KeyRound size={14} /> Password
                      </NavLink>
                      <hr className="border-white/5" />
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          logout();
                        }}
                        className="w-full flex items-center gap-2.5 px-4 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <LogOut size={14} /> Logout
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </header>

          {/* Main Content Router Outlet wrapped in ErrorBoundary and Suspense */}
          <main className="flex-1 p-4 md:p-8 overflow-y-auto flex flex-col">
            <ErrorBoundary>
              <Suspense
                fallback={
                  <div className="flex-1 w-full min-h-[70vh] flex flex-col items-center justify-center m-auto p-8 animate-fade-in text-center">
                    <Spinner size="xl" label="Synchronizing NeuralOps Quantum Workspace..." />
                  </div>
                }
              >
                <Outlet />
              </Suspense>
            </ErrorBoundary>
          </main>
        </div>
      </div>
      <CommandPalette />
      <KeyboardShortcutsHelp />
    </>
  );
};

export default MainLayout;
