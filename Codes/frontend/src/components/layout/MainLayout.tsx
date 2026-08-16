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
} from "lucide-react";

export const MainLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const notificationRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);

  const [notifications] = useState([
    {
      id: 1,
      text: 'Experiment "bert-mrpc" completed successfully',
      time: "5 min ago",
      read: false,
    },
    {
      id: 2,
      text: "New note shared with you by Research Team",
      time: "1 hour ago",
      read: true,
    },
  ]);

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
    { label: "Paper Studio", path: "/paper/editor", icon: <FileText size={18} /> },
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
                    onClick={() => setShowNotifications(!showNotifications)}
                    className="p-2 rounded-xl hover:bg-white/5 text-slate-400 hover:text-white transition-colors relative"
                    aria-label="Notifications"
                  >
                    <Bell size={18} />
                    {notifications.filter((n) => !n.read).length > 0 && (
                      <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-neon-blue rounded-full shadow-sm shadow-neon-blue" />
                    )}
                  </button>

                  {showNotifications && (
                    <div className="absolute right-0 top-12 w-80 glass border border-white/10 rounded-2xl shadow-2xl p-4 z-50 animate-scale-in">
                      <div className="flex items-center justify-between mb-3 border-b border-white/5 pb-2">
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                          Notifications
                        </h3>
                        <span className="text-[10px] text-neon-blue font-semibold">
                          {notifications.filter((n) => !n.read).length} new
                        </span>
                      </div>
                      <div className="space-y-2">
                        {notifications.map((n) => (
                          <div
                            key={n.id}
                            className={`p-2.5 rounded-xl transition-colors ${
                              n.read
                                ? "bg-white/[0.02] opacity-70"
                                : "bg-neon-blue/5 border border-neon-blue/20"
                            }`}
                          >
                            <p className="text-xs text-white leading-relaxed">
                              {n.text}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">
                              {n.time}
                            </p>
                          </div>
                        ))}
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
          <main className="flex-1 p-4 md:p-8 overflow-y-auto">
            <ErrorBoundary>
              <Suspense
                fallback={
                  <div className="min-h-[400px] flex flex-col items-center justify-center gap-3">
                    <Spinner size="lg" />
                    <p className="text-xs text-slate-500 font-mono tracking-wider animate-pulse">
                      Loading NeuralOps module...
                    </p>
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
