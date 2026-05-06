import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import { HardwareProvider } from './contexts/HardwareContext';
import DashboardPage from './pages/DashboardPage';
import ExperimentListPage from './pages/ExperimentListPage';
import ExperimentDetailPage from './pages/ExperimentDetailPage';
import LauncherPage from './pages/LauncherPage';
export default function App() {
    const location = useLocation();
    const navItems = [
        { label: 'Dashboard', path: '/dashboard' },
        { label: 'Experiments', path: '/experiments' },
        { label: 'Launcher', path: '/launcher' },
    ];
    return (_jsx(HardwareProvider, { children: _jsxs("div", { className: "min-h-screen bg-[#0a0c10] cyber-grid", children: [_jsx("nav", { className: "sticky top-0 z-50 bg-[#0a0c10]/70 backdrop-blur-xl border-b border-white/5 px-6 py-4", children: _jsxs("div", { className: "max-w-7xl mx-auto flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-12", children: [_jsxs(Link, { to: "/dashboard", className: "flex items-center gap-3 group", children: [_jsx("div", { className: "w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-110 transition-transform", children: _jsx("span", { className: "text-white font-black text-sm tracking-tighter", children: "ET" }) }), _jsxs("div", { className: "flex flex-col", children: [_jsxs("span", { className: "text-xl font-black text-white leading-none tracking-tighter", children: ["NEURAL", _jsx("span", { className: "text-blue-500", children: "OPS" })] }), _jsx("span", { className: "text-[10px] font-bold text-slate-500 tracking-[0.3em] uppercase", children: "Control Suite" })] })] }), _jsx("div", { className: "hidden md:flex items-center gap-2", children: navItems.map((item) => (_jsx(Link, { to: item.path, className: `px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${location.pathname.startsWith(item.path)
                                                ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20'
                                                : 'text-slate-500 hover:text-white hover:bg-white/5'}`, children: item.label }, item.path))) })] }), _jsxs("div", { className: "flex items-center gap-6", children: [_jsxs("div", { className: "hidden sm:flex flex-col items-end", children: [_jsx("span", { className: "text-[10px] font-bold text-slate-500 uppercase tracking-widest", children: "Global Status" }), _jsx("span", { className: "text-xs font-black text-green-400 uppercase tracking-tighter", children: "Operational" })] }), _jsx("div", { className: "h-10 w-[1px] bg-white/5 mx-2" }), _jsx("div", { className: "w-8 h-8 rounded-full bg-white/10 border border-white/10 flex items-center justify-center", children: _jsx("div", { className: "w-2 h-2 rounded-full bg-blue-500 animate-pulse" }) })] })] }) }), _jsx("main", { className: "max-w-7xl mx-auto p-4 md:p-10", children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Navigate, { to: "/dashboard", replace: true }) }), _jsx(Route, { path: "/dashboard", element: _jsx(DashboardPage, {}) }), _jsx(Route, { path: "/experiments", element: _jsx(ExperimentListPage, {}) }), _jsx(Route, { path: "/experiments/:id", element: _jsx(ExperimentDetailPage, {}) }), _jsx(Route, { path: "/launcher", element: _jsx(LauncherPage, {}) })] }) })] }) }));
}
