import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../api/client';
import { useAuthStore } from '../store/authStore';
export default function LoginPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const setAuth = useAuthStore((s) => s.setAuth);
    const navigate = useNavigate();
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const res = await api.post('/auth/login/', { username, password });
            const { token, ...user } = res.data;
            setAuth(user, token);
            navigate('/dashboard');
        }
        catch (err) {
            setError(err.response?.data?.error || 'Failed to login');
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsx("div", { className: "min-h-[80vh] flex items-center justify-center", children: _jsxs("div", { className: "glass-card w-full max-w-md p-8 bg-slate-900/50 border border-white/10 backdrop-blur-xl", children: [_jsxs("div", { className: "text-center mb-8", children: [_jsxs("h1", { className: "text-3xl font-black text-white tracking-tighter", children: ["WELCOME ", _jsx("span", { className: "text-blue-500", children: "BACK" })] }), _jsx("p", { className: "text-slate-400 text-sm mt-2", children: "Sign in to access your dashboard" })] }), error && (_jsx("div", { className: "bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-4 rounded-xl mb-6", children: error })), _jsxs("form", { onSubmit: handleSubmit, className: "space-y-6", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("label", { className: "text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1", children: "Username" }), _jsx("input", { type: "text", required: true, className: "w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 transition-colors", placeholder: "Enter your username", value: username, onChange: (e) => setUsername(e.target.value) })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("label", { className: "text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1", children: "Password" }), _jsx("input", { type: "password", required: true, className: "w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 transition-colors", placeholder: "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022", value: password, onChange: (e) => setPassword(e.target.value) })] }), _jsx("button", { type: "submit", disabled: loading, className: "w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-black py-4 rounded-xl shadow-lg shadow-blue-500/20 transition-all active:scale-[0.98] uppercase tracking-widest text-xs", children: loading ? 'Authenticating...' : 'Sign In' })] }), _jsxs("div", { className: "mt-8 text-center text-xs text-slate-500 font-bold uppercase tracking-widest", children: ["No account?", ' ', _jsx(Link, { to: "/register", className: "text-blue-500 hover:text-blue-400 transition-colors", children: "Create One" })] })] }) }));
}
