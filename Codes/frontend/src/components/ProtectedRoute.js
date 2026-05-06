import { jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
export default function ProtectedRoute({ children }) {
    const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
    if (!isAuthenticated) {
        return _jsx(Navigate, { to: "/login", replace: true });
    }
    return _jsx(_Fragment, { children: children });
}
