import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Earnings from './components/Earnings';
import Jobs from './components/Jobs';
import Logs from './components/Logs';
import Settings from './components/Settings';

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <Router>
      <div className="flex h-screen overflow-hidden">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
        <main className="flex-1 overflow-y-auto p-6 bg-dark-bg">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/earnings" element={<Earnings />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}