import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { NotificationProvider } from './context/NotificationContext';
import { ThemeProvider } from './context/ThemeContext';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Earnings from './components/Earnings';
import Jobs from './components/Jobs';
import Logs from './components/Logs';
import Settings from './components/Settings';
import GPUDetails from './components/GPUDetails';
import OnboardingWizard from './components/OnboardingWizard';
import About from './components/About';
import Notifications from './components/Notifications';

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [onboardingDone, setOnboardingDone] = useState(() => {
    try { return localStorage.getItem('neuralops-onboarded') === 'true'; } catch { return true; }
  });

  const handleOnboardComplete = () => {
    localStorage.setItem('neuralops-onboarded', 'true');
    setOnboardingDone(true);
  };

  if (!onboardingDone) {
    return (
      <ThemeProvider>
        <OnboardingWizard onComplete={handleOnboardComplete} />
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <NotificationProvider>
        <Router>
          <div className="flex h-screen overflow-hidden">
            <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
            <main className="flex-1 overflow-y-auto bg-dark-bg">
              {/* Title bar */}
              <div className="flex items-center justify-end px-4 py-2 border-b border-dark-border bg-dark-card">
                <Notifications />
              </div>
              <div className="p-6">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/earnings" element={<Earnings />} />
                  <Route path="/jobs" element={<Jobs />} />
                  <Route path="/logs" element={<Logs />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="/gpu" element={<GPUDetails />} />
                  <Route path="/about" element={<About />} />
                </Routes>
              </div>
            </main>
          </div>
        </Router>
      </NotificationProvider>
    </ThemeProvider>
  );
}