import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './components/layout/MainLayout'

function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold text-white">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {['Experiments', 'GPU Marketplace', 'Notes', 'Papers', 'Profile'].map((item) => (
          <div key={item} className="glass p-5 card-hover-effect text-white font-bold">
            {item}
          </div>
        ))}
      </div>
    </div>
  )
}

function Login() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface">
      <div className="glass p-8 w-full max-w-md text-center">
        <h1 className="text-2xl font-bold text-white mb-4">Login</h1>
        <a href="/dashboard" className="text-neon-blue hover:underline">Go to Dashboard</a>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<MainLayout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}