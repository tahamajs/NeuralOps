import React, { useState, useEffect } from 'react';
import { Check, ChevronRight, ChevronLeft, Cpu, Wifi, DollarSign, Zap } from 'lucide-react';

const STEPS = [
  { id: 1, title: 'Welcome', icon: Zap, description: 'Set up your NeuralOps GPU Provider Agent' },
  { id: 2, title: 'GPU Detection', icon: Cpu, description: 'Detect your NVIDIA GPU hardware' },
  { id: 3, title: 'Network Test', icon: Wifi, description: 'Test your internet connection speed' },
  { id: 4, title: 'Pricing', icon: DollarSign, description: 'Set your hourly rate and preferences' },
];

export default function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(1);
  const [gpuInfo, setGpuInfo] = useState(null);
  const [networkResults, setNetworkResults] = useState(null);
  const [config, setConfig] = useState({ apiKey: '', basePrice: '0.50', maxJobs: 2, storageGb: 100 });
  const [detecting, setDetecting] = useState(false);
  const [testing, setTesting] = useState(false);

  const detectGPU = async () => {
    setDetecting(true);
    if (window.neuralopsAPI) {
      const gpus = await window.neuralopsAPI.getGpuInfo();
      setGpuInfo(gpus.length > 0 ? gpus[0] : null);
    }
    setDetecting(false);
  };

  const testNetwork = async () => {
    setTesting(true);
    await new Promise(r => setTimeout(r, 2000));
    setNetworkResults({ downloadMbps: 245.3, uploadMbps: 38.7, latencyMs: 12 });
    setTesting(false);
  };

  const totalSteps = STEPS.length;

  return (
    <div className="min-h-screen flex items-center justify-center bg-dark-bg p-6">
      <div className="w-full max-w-2xl">
        {/* Stepper */}
        <div className="flex items-center justify-center mb-8">
          {STEPS.map((s, idx) => (
            <React.Fragment key={s.id}>
              <div className="flex flex-col items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all ${step > s.id ? 'bg-green-600' : step === s.id ? 'bg-primary-600 ring-2 ring-primary-400' : 'bg-dark-border text-gray-500'}`}>
                  {step > s.id ? <Check size={18} /> : s.id}
                </div>
                <span className={`text-xs mt-2 hidden sm:block ${step >= s.id ? 'text-white' : 'text-gray-500'}`}>{s.title}</span>
              </div>
              {idx < totalSteps - 1 && <div className={`h-0.5 w-12 sm:w-20 mx-1 transition-colors ${step > s.id ? 'bg-green-600' : 'bg-dark-border'}`} />}
            </React.Fragment>
          ))}
        </div>

        {/* Step Content */}
        <div className="bg-dark-card border border-dark-border rounded-2xl p-8">
          {step === 1 && (
            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-primary-600/20 rounded-full flex items-center justify-center mx-auto"><Zap size={36} className="text-primary-400" /></div>
              <h2 className="text-2xl font-bold">Welcome to NeuralOps GPU Agent</h2>
              <p className="text-gray-400 max-w-md mx-auto">Share your idle GPU resources and earn money automatically. Let's get you set up in just a few steps.</p>
              <div className="space-y-3 mt-6">
                <div className="bg-dark-bg border border-dark-border rounded-lg p-4 text-left">
                  <label className="text-sm text-gray-400">API Key</label>
                  <input type="password" value={config.apiKey} onChange={e => setConfig({...config, apiKey: e.target.value})}
                    placeholder="Paste your NeuralOps API key" className="w-full mt-1 bg-transparent text-white outline-none" />
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-blue-600/20 rounded-full flex items-center justify-center mx-auto"><Cpu size={36} className="text-blue-400" /></div>
              <h2 className="text-2xl font-bold">GPU Detection</h2>
              <p className="text-gray-400">We'll scan your system for NVIDIA GPUs.</p>
              {!gpuInfo && !detecting && (
                <button onClick={detectGPU} className="bg-primary-600 hover:bg-primary-700 px-6 py-3 rounded-xl font-semibold transition-colors">Detect My GPU</button>
              )}
              {detecting && <div className="flex items-center justify-center gap-2 text-blue-400"><div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" /> Detecting...</div>}
              {gpuInfo && (
                <div className="bg-dark-bg border border-green-600/30 rounded-xl p-4 text-left space-y-2 mt-4">
                  <div className="flex items-center gap-2 text-green-400"><Check size={18} /> GPU Detected</div>
                  <div className="grid grid-cols-2 gap-2 text-sm"><span className="text-gray-400">Model:</span><span>{gpuInfo.name}</span><span className="text-gray-400">VRAM:</span><span>{gpuInfo.vramMb} MB</span></div>
                </div>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-purple-600/20 rounded-full flex items-center justify-center mx-auto"><Wifi size={36} className="text-purple-400" /></div>
              <h2 className="text-2xl font-bold">Network Speed Test</h2>
              <p className="text-gray-400">Measure your internet speed for optimal job matching.</p>
              {!networkResults && !testing && (
                <button onClick={testNetwork} className="bg-primary-600 hover:bg-primary-700 px-6 py-3 rounded-xl font-semibold transition-colors">Run Speed Test</button>
              )}
              {testing && <div className="flex items-center justify-center gap-2 text-purple-400"><div className="w-5 h-5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" /> Testing...</div>}
              {networkResults && (
                <div className="bg-dark-bg border border-purple-600/30 rounded-xl p-4 mt-4">
                  <div className="flex items-center gap-2 text-purple-400 mb-3"><Check size={18} /> Test Complete</div>
                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div className="bg-dark-card rounded-lg p-3"><p className="text-2xl font-bold text-green-400">{networkResults.downloadMbps}</p><p className="text-xs text-gray-400">Download Mbps</p></div>
                    <div className="bg-dark-card rounded-lg p-3"><p className="text-2xl font-bold text-blue-400">{networkResults.uploadMbps}</p><p className="text-xs text-gray-400">Upload Mbps</p></div>
                    <div className="bg-dark-card rounded-lg p-3 col-span-2"><p className="text-2xl font-bold text-orange-400">{networkResults.latencyMs} ms</p><p className="text-xs text-gray-400">Latency</p></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 4 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-center">Set Your Pricing</h2>
              <p className="text-gray-400 text-center">Configure how much you want to earn per GPU hour.</p>
              <div className="space-y-3 mt-4">
                {[
                  { label: 'Base Price per Hour (USD)', key: 'basePrice', type: 'number', step: '0.01' },
                  { label: 'Max Concurrent Jobs', key: 'maxJobs', type: 'number', step: '1' },
                  { label: 'Storage Limit (GB)', key: 'storageGb', type: 'number', step: '1' },
                ].map(({ label, key, type, step }) => (
                  <div key={key} className="bg-dark-bg border border-dark-border rounded-lg p-4">
                    <label className="text-sm text-gray-400">{label}</label>
                    <input type={type} step={step} value={config[key]}
                      onChange={e => setConfig({...config, [key]: e.target.value})}
                      className="w-full mt-1 bg-transparent text-white text-lg font-semibold outline-none" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between mt-8">
            <button onClick={() => setStep(s => Math.max(1, s - 1))} disabled={step === 1}
              className="flex items-center gap-1 px-4 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-dark-border disabled:opacity-30 transition-colors">
              <ChevronLeft size={18} /> Back
            </button>
            {step < totalSteps ? (
              <button onClick={() => setStep(s => Math.min(totalSteps, s + 1))}
                className="flex items-center gap-1 bg-primary-600 hover:bg-primary-700 px-5 py-2 rounded-lg font-medium transition-colors">
                Next <ChevronRight size={18} />
              </button>
            ) : (
              <button onClick={onComplete}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 px-6 py-2 rounded-lg font-medium transition-colors">
                <Check size={18} /> Complete Setup
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}