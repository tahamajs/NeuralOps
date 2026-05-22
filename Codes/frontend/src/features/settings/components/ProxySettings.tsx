// src/features/settings/components/ProxySettings.tsx
import { useState, useEffect } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button, Input } from '../../../components/ui'
import { getProxyUrl, setProxyUrl } from '../../../api/client'
import { useToast } from '../../../components/ui/Toast'

export function ProxySettings() {
  const [proxyUrl, setProxyUrlState] = useState(getProxyUrl())
  const [enabled, setEnabled] = useState(!!getProxyUrl())
  const { success, error } = useToast()

  const handleSave = () => {
    if (enabled && proxyUrl) {
      setProxyUrl(proxyUrl)
      success('Proxy enabled', `Requests will go through ${proxyUrl}`)
    } else {
      setProxyUrl('')
      success('Proxy disabled', 'Direct API calls will be used')
    }
    window.location.reload() // to reinitialize axios instance
  }

  useEffect(() => {
    setEnabled(!!getProxyUrl())
  }, [])

  return (
    <TerminalCard className="p-4 space-y-4">
      <h2 className="terminal-heading">Proxy Configuration</h2>
      <p className="text-sm text-slate-400">Route all API requests through a proxy server (e.g., to bypass CORS or use a different backend).</p>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        <span className="text-sm">Enable Proxy</span>
      </label>
      {enabled && (
        <Input
          label="Proxy URL"
          value={proxyUrl}
          onChange={(e) => setProxyUrlState(e.target.value)}
          placeholder="https://my-proxy.example.com/"
        />
      )}
      <Button onClick={handleSave}>Save Proxy Settings</Button>
    </TerminalCard>
  )
}