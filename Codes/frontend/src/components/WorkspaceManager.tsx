import { useState } from 'react'
import { useWindowStore } from '../store/windowStore'
import { Save, FolderOpen } from 'lucide-react'

export function WorkspaceManager() {
  const { saveLayout, loadLayout } = useWindowStore()
  const [name, setName] = useState('')
  return (
    <div className="fixed top-4 right-4 z-[100] glass-terminal rounded-lg p-2 flex gap-2">
      <input
        type="text"
        placeholder="Layout name"
        className="bg-black/30 border border-white/10 rounded px-2 py-1 text-sm text-white"
        value={name}
        onChange={e => setName(e.target.value)}
      />
      <button onClick={() => name && saveLayout(name)} className="p-1"><Save size={16} /></button>
      <button onClick={() => name && loadLayout(name)} className="p-1"><FolderOpen size={16} /></button>
    </div>
  )
}